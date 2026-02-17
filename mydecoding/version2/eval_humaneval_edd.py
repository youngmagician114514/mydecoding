import argparse
import os
import random
import time

import jsonlines
import numpy as np
import torch
from datasets import Dataset
from tqdm import tqdm
from transformers import AutoConfig, AutoTokenizer

from model import Effective_Draft_Decoder
from modeling_llama import LlamaForCausalLM

os.environ["NCCL_P2P_DISABLE"] = "1"
os.environ["NCCL_IB_DISABLE"] = "1"

seed_val = 888
np.random.seed(seed_val)
torch.manual_seed(seed_val)
torch.cuda.manual_seed_all(seed_val)
random.seed(seed_val)


def _parse_layers(s: str):
    if s is None:
        return None
    s = s.strip()
    if not s:
        return None
    s = s.replace("[", "").replace("]", "")
    layers = [int(x) for x in s.split(",") if x.strip()]
    return layers if layers else None


def _select_encoder_out(outputs, hidden_layer: int, layer_indices):
    if layer_indices is None:
        return outputs.hidden_states[hidden_layer]  # (B, T, H)
    return torch.stack([outputs.hidden_states[i] for i in layer_indices], dim=1)  # (B, K, T, H)


def _select_accepted_layers(outputs, hidden_layer: int, layer_indices, acc_ids):
    keep_pos = [0] + acc_ids
    if layer_indices is None:
        return outputs.hidden_states[hidden_layer][:, keep_pos, :].unsqueeze(1)  # (B,1,adv,H)

    cur_layers = torch.stack([outputs.hidden_states[i] for i in layer_indices], dim=1)
    return cur_layers[:, :, keep_pos, :]  # (B,K,adv,H)


parser = argparse.ArgumentParser(description="EDD-only HumanEval verification (no PCT)")
parser.add_argument("--num_layers", type=int, default=1, help="Draft decoder layers")
parser.add_argument("--model_checkpoint", type=str, default="./llama-2-7b-chat-hf", help="Target LLM checkpoint")
parser.add_argument("--data_dir", type=str, default="./data/humaneval.jsonl", help="Path to humaneval.jsonl (prompt/pred)")
parser.add_argument("--draft_model_checkpoint", type=str, default="./draft_model.pt", help="Path to draft model checkpoint (.pt)")
parser.add_argument("--hidden_layer", type=int, default=-4, help="Single hidden layer index (used when --fusion_layers is empty)")
parser.add_argument("--fusion_layers", type=str, default="", help="Comma-separated layer indices, e.g. -3,-2,-1")
parser.add_argument("--draft_len", type=int, default=5, help="Draft tokens per iteration")
parser.add_argument("--dtype", type=str, default="bf16", choices=["bf16", "fp16", "fp32"], help="Inference dtype")
parser.add_argument("--tokenizer_dir", type=str, default="", help="Optional tokenizer cache directory")
args = parser.parse_args()

num_layers = args.num_layers
model_checkpoint = args.model_checkpoint
data_dir = args.data_dir
draft_model_model_checkpoint = args.draft_model_checkpoint
hidden_layer = args.hidden_layer
fusion_layers = _parse_layers(args.fusion_layers)
draft_len = args.draft_len


def _resolve_dtype(name: str):
    if name == "bf16":
        return torch.bfloat16
    if name == "fp16":
        return torch.float16
    return torch.float32


def _load_tokenizer(model_ckpt: str, tok_dir: str):
    tokenizer_kwargs = {"use_fast": True, "padding_side": "left", "trust_remote_code": True}
    if tok_dir and os.path.isdir(tok_dir):
        tokenizer = AutoTokenizer.from_pretrained(tok_dir, **tokenizer_kwargs)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_ckpt, **tokenizer_kwargs)
        if tok_dir:
            os.makedirs(tok_dir, exist_ok=True)
            tokenizer.save_pretrained(tok_dir)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    return tokenizer


def load_data(tokenizer: AutoTokenizer):
    def preprocess_function(examples):
        tokenizer.padding_side = "left"
        prompt = tokenizer(examples["prompt"])
        tokenizer.padding_side = "right"
        pred = tokenizer(examples["pred"], add_special_tokens=False)
        return {
            "input_ids": prompt["input_ids"],
            "labels": pred["input_ids"],
            "input_attention_mask": prompt["attention_mask"],
            "labels_attention_mask": pred["attention_mask"],
        }

    test_data = []
    with jsonlines.open(data_dir) as f:
        for line in f:
            test_data.append({"prompt": line["prompt"], "pred": line["pred"]})

    dataset_test = Dataset.from_list(test_data)
    print(dataset_test)
    tokenized = dataset_test.map(
        preprocess_function,
        batched=True,
        num_proc=16,
        remove_columns=["prompt", "pred"],
    )
    tokenized.set_format("torch")
    return tokenized


@torch.no_grad()
def edd_greedy_draft(
    edd: Effective_Draft_Decoder,
    encoder_self: torch.Tensor,  # (B,T,H)
    memory_base: torch.Tensor,  # (B,S,H)
    start_token: torch.Tensor,  # (B,1)
    max_new_tokens: int,
    eos_id: int,
):
    # Keep the same decoding skeleton as model.generate(), but force greedy single-path expansion.
    decoder_inp_token = start_token
    past_key_values = None

    for _ in range(max_new_tokens):
        if past_key_values is None:
            dec_inp = torch.cat([encoder_self, edd.embedding_layer(decoder_inp_token)], dim=1)
            hidden_states = edd.decoder(dec_inp, use_cache=True)
            past_key_values = tuple(kv[:, :, : encoder_self.shape[1], :] for kv in hidden_states[1])
        else:
            cur_past = [kv.repeat(decoder_inp_token.shape[0], 1, 1, 1) for kv in past_key_values]
            position_ids = torch.arange(
                encoder_self.shape[1],
                encoder_self.shape[1] + decoder_inp_token.shape[1],
                dtype=torch.long,
                device=encoder_self.device,
            )[None, :]
            dec_inp = edd.embedding_layer(decoder_inp_token)
            hidden_states = edd.decoder(dec_inp, past_key_value=cur_past, position_ids=position_ids)

        if memory_base.shape[0] != decoder_inp_token.shape[0]:
            memory = memory_base.repeat(decoder_inp_token.shape[0], 1, 1)
        else:
            memory = memory_base

        token_state = hidden_states[0][:, -1:, :]
        cross_out, _ = edd.cross_attn(
            edd.cross_norm(token_state), memory, memory, attn_mask=None, need_weights=False
        )
        token_state = edd.norm(token_state + cross_out)
        logits = edd.lm_head(token_state)

        next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
        decoder_inp_token = torch.cat([decoder_inp_token, next_token], dim=1)

        if (next_token == eos_id).all():
            break

    return decoder_inp_token[:, 1:]


@torch.no_grad()
def test(batch, llm_model, draft_model, eos_id: int, hidden_layer: int, layer_indices):
    iter_num, sum_acc_num, iter_num_small = 0, 0, 0

    for i in tqdm(range(len(batch)), desc="Testing Progress"):
        input_ids = batch[i]["input_ids"].unsqueeze(0).cuda()
        attention_mask_input = batch[i]["input_attention_mask"].unsqueeze(0).cuda()
        label_num = batch[i]["labels_attention_mask"].sum().item()

        cur_num, past_key_values, pred_ids = 0, None, None

        while cur_num < label_num:
            iter_num += 1

            if past_key_values is None:
                outputs = llm_model(input_ids, attention_mask=attention_mask_input, output_hidden_states=True)
                past_key_values = outputs.past_key_values
                encoder_out = _select_encoder_out(outputs, hidden_layer, layer_indices)
                slot_cache = draft_model.init_slot_cache(
                    encoder_out,
                    slot_size=draft_len,
                    apply_dropout=False,
                )
                new_token_1 = outputs.logits[:, -1:, :].argmax(dim=-1)
                cur_num += 1

            last_past_key_values_len = past_key_values[0][0].shape[2]
            encoder_self, memory_base = draft_model.split_from_slot_cache(slot_cache)

            draft_tokens = edd_greedy_draft(
                draft_model,
                encoder_self=encoder_self,
                memory_base=memory_base,
                start_token=new_token_1,
                max_new_tokens=draft_len,
                eos_id=eos_id,
            )
            dlen = draft_tokens.shape[1]
            if dlen == 0:
                draft_tokens = new_token_1.new_full((1, 1), eos_id)
                dlen = 1

            outputs = llm_model(
                torch.cat([new_token_1, draft_tokens], dim=1),
                attention_mask=torch.cat([attention_mask_input, torch.ones(1, 1 + dlen).cuda()], dim=1),
                past_key_values=past_key_values,
                output_hidden_states=True,
            )

            llm_pred = outputs.logits[:, -(dlen + 1) :, :].argmax(dim=-1)
            llm_next = llm_pred[:, 0]
            llm_verify = llm_pred[:, 1:]

            cur_llm_p = llm_next.item()
            acc_token = [cur_llm_p]
            acc_ids = []
            acc_num = 1

            llm_verify_cpu = llm_verify.cpu()
            draft_cpu = draft_tokens.cpu()

            for j in range(dlen):
                if cur_llm_p == int(draft_cpu[0, j]):
                    acc_ids.append(j + 1)
                    cur_llm_p = int(llm_verify_cpu[0, j])
                    acc_token.append(cur_llm_p)
                    acc_num += 1
                else:
                    break

            cur_num += acc_num
            sum_acc_num += acc_num
            iter_num_small += dlen

            acc_token_tensor = torch.tensor(acc_token, device="cuda", dtype=torch.long).unsqueeze(0)
            pred_ids = (
                torch.cat([new_token_1, acc_token_tensor], dim=1)
                if pred_ids is None
                else torch.cat([pred_ids, acc_token_tensor], dim=1)
            )

            if (pred_ids == eos_id).any():
                break

            past_key_values = [list(x) for x in outputs.past_key_values]
            for x in range(len(past_key_values)):
                for y in range(len(past_key_values[x])):
                    keep_idx = [idx + last_past_key_values_len for idx in acc_ids]
                    past_key_values[x][y] = torch.cat(
                        [
                            past_key_values[x][y][:, :, : last_past_key_values_len + 1, :],
                            past_key_values[x][y][:, :, keep_idx, :]
                            if len(keep_idx) > 0
                            else past_key_values[x][y][:, :, :0, :],
                        ],
                        dim=2,
                    )
            past_key_values = tuple(tuple(x) for x in past_key_values)

            accepted_layers = _select_accepted_layers(outputs, hidden_layer, layer_indices, acc_ids)
            slot_cache = draft_model.update_slot_cache(slot_cache, accepted_layers, apply_dropout=False)

            new_token_1 = pred_ids[:, -1:]
            attention_mask_input = torch.cat([attention_mask_input, torch.ones(1, acc_num).cuda()], dim=1)

    print(
        "avg accept rate:",
        sum_acc_num / max(iter_num_small, 1),
        "avg accepted length:",
        sum_acc_num / max(iter_num, 1),
    )


if __name__ == "__main__":
    tokenizer = _load_tokenizer(model_checkpoint, args.tokenizer_dir.strip())
    torch_dtype = _resolve_dtype(args.dtype)

    model = LlamaForCausalLM.from_pretrained(
        model_checkpoint,
        torch_dtype=torch_dtype,
        device_map="auto",
    )
    model = model.cuda().eval()

    config = AutoConfig.from_pretrained(model_checkpoint)
    Draft_Decoder = Effective_Draft_Decoder(
        config.hidden_size,
        config.hidden_size * 2,
        config.num_attention_heads,
        num_layers,
        config,
    )
    Draft_Decoder.lm_head.load_state_dict(model.lm_head.state_dict())
    Draft_Decoder.embedding_layer.load_state_dict(model.model.embed_tokens.state_dict())
    Draft_Decoder = Draft_Decoder.to(dtype=torch_dtype)
    Draft_Decoder.load_state_dict(torch.load(draft_model_model_checkpoint, map_location="cpu"))
    Draft_Decoder = Draft_Decoder.cuda().eval()

    test_tokenized_datasets = load_data(tokenizer)

    print("start testing (EDD-only, no PCT)...")
    start_time = time.time()
    test(test_tokenized_datasets, model, Draft_Decoder, tokenizer.eos_token_id, hidden_layer, fusion_layers)
    end_time = time.time()
    print(f"{end_time - start_time:.4f} seconds")
