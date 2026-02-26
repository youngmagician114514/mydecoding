import argparse
import os
import random
import time
from typing import Dict, List, Optional, Tuple

import jsonlines
import numpy as np
import torch
from datasets import Dataset
from tqdm import tqdm
from transformers import AutoConfig, AutoTokenizer

from model import Effective_Draft_Decoder, Flatten_tree
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


def _select_encoder_out(outputs, hidden_layer: int, layer_indices):
    if layer_indices is None:
        return outputs.hidden_states[hidden_layer]  # (B,T,H)
    return torch.stack([outputs.hidden_states[i] for i in layer_indices], dim=1)  # (B,K,T,H)


def _select_accepted_layers(outputs, hidden_layer: int, layer_indices, acc_ids):
    keep_pos = [0] + acc_ids
    if layer_indices is None:
        return outputs.hidden_states[hidden_layer][:, keep_pos, :].unsqueeze(1)  # (B,1,adv,H)

    cur_layers = torch.stack([outputs.hidden_states[i] for i in layer_indices], dim=1)
    return cur_layers[:, :, keep_pos, :]  # (B,K,adv,H)


def load_data(tokenizer: AutoTokenizer, data_dir: str, num_proc: int):
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
    test_tokenized_datasets = dataset_test.map(
        preprocess_function,
        batched=True,
        num_proc=num_proc,
        remove_columns=["prompt", "pred"],
    )
    test_tokenized_datasets.set_format("torch")
    return test_tokenized_datasets


class NGramCache:
    """
    Multi-level n-gram cache:
      D_k: suffix(k) -> next token
    Uses per-key counts and keeps the most frequent next token.
    """

    def __init__(self, max_order: int = 4):
        self.max_order = max(1, int(max_order))
        self.best_tables: List[Dict[Tuple[int, ...], int]] = [{} for _ in range(self.max_order)]
        self.count_tables: List[Dict[Tuple[int, ...], Dict[int, int]]] = [{} for _ in range(self.max_order)]

    def add_transition(self, prefix_tokens: List[int], next_token: int):
        plen = len(prefix_tokens)
        if plen <= 0:
            return
        next_token = int(next_token)
        max_k = min(self.max_order, plen)
        for k in range(1, max_k + 1):
            key = tuple(prefix_tokens[-k:])
            bucket = self.count_tables[k - 1].setdefault(key, {})
            new_count = bucket.get(next_token, 0) + 1
            bucket[next_token] = new_count

            prev_best = self.best_tables[k - 1].get(key, None)
            if prev_best is None or new_count >= bucket.get(prev_best, 0):
                self.best_tables[k - 1][key] = next_token

    def add_transition_from_sequence(self, tokens: List[int], next_idx: int):
        if next_idx <= 0 or next_idx >= len(tokens):
            return
        next_token = int(tokens[next_idx])
        max_k = min(self.max_order, next_idx)
        for k in range(1, max_k + 1):
            key = tuple(tokens[next_idx - k : next_idx])
            bucket = self.count_tables[k - 1].setdefault(key, {})
            new_count = bucket.get(next_token, 0) + 1
            bucket[next_token] = new_count

            prev_best = self.best_tables[k - 1].get(key, None)
            if prev_best is None or new_count >= bucket.get(prev_best, 0):
                self.best_tables[k - 1][key] = next_token

    def ingest_sequence(self, tokens: List[int]):
        if len(tokens) < 2:
            return
        for idx in range(1, len(tokens)):
            self.add_transition_from_sequence(tokens, idx)

    def longest_hit(self, prefix_tokens: List[int]) -> Tuple[int, Optional[int]]:
        plen = len(prefix_tokens)
        if plen <= 0:
            return 0, None
        max_k = min(self.max_order, plen)
        for k in range(max_k, 0, -1):
            key = tuple(prefix_tokens[-k:])
            pred = self.best_tables[k - 1].get(key, None)
            if pred is not None:
                return k, pred
        return 0, None


@torch.no_grad()
def ngram_enhanced_generate(
    draft_model: Effective_Draft_Decoder,
    encoder_self: torch.Tensor,  # (B,T,H)
    memory_base: torch.Tensor,  # (B,S,H)
    decoder_inp_token: torch.Tensor,  # (B,1), start token
    history_tail: List[int],  # committed tail including start token
    ngram_cache: Optional[NGramCache],
    max_length: int = 10,
    top_k: int = 5,
    threshold: float = 0.036,
    beta_max: float = 0.25,
    enhance_top_paths: int = 64,
    beam_cap: int = 256,
):
    if max_length <= 0:
        raise ValueError("max_length must be > 0")

    cur_p = torch.tensor([1.0], device=encoder_self.device).unsqueeze(-1)
    all_candidates_new = []
    past_key_values = None

    for _ in range(max_length):
        if past_key_values is None:
            dec_inp = torch.cat([encoder_self, draft_model.embedding_layer(decoder_inp_token)], dim=1)
            hidden_states = draft_model.decoder(dec_inp, use_cache=True)
            past_key_values = tuple(kv[:, :, : encoder_self.shape[1], :] for kv in hidden_states[1])
        else:
            cur_past_key_values = [kv.repeat(decoder_inp_token.shape[0], 1, 1, 1) for kv in past_key_values]
            position_ids = torch.arange(
                encoder_self.shape[1],
                encoder_self.shape[1] + decoder_inp_token.shape[1],
                dtype=torch.long,
                device=encoder_self.device,
            )[None, :]
            dec_inp = draft_model.embedding_layer(decoder_inp_token)
            hidden_states = draft_model.decoder(dec_inp, past_key_value=cur_past_key_values, position_ids=position_ids)

        if memory_base.shape[0] != decoder_inp_token.shape[0]:
            memory = memory_base.repeat(decoder_inp_token.shape[0], 1, 1)
        else:
            memory = memory_base

        token_state = hidden_states[0][:, -1:, :]
        if memory.shape[1] > 0:
            cross_out, _ = draft_model.cross_attn(
                draft_model.cross_norm(token_state),
                memory,
                memory,
                attn_mask=None,
                need_weights=False,
            )
            token_state = draft_model.norm(token_state + cross_out)
        else:
            token_state = draft_model.norm(token_state)

        logits = draft_model.lm_head(token_state)
        top_scores, top_indices = logits[:, -1, :].softmax(-1).topk(top_k, dim=-1)

        if ngram_cache is not None and beta_max > 0:
            num_paths = decoder_inp_token.shape[0]
            if enhance_top_paths > 0 and num_paths > enhance_top_paths:
                selected_rows = torch.topk(cur_p.squeeze(-1), enhance_top_paths).indices.tolist()
            else:
                selected_rows = range(num_paths)

            for row in selected_rows:
                row_tokens = decoder_inp_token[row].tolist()
                prefix_tail = history_tail
                if len(row_tokens) > 1:
                    prefix_tail = (prefix_tail + row_tokens[1:])[-ngram_cache.max_order :]

                hit_k, pred_tok = ngram_cache.longest_hit(prefix_tail)
                if hit_k <= 0 or pred_tok is None:
                    continue

                hit_pos = (top_indices[row] == int(pred_tok)).nonzero(as_tuple=False)
                if hit_pos.numel() == 0:
                    continue

                col = int(hit_pos[0].item())
                beta = float(beta_max) * (float(hit_k) / float(ngram_cache.max_order))
                p = top_scores[row, col]
                top_scores[row, col] = p + beta * (1.0 - p)

        cur_p = cur_p * top_scores
        mask = cur_p > threshold

        alive = int(mask.sum().item())
        if beam_cap > 0 and alive > beam_cap:
            flat_scores = cur_p.masked_fill(~mask, -1.0).reshape(-1)
            keep_idx = torch.topk(flat_scores, k=beam_cap).indices
            new_mask = torch.zeros_like(mask, dtype=torch.bool)
            new_mask.reshape(-1)[keep_idx] = True
            mask = new_mask

        if mask.sum().item() == 0:
            decoder_inp_token = torch.cat([decoder_inp_token, top_indices[:, 0].unsqueeze(-1)], dim=1)
            all_candidates_new.append(decoder_inp_token[:, 1:])
            break

        decoder_inp_token = torch.cat(
            [
                decoder_inp_token.unsqueeze(1).repeat(1, top_k, 1),
                top_indices.unsqueeze(-1),
            ],
            dim=2,
        )
        dropped = decoder_inp_token[~mask]
        if dropped.shape[0] > 0:
            all_candidates_new.append(dropped[:, 1:])

        decoder_inp_token, cur_p = decoder_inp_token[mask], cur_p[mask].unsqueeze(-1)

    if len(all_candidates_new) == 0:
        all_candidates_new.append(decoder_inp_token[:, 1:])

    seq, attention_mask, division, position_ids = Flatten_tree(all_candidates_new)
    return seq.unsqueeze(0), attention_mask, division, position_ids


@torch.no_grad()
def test(
    batch,
    llm_model,
    small_model,
    eos_id: int,
    hidden_layer: int,
    layer_indices,
    threshold: float,
    top_k: int,
    draft_max_len: int,
    n: int,
    ngram_beta_max: float,
    ngram_top_paths: int,
    beam_cap: int,
    disable_ngram: bool,
    seed_from_prompt: bool,
):
    iter_num, sum_acc_num, iter_num_small = 0, 0, 0
    decode_time_s, decode_tokens = 0.0, 0

    for i in tqdm(range(len(batch)), desc="Testing Progress"):
        input_ids = batch[i]["input_ids"].unsqueeze(0).cuda()
        attention_mask_input = batch[i]["input_attention_mask"].unsqueeze(0).cuda()
        label_num = int(batch[i]["labels_attention_mask"].sum().item())
        cur_num, past_key_values, pred_ids = 0, None, None
        sample_decode_started = False
        sample_decode_start = 0.0

        ngram_cache = None if disable_ngram else NGramCache(max_order=n)
        committed_tokens = input_ids[0].tolist()
        if ngram_cache is not None and seed_from_prompt:
            ngram_cache.ingest_sequence(committed_tokens)

        while cur_num < label_num:
            iter_num += 1

            if past_key_values is None:
                outputs = llm_model(input_ids, attention_mask=attention_mask_input, output_hidden_states=True)
                past_key_values = outputs.past_key_values
                encoder_out = _select_encoder_out(outputs, hidden_layer, layer_indices)
                slot_cache = small_model.init_slot_cache(
                    encoder_out,
                    slot_size=small_model.slot_size,
                    apply_dropout=False,
                )
                new_token_1 = outputs.logits[:, -1:, :].argmax(dim=-1)
                cur_num += 1

                t0 = int(new_token_1.item())
                if ngram_cache is not None:
                    ngram_cache.add_transition(committed_tokens, t0)
                committed_tokens.append(t0)

            if not sample_decode_started:
                torch.cuda.synchronize()
                sample_decode_start = time.perf_counter()
                sample_decode_started = True

            last_past_key_values_len = past_key_values[0][0].shape[2]
            encoder_self, memory_base = small_model.split_from_slot_cache(slot_cache)

            history_tail = committed_tokens[-n:] if n > 0 else committed_tokens
            small_pred, verify_mask, root, verify_position_ids = ngram_enhanced_generate(
                draft_model=small_model,
                encoder_self=encoder_self,
                memory_base=memory_base,
                decoder_inp_token=new_token_1,
                history_tail=history_tail,
                ngram_cache=ngram_cache,
                max_length=draft_max_len,
                top_k=top_k,
                threshold=threshold,
                beta_max=ngram_beta_max,
                enhance_top_paths=ngram_top_paths,
                beam_cap=beam_cap,
            )
            if small_pred.shape[1] == 0:
                break

            outputs = llm_model(
                torch.cat([new_token_1, small_pred], dim=1),
                attention_mask=torch.cat([attention_mask_input, torch.ones(1, 1 + small_pred.shape[1]).cuda()], dim=1),
                past_key_values=past_key_values,
                output_hidden_states=True,
                verify_mask=verify_mask,
                verify_position_ids=verify_position_ids,
            )

            llm_pred = outputs.logits[:, -small_pred.shape[1] - 1 :, :].argmax(dim=-1)
            new_token_2, llm_verify_token = llm_pred[:, 0], llm_pred[:, 1:]

            max_draft_len = int(verify_position_ids.max().item()) if verify_position_ids.numel() > 0 else 0
            llm_verify_token, cur_llm_p = llm_verify_token.cpu(), int(new_token_2.item())
            acc_token, acc_ids, acc_num = [cur_llm_p], [], 1
            cur_node = root
            for _ in range(max_draft_len):
                cur_node = cur_node.get_child(cur_llm_p)
                if cur_node is not None:
                    acc_num += 1
                    cur_llm_p = int(llm_verify_token[0, cur_node.idx])
                    acc_token.append(cur_llm_p)
                    acc_ids.append(cur_node.idx + 1)
                else:
                    break

            cur_num += acc_num
            sum_acc_num += acc_num
            iter_num_small += max_draft_len
            decode_tokens += acc_num
            cur_tokens = torch.tensor(acc_token, device="cuda", dtype=torch.long).unsqueeze(0)
            pred_ids = (
                torch.cat([new_token_1, cur_tokens], dim=1)
                if pred_ids is None
                else torch.cat([pred_ids, cur_tokens], dim=1)
            )

            for t in acc_token:
                if ngram_cache is not None:
                    ngram_cache.add_transition(committed_tokens, int(t))
                committed_tokens.append(int(t))

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
            slot_cache = small_model.update_slot_cache(slot_cache, accepted_layers, apply_dropout=False)

            new_token_1 = pred_ids[:, -1:]
            attention_mask_input = torch.cat([attention_mask_input, torch.ones(1, acc_num).cuda()], dim=1)

        if sample_decode_started:
            torch.cuda.synchronize()
            decode_time_s += time.perf_counter() - sample_decode_start

    decode_tps = decode_tokens / max(decode_time_s, 1e-6)
    print(
        "avg accept rate:",
        sum_acc_num / max(iter_num_small, 1),
        "avg accepted length:",
        sum_acc_num / max(iter_num, 1),
    )
    print("decode time(s):", f"{decode_time_s:.4f}")
    print("decode tokens:", decode_tokens)
    print("decode tokens/s:", f"{decode_tps:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="HumanEval with n-gram path enhancement")
    parser.add_argument("--num_layers", type=int, default=1, help="Draft decoder layers")
    parser.add_argument("--model_checkpoint", type=str, required=True, help="Target LLM checkpoint")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to humaneval.jsonl (prompt/pred)")
    parser.add_argument("--draft_model_checkpoint", type=str, required=True, help="Path to draft model checkpoint")
    parser.add_argument("--hidden_layer", type=int, default=-4, help="Single hidden layer index")
    parser.add_argument("--fusion_layers", type=str, default="", help="Comma-separated layer indices")
    parser.add_argument("--threshold", type=float, default=0.036, help="Draft pruning threshold")
    parser.add_argument("--top_k", type=int, default=5, help="Top-k branch expansion in draft tree")
    parser.add_argument("--draft_max_len", type=int, default=10, help="Max draft depth per iteration")
    parser.add_argument("--n", type=int, default=4, help="Max n-gram order (paper N)")
    parser.add_argument("--ngram_order", type=int, default=None, help="Deprecated alias of --n")
    parser.add_argument("--ngram_beta_max", type=float, default=0.25, help="Max boost strength")
    parser.add_argument("--ngram_top_paths", type=int, default=64, help="Apply n-gram boost to top-M active paths")
    parser.add_argument("--beam_cap", type=int, default=256, help="Max active branches after thresholding")
    parser.add_argument("--disable_ngram", action="store_true", help="Disable n-gram path enhancement")
    parser.add_argument("--no_prompt_seed", action="store_true", help="Do not seed n-gram cache from prompt")
    parser.add_argument("--dtype", type=str, default="bf16", choices=["bf16", "fp16", "fp32"], help="Inference dtype")
    parser.add_argument("--tokenizer_dir", type=str, default="", help="Optional tokenizer cache directory")
    parser.add_argument("--num_proc", type=int, default=16, help="Num workers for datasets.map")
    args = parser.parse_args()
    if args.ngram_order is not None:
        args.n = int(args.ngram_order)

    tokenizer = _load_tokenizer(args.model_checkpoint, args.tokenizer_dir.strip())
    torch_dtype = _resolve_dtype(args.dtype)
    layer_indices = _parse_layers(args.fusion_layers)

    model = LlamaForCausalLM.from_pretrained(
        args.model_checkpoint,
        torch_dtype=torch_dtype,
        device_map="auto",
    )

    config = AutoConfig.from_pretrained(args.model_checkpoint)
    draft_decoder = Effective_Draft_Decoder(
        config.hidden_size,
        config.hidden_size * 2,
        config.num_attention_heads,
        args.num_layers,
        config,
    )
    draft_decoder.lm_head.load_state_dict(model.lm_head.state_dict())
    draft_decoder.embedding_layer.load_state_dict(model.model.embed_tokens.state_dict())
    draft_decoder = draft_decoder.to(dtype=torch_dtype)
    draft_decoder.load_state_dict(torch.load(args.draft_model_checkpoint, map_location="cpu"))

    model = model.cuda().eval()
    draft_decoder = draft_decoder.cuda().eval()

    test_tokenized_datasets = load_data(tokenizer, args.data_dir, args.num_proc)

    print("start testing (n-gram path enhancement)...")
    with torch.no_grad():
        test(
            batch=test_tokenized_datasets,
            llm_model=model,
            small_model=draft_decoder,
            eos_id=tokenizer.eos_token_id,
            hidden_layer=args.hidden_layer,
            layer_indices=layer_indices,
            threshold=args.threshold,
            top_k=args.top_k,
            draft_max_len=args.draft_max_len,
            n=args.n,
            ngram_beta_max=args.ngram_beta_max,
            ngram_top_paths=args.ngram_top_paths,
            beam_cap=args.beam_cap,
            disable_ngram=args.disable_ngram,
            seed_from_prompt=(not args.no_prompt_seed),
        )
