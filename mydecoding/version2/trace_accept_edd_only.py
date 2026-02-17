import argparse
import os
import random

import jsonlines
import numpy as np
import torch
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


def _tok_repr(tokenizer: AutoTokenizer, tid: int) -> str:
    return repr(tokenizer.decode([tid]))


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


@torch.no_grad()
def edd_greedy_draft_tokens(
    edd: Effective_Draft_Decoder,
    encoder_self: torch.Tensor,  # (1,T,H)
    memory_base: torch.Tensor,  # (1,S,H)
    start_token: torch.Tensor,  # (1,1)
    steps: int,
    eos_id: int,
) -> torch.Tensor:
    decoder_inp_token = start_token
    past_key_values = None

    for _ in range(steps):
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
        logits = edd.lm_head(token_state)[:, -1, :]

        next_token = logits.argmax(dim=-1, keepdim=True)
        decoder_inp_token = torch.cat([decoder_inp_token, next_token], dim=1)

        if int(next_token.item()) == eos_id:
            break

    return decoder_inp_token[:, 1:]


def load_prompt_from_humaneval(path: str, index: int) -> str:
    with jsonlines.open(path) as r:
        for i, ex in enumerate(r):
            if i == index:
                return ex["prompt"]
    raise ValueError(f"index={index} out of range for {path}")


@torch.no_grad()
def run_trace(
    tokenizer: AutoTokenizer,
    llm: LlamaForCausalLM,
    edd: Effective_Draft_Decoder,
    prompt: str,
    hidden_layer: int,
    layer_indices,
    draft_len: int,
    max_new_tokens: int,
):
    device = "cuda"

    print("\n========== PROMPT ==========")
    print(prompt)

    tokenizer.padding_side = "left"
    enc = tokenizer(prompt, return_tensors="pt")
    input_ids = enc["input_ids"].to(device)
    attn = enc["attention_mask"].to(device)

    out0 = llm(input_ids, attention_mask=attn, output_hidden_states=True)
    past_kv = out0.past_key_values
    encoder_out = _select_encoder_out(out0, hidden_layer, layer_indices)
    slot_cache = edd.init_slot_cache(encoder_out, slot_size=draft_len, apply_dropout=False)
    new_token_1 = out0.logits[:, -1:, :].argmax(dim=-1)

    generated = [int(new_token_1.item())]
    cur_gen = 1

    print("\n[init] new_token_1 =", _tok_repr(tokenizer, generated[-1]), f"(id={generated[-1]})")
    if generated[-1] == tokenizer.eos_token_id or cur_gen >= max_new_tokens:
        print("\n[stop] reached eos or max_new_tokens.")
        print("========== OUTPUT ==========")
        print(tokenizer.decode(generated))
        return

    it = 0
    sum_acc_num = 0
    iter_num = 0
    iter_num_small = 0

    while cur_gen < max_new_tokens:
        it += 1
        last_past_len = past_kv[0][0].shape[2]
        encoder_self, memory_base = edd.split_from_slot_cache(slot_cache)

        draft_tokens = edd_greedy_draft_tokens(
            edd=edd,
            encoder_self=encoder_self,
            memory_base=memory_base,
            start_token=new_token_1,
            steps=draft_len,
            eos_id=tokenizer.eos_token_id,
        )
        dlen = draft_tokens.shape[1]
        if dlen == 0:
            print(f"\n[iter {it}] draft_len=0 (unexpected), stop.")
            break

        draft_list = [int(x) for x in draft_tokens[0].tolist()]
        draft_show = " ".join(_tok_repr(tokenizer, t) for t in draft_list)

        ones = torch.ones(1, 1 + dlen, device=device, dtype=attn.dtype)
        outv = llm(
            torch.cat([new_token_1, draft_tokens], dim=1),
            attention_mask=torch.cat([attn, ones], dim=1),
            past_key_values=past_kv,
            output_hidden_states=True,
        )

        llm_pred = outv.logits[:, -(dlen + 1) :, :].argmax(dim=-1)
        llm_next = int(llm_pred[0, 0].item())
        llm_verify = llm_pred[:, 1:].cpu()

        cur_llm_p = llm_next
        acc_token = [cur_llm_p]
        acc_ids = []
        accepted_draft = 0

        draft_cpu = draft_tokens.cpu()
        for j in range(dlen):
            if cur_llm_p == int(draft_cpu[0, j]):
                acc_ids.append(j + 1)
                accepted_draft += 1
                cur_llm_p = int(llm_verify[0, j])
                acc_token.append(cur_llm_p)
            else:
                break

        advanced = 1 + accepted_draft
        remaining = max_new_tokens - cur_gen
        if advanced > remaining:
            acc_token = acc_token[:remaining]
            advanced = len(acc_token)

        acc_show = " ".join(_tok_repr(tokenizer, t) for t in acc_token)

        print(f"\n[iter {it}] draft_proposed(dlen={dlen}) = {draft_show}")
        print(f"         accepted_draft = {accepted_draft}/{dlen} | advanced = {advanced}")
        print(f"         appended_tokens = {acc_show}")

        iter_num += 1
        iter_num_small += dlen
        sum_acc_num += advanced
        generated.extend(acc_token)
        cur_gen += advanced

        if tokenizer.eos_token_id in acc_token:
            print("\n[stop] eos encountered.")
            break
        if cur_gen >= max_new_tokens:
            print("\n[stop] reached max_new_tokens.")
            break

        past_kv_new = [list(x) for x in outv.past_key_values]
        for x in range(len(past_kv_new)):
            for y in range(len(past_kv_new[x])):
                keep_idx = [idx + last_past_len for idx in acc_ids]
                past_kv_new[x][y] = torch.cat(
                    [
                        past_kv_new[x][y][:, :, : last_past_len + 1, :],
                        past_kv_new[x][y][:, :, keep_idx, :]
                        if len(keep_idx) > 0
                        else past_kv_new[x][y][:, :, :0, :],
                    ],
                    dim=2,
                )
        past_kv = tuple(tuple(x) for x in past_kv_new)

        accepted_layers = _select_accepted_layers(outv, hidden_layer, layer_indices, acc_ids)
        slot_cache = edd.update_slot_cache(slot_cache, accepted_layers, apply_dropout=False)

        new_token_1 = torch.tensor([[generated[-1]]], device=device, dtype=torch.long)
        attn = torch.cat([attn, torch.ones(1, advanced, device=device, dtype=attn.dtype)], dim=1)

    print("\n========== OUTPUT (decoded) ==========")
    print(tokenizer.decode(generated))
    avg_accept_rate = sum_acc_num / max(iter_num_small, 1)
    avg_accept_len = sum_acc_num / max(iter_num, 1)
    print("\n========== SUMMARY ==========")
    print(f"total iterations: {iter_num}")
    print(f"total proposed draft tokens: {iter_num_small}")
    print(f"total advanced tokens: {sum_acc_num}")
    print(f"avg accept rate (alpha): {avg_accept_rate}")
    print(f"avg accepted length (tau): {avg_accept_len}")


def main():
    ap = argparse.ArgumentParser("Trace EDD-only accept length per iteration")
    ap.add_argument("--model_checkpoint", type=str, required=True)
    ap.add_argument("--draft_model_checkpoint", type=str, required=True)
    ap.add_argument("--hidden_layer", type=int, default=-4, help="Single hidden layer index (used when --fusion_layers is empty)")
    ap.add_argument("--fusion_layers", type=str, default="", help="Comma-separated layer indices, e.g. -3,-2,-1")
    ap.add_argument("--draft_len", type=int, default=5)
    ap.add_argument("--max_new_tokens", type=int, default=64)
    ap.add_argument("--prompt", type=str, default=None, help="Direct prompt string")
    ap.add_argument("--data_dir", type=str, default=None, help="humaneval.jsonl with prompt/pred")
    ap.add_argument("--index", type=int, default=0, help="which row to pick from data_dir")
    ap.add_argument("--dtype", type=str, default="bf16", choices=["bf16", "fp16"])
    ap.add_argument("--tokenizer_dir", type=str, default="", help="Optional tokenizer cache directory")
    args = ap.parse_args()

    if args.prompt is None:
        if args.data_dir is None:
            raise ValueError("Provide either --prompt or --data_dir")
        prompt = load_prompt_from_humaneval(args.data_dir, args.index)
    else:
        prompt = args.prompt

    tokenizer = _load_tokenizer(args.model_checkpoint, args.tokenizer_dir.strip())

    torch_dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16

    llm = LlamaForCausalLM.from_pretrained(
        args.model_checkpoint,
        torch_dtype=torch_dtype,
        device_map="auto",
    ).cuda().eval()

    cfg = AutoConfig.from_pretrained(args.model_checkpoint)
    edd = Effective_Draft_Decoder(cfg.hidden_size, cfg.hidden_size * 2, cfg.num_attention_heads, 1, cfg)
    edd.lm_head.load_state_dict(llm.lm_head.state_dict())
    edd.embedding_layer.load_state_dict(llm.model.embed_tokens.state_dict())
    edd = edd.to(dtype=torch_dtype)

    sd = torch.load(args.draft_model_checkpoint, map_location="cpu")
    edd.load_state_dict(sd)
    edd = edd.cuda().eval()

    run_trace(
        tokenizer=tokenizer,
        llm=llm,
        edd=edd,
        prompt=prompt,
        hidden_layer=args.hidden_layer,
        layer_indices=_parse_layers(args.fusion_layers),
        draft_len=args.draft_len,
        max_new_tokens=args.max_new_tokens,
    )


if __name__ == "__main__":
    main()
