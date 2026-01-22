# eval_edd_speedup.py
# Evaluate EDD draft acceptance vs target greedy (speedup-oriented, like eval_speedup.py)

from __future__ import annotations
import argparse
import json
import os
import random
from typing import List, Dict, Any

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM


# -------------------------
# Dataset (same parsing style as your training ShareGPTDataset)
# -------------------------
class ShareGPTDataset:
    def __init__(self, path: str, tokenizer, max_length: int = 2048):
        self.samples: List[str] = []
        self.tokenizer = tokenizer
        self.max_length = max_length

        items = []
        if path.lower().endswith(".jsonl"):
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    items.append(json.loads(line))
        elif path.lower().endswith(".json"):
            with open(path, "r", encoding="utf-8") as f:
                obj = json.load(f)
            if isinstance(obj, list):
                items = obj
            elif isinstance(obj, dict):
                # try common roots
                for k in ("data", "train", "val"):
                    if k in obj and isinstance(obj[k], list):
                        items = obj[k]
                        break
                if not isinstance(items, list):
                    # fallback: first list value
                    for v in obj.values():
                        if isinstance(v, list):
                            items = v
                            break
                if not isinstance(items, list):
                    raise ValueError(f"Unsupported JSON structure: keys={list(obj.keys())[:20]}")
            else:
                raise ValueError(f"Unsupported JSON type: {type(obj)}")
        else:
            raise ValueError("data file must end with .jsonl or .json")

        for sample in items:
            conv = sample.get("conversations", [])
            msgs = []
            for turn in conv:
                frm = turn.get("from")
                val = turn.get("value", "")
                if frm in ("human", "user"):
                    role = "user"
                elif frm in ("gpt", "assistant"):
                    role = "assistant"
                else:
                    continue
                msgs.append({"role": role, "content": val})

            if not msgs:
                continue

            # prefer chat_template
            try:
                text = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
            except Exception:
                text = ""
                for m in msgs:
                    text += f"{m['role'].upper()}: {m['content']}\n"

            self.samples.append(text)

    def __len__(self):
        return len(self.samples)

    def get_tokenized(self, idx: int) -> Dict[str, torch.Tensor]:
        text = self.samples[idx]
        enc = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding=False,
            return_tensors="pt",
        )
        return {"input_ids": enc["input_ids"][0], "attention_mask": enc["attention_mask"][0]}


# -------------------------
# Core metric utils (copied from eval_speedup.py style)
# -------------------------
def longest_prefix_match(student: torch.Tensor, teacher: torch.Tensor) -> int:
    K = min(student.numel(), teacher.numel())
    m = 0
    for i in range(K):
        if int(student[i].item()) == int(teacher[i].item()):
            m += 1
        else:
            break
    return m


# -------------------------
# EDD draft propose K tokens
# -------------------------
@torch.no_grad()
def edd_propose_k(
    target: AutoModelForCausalLM,
    draft: AutoModelForCausalLM,
    ctx_ids: torch.Tensor,          # (1, t)
    ctx_attn: torch.Tensor,         # (1, t)
    K: int,
    enc_layer_index: int,
) -> torch.Tensor:
    """
    Propose K tokens using EDD-style: use target hidden states as soft prompt prefix.
    Implementation (simple & faithful to "hidden-state prompt"):
      - Run target once on ctx -> get enc_hidden = hidden_states[layer]
      - Then autoregressively generate K tokens with draft, conditioning on prefix enc_hidden
        (we DO NOT feed ctx token embeddings again; only feed generated tokens embeddings)
    """
    device = ctx_ids.device

    # 1) encode with target
    out_t = target(
        input_ids=ctx_ids,
        attention_mask=ctx_attn,
        output_hidden_states=True,
        use_cache=False,
        return_dict=True,
    )
    enc_hidden = out_t.hidden_states[enc_layer_index]  # (1,t,H)

    # 2) autoregressive K steps with draft
    gen_tokens: List[torch.Tensor] = []
    block_ids = None  # (1, s) tokens generated so far

    for _ in range(K):
        if block_ids is None:
            # only prefix
            inputs_embeds = enc_hidden
            attn_mask = torch.ones((1, enc_hidden.size(1)), device=device, dtype=ctx_attn.dtype)
        else:
            tok_embeds = draft.get_input_embeddings()(block_ids)  # (1,s,H)
            inputs_embeds = torch.cat([enc_hidden, tok_embeds], dim=1)  # (1,t+s,H)
            attn_mask = torch.ones((1, inputs_embeds.size(1)), device=device, dtype=ctx_attn.dtype)

        out_d = draft.model(
            inputs_embeds=inputs_embeds,
            attention_mask=attn_mask,
            use_cache=False,
            return_dict=True,
        )
        h_last = out_d.last_hidden_state[:, -1, :]             # (1,H)
        logits = draft.lm_head(h_last)                         # (1,V)
        next_id = torch.argmax(logits, dim=-1, keepdim=True)   # (1,1)

        gen_tokens.append(next_id)

        block_ids = next_id if block_ids is None else torch.cat([block_ids, next_id], dim=1)

    return torch.cat(gen_tokens, dim=1)  # (1,K)


@torch.no_grad()
def target_greedy_k(target: AutoModelForCausalLM, ctx_ids: torch.Tensor, ctx_attn: torch.Tensor, K: int) -> torch.Tensor:
    """
    Greedy generate K tokens from target as reference.
    """
    gen = target.generate(
        input_ids=ctx_ids,
        attention_mask=ctx_attn,
        max_new_tokens=K,
        do_sample=False,
        use_cache=True,
    )
    return gen[:, -K:]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target_model", type=str, default="Qwen/Qwen2.5-3B")
    ap.add_argument("--edd_dir", type=str, required=True, help="your trained EDD draft HF folder (has model.safetensors)")
    ap.add_argument("--data_jsonl", type=str, required=True, help="ShareGPT train/val jsonl")
    ap.add_argument("--num_samples", type=int, default=200)
    ap.add_argument("--seq_len", type=int, default=256)
    ap.add_argument("--min_t", type=int, default=32)
    ap.add_argument("--K", type=int, default=3, help="speculation length")
    ap.add_argument("--enc_layer_index", type=int, default=-1, help="which target hidden layer to use as prompt")
    ap.add_argument("--bf16", action="store_true")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if args.bf16 else torch.float16

    tokenizer = AutoTokenizer.from_pretrained(args.target_model, use_fast=True, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # target = reference greedy model
    target = AutoModelForCausalLM.from_pretrained(
        args.target_model,
        torch_dtype=dtype,
        device_map=None,
        trust_remote_code=True,
    ).to(device)
    target.eval()
    for p in target.parameters():
        p.requires_grad_(False)

    # draft = trained EDD draft (your output dir)
    draft = AutoModelForCausalLM.from_pretrained(
        args.edd_dir,
        torch_dtype=dtype,
        device_map=None,
        trust_remote_code=True,
    ).to(device)
    draft.eval()

    ds = ShareGPTDataset(args.data_jsonl, tokenizer, max_length=args.seq_len)

    accepted_lens: List[int] = []
    accept_at = [0 for _ in range(args.K + 1)]

    n = 0
    idx = 0
    while n < args.num_samples and idx < len(ds):
        ex = ds.get_tokenized(idx)
        idx += 1
        input_ids = ex["input_ids"].unsqueeze(0).to(device)
        attn = ex["attention_mask"].unsqueeze(0).to(device)

        valid_len = int(attn.sum().item())
        if valid_len <= args.min_t + args.K + 1:
            continue

        ctx_len = random.randint(args.min_t, min(valid_len - args.K - 1, args.seq_len - args.K - 1))
        ctx_ids = input_ids[:, :ctx_len]
        ctx_attn = attn[:, :ctx_len]

        teacher_tokens = target_greedy_k(target, ctx_ids, ctx_attn, args.K)
        student_tokens = edd_propose_k(target, draft, ctx_ids, ctx_attn, args.K, args.enc_layer_index)

        m = longest_prefix_match(student_tokens[0], teacher_tokens[0])
        accepted_lens.append(m)
        for j in range(1, args.K + 1):
            if m >= j:
                accept_at[j] += 1
        n += 1

    if n == 0:
        print("No valid samples.")
        return

    avg_accept = sum(accepted_lens) / n
    print("==================================")
    print("[EDD Speedup Eval] (target greedy as reference)")
    print(f"num_samples = {n}")
    print(f"K = {args.K}")
    print(f"enc_layer_index = {args.enc_layer_index}")
    print(f"avg_accepted_len = {avg_accept:.3f}")
    for j in range(1, args.K + 1):
        print(f"accept@{j}: {accept_at[j]}/{n} = {accept_at[j]/n:.4f}")
    print(f"rough_speedup_proxy ≈ 1 + avg_accepted_len = {1.0 + avg_accept:.3f}")
    print("==================================")


if __name__ == "__main__":
    main()
