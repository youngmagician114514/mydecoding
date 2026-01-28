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


def _prefix_plus_causal_bias(prefix_len: int, total_len: int, dtype, device):
    """
    sequence = [PREFIX (hidden states) ; TOKEN_STREAM (embeddings)]
    prefix_len = P
    total_len = P + S

    Rules:
      - Token-stream queries (positions >= P) can attend to:
          * all prefix keys [0..P-1]
          * causal over token-stream keys [P..q]
      - Prefix queries don't matter (we never take logits there), keep them fully open.
    Returns additive bias (1,1,total_len,total_len) with 0 allowed, -inf masked.
    """
    neg_inf = torch.finfo(dtype).min
    bias = torch.zeros((1, 1, total_len, total_len), device=device, dtype=dtype)

    P = prefix_len
    if total_len <= P:
        return bias

    # mask token-stream future
    q = torch.arange(P, total_len, device=device).view(-1, 1)  # (S,1)
    k = torch.arange(P, total_len, device=device).view(1, -1)  # (1,S)
    allow = (k <= q)                                           # (S,S)
    dd = torch.where(
        allow,
        torch.zeros_like(allow, dtype=dtype),
        torch.full_like(allow, neg_inf, dtype=dtype),
    )
    bias[:, :, P:total_len, P:total_len] = dd
    return bias


@torch.no_grad()
def propose_k_tminus1_hidden(
    target,                 # AutoModelForCausalLM
    draft,                  # AutoModelForCausalLM (your 1-layer draft)
    ctx_ids: torch.Tensor,  # (1,T) verified prefix tokens
    ctx_attn: torch.Tensor, # (1,T)
    K: int = 5,
    enc_layer_index: int = -4,
):
    """
    Block-free proposal:
      PREFIX  = h1..h_{T-1} (from target layer enc_layer_index)
      TOKENS  = e(t_T), then e(d1), e(d2), ... autoregressively
      Each step uses last position (a token position) -> lm_head -> next draft token.
    """
    device = ctx_ids.device
    dtype = next(draft.parameters()).dtype

    T = ctx_ids.size(1)
    assert T >= 1, "ctx_ids must contain at least 1 verified token"

    # 1) get hidden states from target
    out_t = target(
        input_ids=ctx_ids,
        attention_mask=ctx_attn,
        output_hidden_states=True,
        use_cache=False,
        return_dict=True,
    )
    enc = out_t.hidden_states[enc_layer_index].to(dtype=dtype)  # (1,T,H)

    # PREFIX: h1..h_{T-1}
    if T > 1:
        prefix = enc[:, :-1, :]  # (1,T-1,H)
        P = T - 1
    else:
        prefix = None
        P = 0

    # TOKEN STREAM starts from last verified token t_T
    tok_stream = ctx_ids[:, -1:].clone()  # (1,1) = [t_T]
    gen = []

    for _ in range(K):
        tok_emb = draft.get_input_embeddings()(tok_stream).to(dtype=dtype)  # (1,S,H)

        if prefix is None:
            inputs_embeds = tok_emb                      # (1,S,H)
        else:
            inputs_embeds = torch.cat([prefix, tok_emb], dim=1)  # (1,P+S,H)

        L = inputs_embeds.size(1)
        attn_bias = _prefix_plus_causal_bias(P, L, inputs_embeds.dtype, device)

        out_d = draft.model(
            inputs_embeds=inputs_embeds,
            attention_mask=attn_bias,
            use_cache=False,
            return_dict=True,
        )

        h_last = out_d.last_hidden_state[:, -1, :]       # last token position
        logits = draft.lm_head(h_last)                   # (1,V)
        next_id = torch.argmax(logits, dim=-1, keepdim=True)  # (1,1)

        gen.append(next_id)
        tok_stream = torch.cat([tok_stream, next_id], dim=1)  # append generated token

    return torch.cat(gen, dim=1)  # (1,K)


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
        student_tokens = propose_k_tminus1_hidden(
            target=target,
            draft=draft,              # 你的 draft 模型
            ctx_ids=ctx_ids,
            ctx_attn=ctx_attn,
            K=5,
            enc_layer_index=-4,
        )

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
