
"""
Speedup-oriented evaluation.

This script answers ONLY: "does my speculative pipeline help inference speed?"

It estimates:
  - accepted_len: how many draft tokens (g1..gK) match base greedy consecutively
  - accept@1 / accept@2 / ... : acceptance rate for each depth
  - expected accepted tokens per step

We intentionally avoid other metrics (entropy, KL curves, etc.) to keep eval minimal.

Usage example:
  python -m mydecoding.eval_speedup \
      --stageA_ckpt checkpoints/stageA_draft_head_step5000_last.pt \
      --stageB_ckpt checkpoints/stageB_student_step5000_last.pt \
      --num_samples 200 --seq_len 256 --min_t 32 --num_phases 3

Dataset:
  default = ShareGPT (same as training)
  override via env:
    SHAREGPT_DATASET / SHAREGPT_SPLIT
"""

from __future__ import annotations

import os
import random
import argparse
from typing import Dict, List

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from mydecoding.models.dual_decoder import DualDecoderModel, DualDecoderConfig
from mydecoding.training.data_sharegpt import build_sharegpt_tokenized_dataset


def load_ckpts(model: DualDecoderModel, stageA_ckpt: str | None, stageB_ckpt: str | None):
    if stageA_ckpt:
        stA = torch.load(stageA_ckpt, map_location="cpu")
        sdA = stA.get("draft_head_state_dict") or stA.get("head1_state_dict") or stA
        model.draft_head.load_state_dict(sdA, strict=False)

    if stageB_ckpt:
        stB = torch.load(stageB_ckpt, map_location="cpu")
        sdF = stB.get("belief_fusion_state_dict") or stB.get("fusion_state_dict")
        sdH = stB.get("fusion_head_state_dict") or stB.get("head2_state_dict")
        if sdF is not None:
            model.belief_fusion.load_state_dict(sdF, strict=False)
        if sdH is not None:
            model.fusion_head.load_state_dict(sdH, strict=False)


@torch.no_grad()
def predict_g_tokens(model: DualDecoderModel, input_ctx: torch.Tensor, attn_ctx: torch.Tensor, num_phases: int) -> torch.Tensor:
    """
    Return predicted tokens g1..gK from the *student* pipeline by argmax at each phase.
    """
    out = model(
        input_ids=input_ctx,
        attention_mask=attn_ctx,
        num_phases=num_phases,
        infer_mode=True,
        teacher_mode="base_greedy",  # so we also get base_greedy_tokens
    )
    g1 = out.draft_head_logits.argmax(dim=-1, keepdim=True)  # (B,1)
    preds = [g1]
    if out.fusion_head_logits_all is not None and out.fusion_head_logits_all.size(1) > 0:
        for i in range(out.fusion_head_logits_all.size(1)):
            gi = out.fusion_head_logits_all[:, i, :].argmax(dim=-1, keepdim=True)
            preds.append(gi)
    return torch.cat(preds, dim=1), out.base_greedy_tokens


def longest_prefix_match(student: torch.Tensor, teacher: torch.Tensor) -> int:
    """
    student/teacher: (K,)
    return max m such that student[:m]==teacher[:m]
    """
    K = min(student.numel(), teacher.numel())
    m = 0
    for i in range(K):
        if int(student[i].item()) == int(teacher[i].item()):
            m += 1
        else:
            break
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, default="Qwen/Qwen2.5-3B")
    ap.add_argument("--stageA_ckpt", type=str, default=None)
    ap.add_argument("--stageB_ckpt", type=str, default=None)
    ap.add_argument("--num_samples", type=int, default=200)
    ap.add_argument("--seq_len", type=int, default=256)
    ap.add_argument("--min_t", type=int, default=32)
    ap.add_argument("--num_phases", type=int, default=3)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    config = DualDecoderConfig(
        base_model_name_or_path=args.model,
        max_speculative_steps=args.num_phases - 1,
    )
    model = DualDecoderModel(config).to(device)
    model.eval()

    load_ckpts(model, args.stageA_ckpt, args.stageB_ckpt)

    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    sharegpt_name = os.environ.get("SHAREGPT_DATASET", "anon8231489123/ShareGPT_Vicuna_unfiltered")
    sharegpt_split = os.environ.get("SHAREGPT_SPLIT", "train")

    ds = build_sharegpt_tokenized_dataset(
        tokenizer=tokenizer,
        seq_len=args.seq_len,
        dataset_name=sharegpt_name,
        split=sharegpt_split,
        streaming=False,
        max_samples=max(args.num_samples * 2, args.num_samples),
    )

    accepted_lens: List[int] = []
    accept_at = [0 for _ in range(args.num_phases + 1)]  # 1..K

    n = 0
    idx = 0
    while n < args.num_samples and idx < len(ds):
        ex = ds[idx]
        idx += 1
        input_ids = ex["input_ids"].unsqueeze(0).to(device)
        attn = ex["attention_mask"].unsqueeze(0).to(device)
        valid_len = int(attn.sum().item())
        if valid_len <= args.min_t + 1:
            continue

        ctx_len = random.randint(args.min_t, min(valid_len, args.seq_len))
        ctx_ids = input_ids[:, :ctx_len]
        ctx_attn = attn[:, :ctx_len]

        student_tokens, teacher_tokens = predict_g_tokens(model, ctx_ids, ctx_attn, args.num_phases)
        if teacher_tokens is None:
            continue

        m = longest_prefix_match(student_tokens[0], teacher_tokens[0])
        accepted_lens.append(m)
        for j in range(1, args.num_phases + 1):
            if m >= j:
                accept_at[j] += 1
        n += 1

    if n == 0:
        print("No valid samples.")
        return

    avg_accept = sum(accepted_lens) / n
    print("==================================")
    print("[Speedup Eval] (base greedy as reference)")
    print(f"num_samples = {n}")
    print(f"num_phases  = {args.num_phases}")
    print(f"avg_accepted_len = {avg_accept:.3f}")

    for j in range(1, args.num_phases + 1):
        print(f"accept@{j}: {accept_at[j]}/{n} = {accept_at[j]/n:.4f}")

    # very rough speedup proxy:
    # if you propose K tokens but only accept m on average, your "useful tokens per verify" ~ avg_accept
    # you still pay one base forward to verify; so speedup proxy ~ (1 + avg_accept) / 1 = 1 + avg_accept
    print(f"rough_speedup_proxy ≈ 1 + avg_accepted_len = {1.0 + avg_accept:.3f}")
    print("==================================")
    print("Tip: for real wall-clock speedup, measure latency with caching + batched verify.")


if __name__ == "__main__":
    main()
