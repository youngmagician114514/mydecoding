
# mydecoding/training/training_stage_A_sharpen_head1.py
"""
Stage A (draft_head, sharpen): train ONLY the draft_head, with extra entropy penalty to make it sharper.

Changes vs old code:
  ✅ uses ShareGPT dataset (chat-style) instead of Wikitext
  ✅ naming normalized: draft_head / fusion_head
  ✅ saves draft_head_state_dict (and keeps head1_state_dict alias for compatibility)
"""

import os
import random
from collections import deque

import torch
from torch.optim import AdamW
from transformers import AutoTokenizer

from mydecoding.models.dual_decoder import DualDecoderModel, DualDecoderConfig
from mydecoding.training.data_sharegpt import build_sharegpt_tokenized_dataset


def count_trainable_params(model: torch.nn.Module) -> float:
    return sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6


def freeze_all_but_draft_head(model: DualDecoderModel):
    for p in model.parameters():
        p.requires_grad = False
    for name, p in model.named_parameters():
        if name.startswith("draft_head."):
            p.requires_grad = True
    # safety
    for p in model.base_model.parameters():
        p.requires_grad = False


def save_draft_head_ckpt(model: DualDecoderModel, optimizer, step: int, out_dir="checkpoints", tag=""):
    os.makedirs(out_dir, exist_ok=True)
    name = f"stageA_draft_head_step{step}{tag}.pt"
    path = os.path.join(out_dir, name)
    torch.save(
        {
            "step": step,
            "draft_head_state_dict": model.draft_head.state_dict(),
            # alias for old scripts:
            "head1_state_dict": model.draft_head.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": getattr(model, "config", None),
        },
        path,
    )
    print(f"[StageA] draft_head ckpt saved: {path}")


def main():
    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("results", exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[StageA] device={device}")

    # ---- hyperparams ----
    base_name = "Qwen/Qwen2.5-3B"
    SEQ_LEN = 256
    GRAD_ACCUM = 8
    MAX_STEPS = 15000

    MIN_T = 32
    LR = 2e-4
    WEIGHT_DECAY = 0.01

    TEMPERATURE = 0.3  # smaller => sharper (used in KL in DualDecoderModel)
    LOG_INTERVAL = 50
    SAVE_INTERVAL = 5000
    PLOT_WINDOW = 20

    # ---- ShareGPT dataset ----
    SHAREGPT_NAME = os.environ.get("SHAREGPT_DATASET", "local")
    SHAREGPT_SPLIT = os.environ.get("SHAREGPT_SPLIT", "train")
    MAX_SAMPLES = int(os.environ.get("SHAREGPT_MAX_SAMPLES", "0")) or None

    # ---- config ----
    config = DualDecoderConfig(
        base_model_name_or_path=base_name,
        draft_head_num_candidates=3,
        max_speculative_steps=1,

        # draft head params
        draft_num_layers=4,
        draft_dropout=0.0,

        # fusion & fusion_head (not trained in StageA, but must exist)
        fusion_hidden_size=1024,
        fusion_num_heads=4,
        fusion_dropout=0.0,

        fusion_head_num_layers=2,
        fusion_head_num_heads=8,
        fusion_head_dropout=0.0,
    )

    tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    tokenized = build_sharegpt_tokenized_dataset(
        tokenizer=tokenizer,
        seq_len=SEQ_LEN,
        dataset_name=SHAREGPT_NAME,
        split=SHAREGPT_SPLIT,
        streaming=False,
        max_samples=MAX_SAMPLES,
    )

    model = DualDecoderModel(config).to(device)
    model.train()
    freeze_all_but_draft_head(model)

    print(f"[StageA] trainable params: {count_trainable_params(model):.1f} M")

    optimizer = AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=LR,
        weight_decay=WEIGHT_DECAY,
    )

    loss_deque = deque(maxlen=PLOT_WINDOW)
    global_step = 0
    idx = 0

    while global_step < MAX_STEPS:
        # sample a batch item (B=1 by default; keep code simple)
        ex = tokenized[idx % len(tokenized)]
        idx += 1

        input_ids = ex["input_ids"].unsqueeze(0).to(device)         # (1,SEQ_LEN)
        attention_mask = ex["attention_mask"].unsqueeze(0).to(device)

        # choose ctx_len t based on valid length (avoid padding)
        valid_len = int(attention_mask.sum().item())
        if valid_len <= MIN_T + 1:
            continue
        ctx_len = random.randint(MIN_T, min(valid_len - 1, SEQ_LEN - 1))  # need at least 1 next token

        # build [x_{1:ctx_len} + 1 target token] so that K=1 works
        seq = input_ids[:, : ctx_len + 1]
        attn = attention_mask[:, : ctx_len + 1]

        out = model(
            input_ids=seq,
            attention_mask=attn,
            num_phases=1,
            temperature=TEMPERATURE,
            train_stage="draft_head",
            infer_mode=False,
            teacher_mode="teacher_forced",
        )

        # extra sharpening: penalize high entropy
        ps = torch.softmax(out.draft_head_logits.float(), dim=-1)
        entropy = (-ps * ps.clamp_min(1e-12).log()).sum(dim=-1).mean()
        ENTROPY_WEIGHT = float(os.environ.get("DRAFT_HEAD_ENTROPY_W", "0.05"))
        loss = (out.loss + ENTROPY_WEIGHT * entropy) / GRAD_ACCUM
        loss.backward()

        if (global_step + 1) % GRAD_ACCUM == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        loss_deque.append(float(loss.item()) * GRAD_ACCUM)

        if global_step % LOG_INTERVAL == 0:
            avg_loss = sum(loss_deque) / max(len(loss_deque), 1)
            # quick sharpness proxy
            ps = torch.softmax(out.draft_head_logits.float(), dim=-1)
            top1_prob = float(ps.max(dim=-1).values.mean().item())
            print(
                f"[StageA][step {global_step}] loss={avg_loss:.4f} "
                f"top1_prob={top1_prob:.4f} (T={TEMPERATURE}, ctx_len={ctx_len})"
            )

        if global_step > 0 and global_step % SAVE_INTERVAL == 0:
            save_draft_head_ckpt(model, optimizer, global_step, tag="_last")

        global_step += 1

    save_draft_head_ckpt(model, optimizer, global_step, tag="_last")
    print("[StageA] done.")


if __name__ == "__main__":
    main()
