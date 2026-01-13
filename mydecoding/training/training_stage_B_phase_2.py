
# mydecoding/training/training_stage_B_phase_2.py
"""
Stage B (phase2..K): train belief_fusion + fusion_head while freezing base + draft_head.

Changes vs old code:
  ✅ uses ShareGPT dataset
  ✅ naming normalized: draft_head / fusion_head
  ✅ fusion_head conditions on (h_{1:t}, z_1..), matching inference usage
  ✅ checkpoint saves belief_fusion_state_dict + fusion_head_state_dict (and keeps old keys as aliases)
"""

import os
import random
from collections import deque

import torch
from torch.optim import AdamW
from transformers import AutoTokenizer

from mydecoding.models.dual_decoder import DualDecoderModel, DualDecoderConfig
from mydecoding.training.data_sharegpt import build_sharegpt_tokenized_dataset


def freeze_all(model: torch.nn.Module):
    for p in model.parameters():
        p.requires_grad = False


def unfreeze_belief_fusion_and_fusion_head(model: DualDecoderModel):
    for p in model.belief_fusion.parameters():
        p.requires_grad = True
    for p in model.fusion_head.parameters():
        p.requires_grad = True

    # keep lm_head frozen
    lm_head = model.base_model.get_output_embeddings()
    if lm_head is not None:
        for p in lm_head.parameters():
            p.requires_grad = False


def load_draft_head_ckpt(model: DualDecoderModel, ckpt_path: str):
    st = torch.load(ckpt_path, map_location="cpu")
    sd = st.get("draft_head_state_dict") or st.get("head1_state_dict") or st.get("model_state_dict") or st
    missing, unexpected = model.draft_head.load_state_dict(sd, strict=False)
    print(f"[StageB] loaded draft_head: missing={len(missing)} unexpected={len(unexpected)}")


def save_stageB_ckpt(model: DualDecoderModel, optimizer, step: int, out_dir="checkpoints", tag=""):
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"stageB_student_step{step}{tag}.pt")
    torch.save(
        {
            "step": step,
            "belief_fusion_state_dict": model.belief_fusion.state_dict(),
            "fusion_head_state_dict": model.fusion_head.state_dict(),
            # aliases for old scripts:
            "fusion_state_dict": model.belief_fusion.state_dict(),
            "head2_state_dict": model.fusion_head.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": getattr(model, "config", None),
        },
        path,
    )
    print(f"[StageB] ckpt saved: {path}")


def main():
    os.makedirs("checkpoints", exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[StageB] device={device}")

    # ---- hyperparams ----
    base_name = "Qwen/Qwen2.5-3B"
    SEQ_LEN = 256
    GRAD_ACCUM = 8
    MAX_STEPS = 5000

    NUM_PHASES = 3   # phase1 + phase2 + phase3 (you can change later)
    MIN_T = 32

    LR = 2e-4
    WEIGHT_DECAY = 0.01
    TEMPERATURE = 1.0

    LOG_INTERVAL = 50
    SAVE_INTERVAL = 5000
    WINDOW = 20

    # ---- paths ----
    STAGEA_CKPT = os.environ.get("STAGEA_DRAFT_HEAD_CKPT", "checkpoints/stageA_draft_head_step5000_last.pt")

    # ---- ShareGPT dataset ----
    SHAREGPT_NAME = os.environ.get("SHAREGPT_DATASET", "anon8231489123/ShareGPT_Vicuna_unfiltered")
    SHAREGPT_SPLIT = os.environ.get("SHAREGPT_SPLIT", "train")
    MAX_SAMPLES = int(os.environ.get("SHAREGPT_MAX_SAMPLES", "0")) or None

    # ---- config ----
    config = DualDecoderConfig(
        base_model_name_or_path=base_name,

        draft_head_num_candidates=3,
        max_speculative_steps=NUM_PHASES - 1,

        # draft_head exists but frozen
        draft_num_layers=4,
        draft_dropout=0.0,

        # belief_fusion trainable
        fusion_hidden_size=1024,
        fusion_num_heads=4,
        fusion_dropout=0.0,

        # fusion_head trainable
        fusion_head_num_candidates=10,
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

    freeze_all(model)
    load_draft_head_ckpt(model, STAGEA_CKPT)
    # freeze base + draft_head, unfreeze student modules
    unfreeze_belief_fusion_and_fusion_head(model)

    optimizer = AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=LR,
        weight_decay=WEIGHT_DECAY,
    )

    loss_hist = deque(maxlen=WINDOW)
    global_step = 0
    idx = 0

    while global_step < MAX_STEPS:
        ex = tokenized[idx % len(tokenized)]
        idx += 1

        input_ids = ex["input_ids"].unsqueeze(0).to(device)
        attention_mask = ex["attention_mask"].unsqueeze(0).to(device)

        valid_len = int(attention_mask.sum().item())
        if valid_len <= MIN_T + NUM_PHASES:
            continue

        # choose ctx_len so that we have room for K dummy suffix tokens (they are ignored under base_greedy teacher)
        ctx_len = random.randint(MIN_T, min(valid_len - NUM_PHASES, SEQ_LEN - NUM_PHASES))
        seq = input_ids[:, : ctx_len + NUM_PHASES]
        attn = attention_mask[:, : ctx_len + NUM_PHASES]

        out = model(
            input_ids=seq,
            attention_mask=attn,
            num_phases=NUM_PHASES,
            temperature=TEMPERATURE,
            train_stage="all",
            infer_mode=False,
            teacher_mode="base_greedy",  # ✅ aligned with speculative setting
        )

        loss = out.loss / GRAD_ACCUM
        loss.backward()

        if (global_step + 1) % GRAD_ACCUM == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        loss_hist.append(float(loss.item()) * GRAD_ACCUM)

        if global_step % LOG_INTERVAL == 0:
            avg = sum(loss_hist) / max(len(loss_hist), 1)
            # quick diagnostics for phase2 logits
            if out.fusion_head_logits_all is not None and out.fusion_head_logits_all.numel() > 0:
                l2 = out.fusion_head_logits_all[:, 0, :].float()
                ps = torch.softmax(l2, dim=-1)
                top1_prob = float(ps.max(dim=-1).values.mean().item())
            else:
                top1_prob = 0.0
            print(f"[StageB][step {global_step}] loss={avg:.4f} phase2_top1_prob={top1_prob:.4f} (ctx_len={ctx_len})")

        if global_step > 0 and global_step % SAVE_INTERVAL == 0:
            save_stageB_ckpt(model, optimizer, global_step, tag="_last")

        global_step += 1

    save_stageB_ckpt(model, optimizer, global_step, tag="_last")
    print("[StageB] done.")


if __name__ == "__main__":
    main()
