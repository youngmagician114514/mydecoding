# mydecoding/training/training_stage_A.py
"""
Stage A: 只训练 decoder1/head1（draft head）。

- num_phases = 1，只跑 phase1
- 只训练 head1 的参数
- 每 20 step 打印最近 20 个 step 的平均 loss
- 训练结束后在 ./results/ 下保存 loss 曲线图
"""

import os
import torch
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer
import matplotlib.pyplot as plt
from collections import deque

from mydecoding.config.dual_decoder import DualDecoderConfig
from mydecoding.models.dual_decoder import DualDecoderModel


def make_dataloader(tokenizer, seq_len: int, batch_size: int) -> DataLoader:
    """
    使用本地文本文件 data/train.txt 构造数据集。
    一行一条样本；你可以换成自己的语料路径。
    """
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")

    def tok_fn(ex):
        out = tokenizer(
            ex["text"],
            truncation=True,
            max_length=seq_len,
            padding="max_length",
        )
        return {
            "input_ids": out["input_ids"],
            "attention_mask": out["attention_mask"],
        }

    ds = ds.map(tok_fn, remove_columns=ds.column_names)
    ds.set_format(type="torch", columns=["input_ids", "attention_mask"])
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
    )
    return loader


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.backends.cuda.matmul.allow_tf32 = True

    # ===== config =====
    config = DualDecoderConfig(
        base_model_name_or_path="Qwen/Qwen2.5-3B",
        num_draft_candidates=3,
        max_speculative_steps=1,  # Stage A 实际只用 phase1
        draft_hidden_size=1024,
        draft_num_layers=2,
        fusion_hidden_size=1024,
        fusion_num_heads=4,
        fusion_dropout=0.1,
        decoder_num_layers=2,
        decoder_num_heads=8,
        decoder_dropout=0.1,
        draft_loss_weight=1.0,
        fusion_loss_weight=1.0,
    )

    SEQ_LEN = 256
    BATCH_SIZE = 1
    GRAD_ACCUM = 8

    # ===== tokenizer & data =====
    tokenizer = AutoTokenizer.from_pretrained(
        config.base_model_name_or_path,
        use_fast=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    loader = make_dataloader(tokenizer, SEQ_LEN, BATCH_SIZE)

    # ===== model =====
    model = DualDecoderModel(config).to(device)
    model.train()

    # 只训练 head1，冻结 fusion/head2
    for p in model.head1.parameters():
        p.requires_grad = True
    for p in model.fusion.parameters():
        p.requires_grad = False
    for p in model.head2.parameters():
        p.requires_grad = False

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    print(f"[StageA] trainable params: {sum(p.numel() for p in trainable_params)/1e6:.1f} M")

    opt = torch.optim.AdamW(
        trainable_params,
        lr=2e-4,
        betas=(0.9, 0.95),
        weight_decay=0.1,
    )

    use_bf16 = torch.cuda.is_bf16_supported()
    autocast_dtype = torch.bfloat16 if use_bf16 else torch.float16
    scaler = torch.cuda.amp.GradScaler(enabled=(autocast_dtype == torch.float16))

    num_phases = 1  # 只跑 phase1
    global_step = 0
    opt.zero_grad(set_to_none=True)

    # 用 deque 记录最近 20 个 step 的 loss
    window_size = 20
    recent_losses = deque(maxlen=window_size)

    # 保存所有 step 的 loss 用来画图
    all_steps = []
    all_losses = []

    for batch in loader:
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(device, non_blocking=True)

        with torch.autocast(device_type="cuda", dtype=autocast_dtype):
            out = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                num_phases=num_phases,
                temperature=1.0,
            )
            # head1_loss 就是我们关心的
            step_loss = out.head1_loss
            loss = step_loss / GRAD_ACCUM

        # 记录 loss（不除 grad_accum）
        loss_value = step_loss.item()
        all_steps.append(global_step)
        all_losses.append(loss_value)
        recent_losses.append(loss_value)

        if scaler.is_enabled():
            scaler.scale(loss).backward()
        else:
            loss.backward()

        if (global_step + 1) % GRAD_ACCUM == 0:
            if scaler.is_enabled():
                scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)

            if scaler.is_enabled():
                scaler.step(opt)
                scaler.update()
            else:
                opt.step()
            opt.zero_grad(set_to_none=True)

        # 每 20 个 step 打一次最近 20 个的平均 loss
        if (global_step + 1) % window_size == 0:
            avg_loss = sum(recent_losses) / len(recent_losses)
            print(f"[StageA] step={global_step+1}  avg_loss(last {len(recent_losses)} steps)={avg_loss:.4f}")

        global_step += 1
        if global_step >= 5000:
            break

    # ===== 保存 checkpoint =====
    os.makedirs("checkpoints", exist_ok=True)
    ckpt_path = "checkpoints/dual_decoder_stageA_head1.pt"
    torch.save(model.state_dict(), ckpt_path)
    print(f"[StageA] saved checkpoint to {ckpt_path}")

    # ===== 画 loss 曲线并保存 =====
    os.makedirs("results", exist_ok=True)
    plt.figure()
    plt.plot(all_steps, all_losses)
    plt.xlabel("step")
    plt.ylabel("head1_loss")
    plt.title("StageA head1 KL loss")
    plt.grid(True)
    out_png = os.path.join("results", "stageA_head1_loss.png")
    plt.savefig(out_png, dpi=150, bbox_inches="tight")
    print(f"[StageA] loss curve saved to {out_png}")


if __name__ == "__main__":
    main()
