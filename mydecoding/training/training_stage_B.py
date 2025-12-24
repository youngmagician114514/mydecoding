# mydecoding/training/training_stageB.py
"""
Stage B: 在已经训练好的 head1 上，训练 fusion + head2。

- 加载 Stage A 的 checkpoint
- 冻结 head1（以及 base，本来就冻结）
- 只让 fusion & head2 的参数 requires_grad=True
- num_phases = 1 + max_speculative_steps  (>=2)
- loss 使用 head2_loss（也可以用 total_loss，但梯度只会流向 head2/fusion）
"""

import os
import torch
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer

from mydecoding.config.dual_decoder import DualDecoderConfig
from mydecoding.models.dual_decoder import DualDecoderModel


def make_dataloader(tokenizer, seq_len: int, batch_size: int) -> DataLoader:
    
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

    # ===== config (和 StageA 保持一致) =====
    config = DualDecoderConfig(
        base_model_name_or_path="Qwen/Qwen2.5-3B",
        num_draft_candidates=3,
        max_speculative_steps=2,   # 你想多 phase 就改成 2，记得 T 要够长
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

    tokenizer = AutoTokenizer.from_pretrained(
        config.base_model_name_or_path,
        use_fast=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    loader = make_dataloader(tokenizer, SEQ_LEN, BATCH_SIZE)

    # ===== model =====
    model = DualDecoderModel(config).to(device)

    # 加载 Stage A checkpoint
    ckpt_path = "checkpoints/dual_decoder_stageA_head1.pt"
    state = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(state, strict=False)
    print(f"[StageB] loaded StageA checkpoint from {ckpt_path}")

    model.train()

    # Stage B：冻结 head1，只训练 fusion + head2
    for p in model.head1.parameters():
        p.requires_grad = False
    for p in model.fusion.parameters():
        p.requires_grad = True
    for p in model.head2.parameters():
        p.requires_grad = True

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    print(f"[StageB] trainable params: {sum(p.numel() for p in trainable_params)/1e6:.1f} M")

    opt = torch.optim.AdamW(
        trainable_params,
        lr=2e-4,
        betas=(0.9, 0.95),
        weight_decay=0.1,
    )

    use_bf16 = torch.cuda.is_bf16_supported()
    autocast_dtype = torch.bfloat16 if use_bf16 else torch.float16
    scaler = torch.cuda.amp.GradScaler(enabled=(autocast_dtype == torch.float16))

    num_phases = 1 + int(config.max_speculative_steps)  # >=2
    global_step = 0
    opt.zero_grad(set_to_none=True)

    for step, batch in enumerate(loader):
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(device, non_blocking=True)

        with torch.autocast(device_type="cuda", dtype=autocast_dtype):
            out = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                num_phases=num_phases,
                temperature=1.0,
            )
            if out.head2_loss is None:
                # 说明实际只跑了 phase1（可能 seq_len 不够长）；直接跳过这步
                continue

            # 这里重点优化 head2_loss，你也可以用 out.loss
            loss = out.head2_loss / GRAD_ACCUM

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

        if global_step % 20 == 0:
            print(
                f"[StageB] step={global_step} "
                f"h1={out.head1_loss.item():.4f} "
                f"h2={out.head2_loss.item():.4f} "
                f"phases={num_phases}"
            )

        global_step += 1
        if global_step >= 3000:
            break

    os.makedirs("checkpoints", exist_ok=True)
    student_state = {
        "head1": model.head1.state_dict(),
        "fusion": model.fusion.state_dict(),
        "head2": model.head2.state_dict(),
    }
    ckpt_path = "checkpoints/dual_decoder_stageB_student.pt"
    torch.save(student_state, ckpt_path)
    print(f"[StageB] saved student-only checkpoint to {ckpt_path}")



if __name__ == "__main__":
    main()
