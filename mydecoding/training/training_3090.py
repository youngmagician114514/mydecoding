import os, math
import torch
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer

from mydecoding.config.dual_decoder import DualDecoderConfig
from mydecoding.models.dual_decoder import DualDecoderModel

def main():
    device = "cuda"
    torch.backends.cuda.matmul.allow_tf32 = True

    # ===== config (3090 safe) =====
    config = DualDecoderConfig(
        base_model_name_or_path="Qwen/Qwen2.5-3B",
        num_draft_candidates=3,
        max_speculative_steps=1,  # start small: total phases=2
        draft_hidden_size=1024,
        draft_num_layers=2,
        fusion_hidden_size=1024,
        fusion_num_heads=4,
        decoder_num_layers=2,
        decoder_num_heads=8,
        decoder_dropout=0.1,
        draft_loss_weight=1.0,
        fusion_loss_weight=1.0,
    )
    num_phases = 1 + config.max_speculative_steps  # phase1 + head2 steps

    # ===== data =====
    SEQ_LEN = 256
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")

    tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    def tok_fn(ex):
        out = tokenizer(ex["text"], truncation=True, max_length=SEQ_LEN, padding="max_length")
        return {"input_ids": out["input_ids"], "attention_mask": out["attention_mask"]}

    tok = ds.map(tok_fn, remove_columns=ds.column_names)
    tok.set_format(type="torch", columns=["input_ids", "attention_mask"])
    loader = DataLoader(tok, batch_size=1, shuffle=True, num_workers=2, pin_memory=True)

    # ===== model =====
    model = DualDecoderModel(config).to(device)
    model.train()

    opt = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=2e-4,
        betas=(0.9, 0.95),
        weight_decay=0.1,
    )
    grad_accum = 8  # effective batch = 8

    use_bf16 = torch.cuda.is_bf16_supported()
    autocast_dtype = torch.bfloat16 if use_bf16 else torch.float16
    scaler = torch.cuda.amp.GradScaler(enabled=(autocast_dtype == torch.float16))

    step = 0
    opt.zero_grad(set_to_none=True)

    for batch in loader:
        # DataLoader 已经给了 batch 维度，这里不要再 unsqueeze 了
        input_ids = batch["input_ids"].to(device, non_blocking=True)        # (B, T)
        attention_mask = batch["attention_mask"].to(device, non_blocking=True)

        # 可选：第一次迭代打印一下，确认是 (B,T)
        if step == 0:
            print("input_ids shape:", input_ids.shape)

        with torch.autocast(device_type="cuda", dtype=autocast_dtype):
            out = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                num_phases=num_phases,
                temperature=1.0,
            )
            loss = out.loss / grad_accum

        if scaler.is_enabled():
            scaler.scale(loss).backward()
        else:
            loss.backward()

        if step == 0:
            torch.cuda.synchronize()
            print("peak mem GB:", torch.cuda.max_memory_allocated() / 1024**3)

        if (step + 1) % grad_accum == 0:
            if scaler.is_enabled():
                scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            if scaler.is_enabled():
                scaler.step(opt)
                scaler.update()
            else:
                opt.step()
            opt.zero_grad(set_to_none=True)

        if step % 20 == 0:
            h2 = out.head2_loss.item() if out.head2_loss is not None else -1.0
            print(
                f"step={step} loss={out.loss.item():.4f} "
                f"h1={out.head1_loss.item():.4f} "
                f"h2={h2:.4f} phases={num_phases} dtype={autocast_dtype}"
            )

        step += 1
        if step >= 300:
            break

    torch.save(model.state_dict(), "dual_decoder_student_3090.pt")
    print("saved dual_decoder_student_3090.pt")

if __name__ == "__main__":
    main()
