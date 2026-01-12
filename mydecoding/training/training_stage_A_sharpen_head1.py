import os
import random
from collections import deque

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from transformers import AutoTokenizer
from datasets import load_dataset
import matplotlib.pyplot as plt

from mydecoding.models.dual_decoder import DualDecoderModel, DualDecoderConfig


# ===========================
# Utils
# ===========================
def count_trainable_params(model: torch.nn.Module) -> float:
    return sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6


def freeze_all_but_head1(model: torch.nn.Module):
    """
    强制只训练 head1（CandidateDraftHead）相关参数。
    其它：base_model / fusion / head2 / lm_head 都冻结。
    """
    for p in model.parameters():
        p.requires_grad = False

    # 只打开 head1
    for name, p in model.named_parameters():
        if name.startswith("head1."):
            p.requires_grad = True

    # 再确认 base_model 冻结（以防万一）
    if hasattr(model, "base_model"):
        for p in model.base_model.parameters():
            p.requires_grad = False


def print_trainable_modules(model: torch.nn.Module, max_lines: int = 80):
    print("[StageA] Trainable parameter names (first few):")
    c = 0
    for n, p in model.named_parameters():
        if p.requires_grad:
            print(f"  {n:<60s} {p.numel()/1e6:.6f} M")
            c += 1
            if c >= max_lines:
                break
    if c == 0:
        print("  (none!)")


def save_head1_ckpt(model: DualDecoderModel, optimizer, step: int, out_dir="checkpoints", tag=""):
    os.makedirs(out_dir, exist_ok=True)
    name = f"stageA_sharp_head1_step{step}{tag}.pt"
    path = os.path.join(out_dir, name)

    torch.save(
        {
            "step": step,
            "head1_state_dict": model.head1.state_dict(),  # ✅只保存 head1
            "optimizer_state_dict": optimizer.state_dict(),
            "config": getattr(model, "config", None),
        },
        path,
    )
    print(f"[StageA] head1 ckpt saved: {path}")


@torch.no_grad()
def compute_teacher_logits(model: DualDecoderModel, input_ids: torch.Tensor, attention_mask: torch.Tensor):
    """
    给定 (B, T) 的 input_ids/attention_mask，返回 teacher logits:
    teacher_logits = base_logits[:, -2, :] -> 预测最后一个 token（因为 logits[i] 预测 token[i+1]）
    StageA 我们喂的是 [:t+1]，因此 teacher 的位置就是 t-1 == -2。
    """
    base_out = model.base_model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        output_hidden_states=False,
        return_dict=True,
    )
    base_logits = base_out.logits  # (B, T, V)
    teacher_logits = base_logits[:, -2, :]  # (B, V)
    return teacher_logits


@torch.no_grad()
def compute_debug_metrics(student_logits: torch.Tensor, teacher_logits: torch.Tensor):
    student_logits_f = student_logits.float()
    teacher_logits_f = teacher_logits.float()

    student_probs = torch.softmax(student_logits_f, dim=-1)

    top1_prob = student_probs.max(dim=-1).values.mean().item()

    teacher_top1 = teacher_logits_f.argmax(dim=-1)
    student_top1 = student_logits_f.argmax(dim=-1)
    match_rate = (teacher_top1 == student_top1).float().mean().item()

    logits_abs_mean = student_logits_f.abs().mean().item()
    logits_std = student_logits_f.std().item()

    entropy = (-student_probs * student_probs.clamp_min(1e-12).log()).sum(dim=-1).mean().item()

    return {
        "top1_prob": top1_prob,
        "match_rate": match_rate,
        "logits_abs_mean": logits_abs_mean,
        "logits_std": logits_std,
        "entropy": entropy,
    }


# ===========================
# Main
# ===========================
def main():
    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("results", exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[StageA] device={device}")

    # ===========================
    # 超参数
    # ===========================
    base_name = "Qwen/Qwen2.5-3B"

    SEQ_LEN = 256
    GRAD_ACCUM = 8
    MAX_STEPS = 5000

    NUM_T_PER_SAMPLE = 4
    MIN_T = 32

    LR = 2e-4
    WEIGHT_DECAY = 0.01

    # ✅温度：先调它（越小越尖）
    TEMPERATURE = 0.7


    # ✅让 head1 更“尖”的额外项：
    # - HARD_CE_W: 额外的 hard CE（以 base greedy 的 g1=argmax(teacher_logits) 为目标）
    #   典型有效区间 0.3~1.0，越大越尖，但过大可能伤 Coverage@K
    # - ENT_W: 熵惩罚（最小化熵使分布更尖），建议先设 0；如仍不尖再尝试 0.005~0.02
    HARD_CE_W = 0.5
    ENT_W = 0.0
    LOG_INTERVAL = 50
    SAVE_INTERVAL = 500
    PLOT_WINDOW = 20

    # bfloat16 默认即可（你现在也是这样跑）
    autocast_dtype = torch.bfloat16 if device == "cuda" else torch.float32
    use_scaler = (device == "cuda" and autocast_dtype == torch.float16)
    scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)

    # ===========================
    # config
    # ===========================
    config = DualDecoderConfig(
        base_model_name_or_path=base_name,
        num_draft_candidates=3,
        max_speculative_steps=1,

        # 这些字段不重要：dual_decoder.py 里会用 base_model.config.hidden_size 对齐 head1
        draft_hidden_size=1024,
        draft_num_layers=4,

        fusion_hidden_size=1024,
        fusion_num_heads=4,
        fusion_dropout=0.0,

        decoder_num_layers=2,
        decoder_num_heads=8,
        decoder_dropout=0.0,
    )

    # ===========================
    # tokenizer & dataset
    # ===========================
    tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")

    def tokenize_fn(ex):
        return tokenizer(
            ex["text"],
            truncation=True,
            max_length=SEQ_LEN,
            padding="max_length",
        )

    tokenized = dataset.map(tokenize_fn, batched=False, remove_columns=dataset.column_names)
    tokenized.set_format(type="torch", columns=["input_ids", "attention_mask"])

    # ===========================
    # model
    # ===========================
    model = DualDecoderModel(config).to(device)
    model.train()

    freeze_all_but_head1(model)

    trainable_m = count_trainable_params(model)
    print(f"[StageA] trainable params: {trainable_m:.1f} M")
    print_trainable_modules(model)

    optimizer = AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=LR,
        weight_decay=WEIGHT_DECAY,
    )

    # ===========================
    # training loop
    # ===========================
    global_step = 0
    loss_deque = deque(maxlen=PLOT_WINDOW)
    avg_losses = []
    avg_steps = []

    idx = 0
    while global_step < MAX_STEPS and idx < len(tokenized):
        sample = tokenized[idx]
        idx += 1

        input_ids_full = sample["input_ids"].unsqueeze(0).to(device)        # (1, SEQ_LEN)
        attn_mask_full = sample["attention_mask"].unsqueeze(0).to(device)   # (1, SEQ_LEN)

        seq_len = int(attn_mask_full.sum(dim=-1).item())
        if seq_len < (MIN_T + 1):
            continue

        # t ∈ [MIN_T, seq_len-1]，确保 t+1 <= seq_len
        max_t = seq_len - 1
        population = list(range(MIN_T, max_t + 1))
        if not population:
            continue

        num_t = min(NUM_T_PER_SAMPLE, len(population))
        ts = sorted(random.sample(population, num_t))

        for t in ts:
            if global_step >= MAX_STEPS:
                break

            # ✅关键：必须包含 t+1
            input_ids = input_ids_full[:, :t + 1]
            attention_mask = attn_mask_full[:, :t + 1]

            # teacher logits：base model next-token 分布
            teacher_logits = compute_teacher_logits(model, input_ids, attention_mask)  # (B,V)

            # forward + loss
            with torch.autocast(device_type="cuda", dtype=autocast_dtype, enabled=(device == "cuda")):
                out = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    num_phases=1,
                    temperature=TEMPERATURE,   # ✅温度仍用于输出/调试；loss 这里我们自行计算
                    train_stage="head1",
                )

                # ===========================
                # ✅ Head1 sharpened objective
                #   目标：同时保持 teacher 分布（soft KL）并显式拉高 g1 的峰值（hard CE）
                # ===========================
                student_logits = out.draft_logits.float()     # (B, V)
                teacher_logits_f = teacher_logits.float()     # (B, V)  (teacher 不回传梯度)

                # (1) soft distillation (KL: teacher || student) with temperature
                T = float(TEMPERATURE)
                teacher_prob = torch.softmax(teacher_logits_f / T, dim=-1).detach()
                student_logprob = torch.log_softmax(student_logits / T, dim=-1)
                loss_kl = F.kl_div(student_logprob, teacher_prob, reduction="batchmean") * (T * T)

                # (2) hard target = base greedy g1
                g1 = teacher_logits_f.argmax(dim=-1)  # (B,)
                loss_ce = F.cross_entropy(student_logits, g1)

                # (3) optional entropy penalty (minimize entropy -> sharper)
                student_prob = torch.softmax(student_logits, dim=-1)
                entropy = (-student_prob * student_prob.clamp_min(1e-12).log()).sum(dim=-1).mean()
                loss_ent = entropy

                step_loss = loss_kl + HARD_CE_W * loss_ce + ENT_W * loss_ent
                loss = step_loss / GRAD_ACCUM

            # backward
            if use_scaler:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            loss_val = float(step_loss.item())
            loss_deque.append(loss_val)

            # optimizer step (grad accum)
            if (global_step + 1) % GRAD_ACCUM == 0:
                if use_scaler:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            # debug logs
            if global_step % LOG_INTERVAL == 0:
                with torch.no_grad():
                    metrics = compute_debug_metrics(out.draft_logits, teacher_logits)

                    # 额外打印 loss 分解，便于判断“是否尖锐”来自 hard CE 还是 KL
                    student_logits = out.draft_logits.float()
                    teacher_logits_f = teacher_logits.float()
                    T = float(TEMPERATURE)
                    teacher_prob = torch.softmax(teacher_logits_f / T, dim=-1)
                    student_logprob = torch.log_softmax(student_logits / T, dim=-1)
                    dbg_loss_kl = F.kl_div(student_logprob, teacher_prob, reduction="batchmean") * (T * T)
                    dbg_g1 = teacher_logits_f.argmax(dim=-1)
                    dbg_loss_ce = F.cross_entropy(student_logits, dbg_g1)
                    student_prob = torch.softmax(student_logits, dim=-1)
                    dbg_entropy = (-student_prob * student_prob.clamp_min(1e-12).log()).sum(dim=-1).mean()

                print(
                    f"[StageA][step {global_step}] "
                    f"loss={loss_val:.4f} "
                    f"(kl={dbg_loss_kl.item():.4f}, ce={dbg_loss_ce.item():.4f}, ent={dbg_entropy.item():.4f}) "
                    f"top1_prob={metrics['top1_prob']:.6g} "
                    f"match={metrics['match_rate']:.4f} "
                    f"logits_abs_mean={metrics['logits_abs_mean']:.4f} "
                    f"logits_std={metrics['logits_std']:.4f} "
                    f"entropy={metrics['entropy']:.4f} "
                    f"(T={TEMPERATURE}, t={t}, seq_len={seq_len})"
                )

            # # save ckpt (head1 only)
            # if global_step > 0 and global_step % SAVE_INTERVAL == 0:
            #     save_head1_ckpt(model, optimizer, global_step, out_dir="checkpoints")

            # record avg loss curve
            if len(loss_deque) == PLOT_WINDOW:
                avg_losses.append(sum(loss_deque) / len(loss_deque))
                avg_steps.append(global_step)

            global_step += 1

    # ✅训练结束：强制保存 last
    save_head1_ckpt(model, optimizer, global_step, out_dir="checkpoints", tag="_last")

    # plot loss curve
    if len(avg_losses) > 0:
        plt.figure(figsize=(7, 4))
        plt.plot(avg_steps, avg_losses)
        plt.xlabel("step")
        plt.ylabel(f"avg head1 loss (window={PLOT_WINDOW})")
        plt.title(f"StageA head1 distillation loss (T={TEMPERATURE})")
        plt.grid(True)
        out_png = os.path.join("results", f"stageA_head1_loss_avg_T{str(TEMPERATURE).replace('.','p')}.png")
        plt.savefig(out_png, dpi=150, bbox_inches="tight")
        print(f"[StageA] loss curve saved to {out_png}")

    print("[StageA] done.")


if __name__ == "__main__":
    main()
