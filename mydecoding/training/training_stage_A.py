# mydecoding/training/training_stage_A.py
"""
Stage A: 只训练 head1（decoder1/head1），采用多位置 distillation，
让它尽量贴近 Hydra++ 单头 draft head 的效果。

- base_model 冻结，只作为 teacher
- 对每条序列随机采多个位置 t，截断到 x_{1:t}，让 head1 拟合 base 对 x_{t+1} 的分布
- loss: KL(student || teacher) 与 Hydra++ 相同类型
- 输出：仅保存 head1 的 checkpoint；画 20-step 平均 loss 曲线
"""

import os
import random
from collections import deque

import torch
from torch.utils.data import DataLoader
from datasets import load_dataset
import matplotlib.pyplot as plt

from transformers import AutoTokenizer, AutoConfig

from mydecoding.config.dual_decoder import DualDecoderConfig
from mydecoding.models.dual_decoder import DualDecoderModel


def make_dataloader(tokenizer, seq_len: int, batch_size: int) -> DataLoader:
    """用 wikitext-2 做 teacher 数据。"""
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

    # ===== base 配置，用它的 hidden_size 决定 head1 MLP 宽度 =====
    base_name = "Qwen/Qwen2.5-3B"
    base_cfg = AutoConfig.from_pretrained(base_name)
    H = base_cfg.hidden_size            # 例如 2560
    ffw = 4 * H                         # 和大模型 FFN 的中间宽度一致

    # ===== DualDecoderConfig：head1 尽量像 Hydra++ 的 draft head =====
    config = DualDecoderConfig(
        base_model_name_or_path=base_name,
        num_draft_candidates=3,          # head1 每步 top-k 草稿
        max_speculative_steps=1,         # StageA 只训练 t+1，等价单步 Hydra++
        draft_hidden_size=2560,           # head1 MLP 中间宽度
        draft_num_layers=4,              # 你可以后面试 2/4/6 比较
        fusion_hidden_size=ffw,          # StageA 不用 fusion，但先占位
        fusion_num_heads=4,
        fusion_dropout=0.1,
        decoder_num_layers=2,            # head2 留给 StageB
        decoder_num_heads=8,
        decoder_dropout=0.1,
        draft_loss_weight=1.0,
        fusion_loss_weight=1.0,
    )

    SEQ_LEN = 256
    BATCH_SIZE = 1          # 脚本里假定 batch=1，方便多位置截断
    GRAD_ACCUM = 8
    MAX_STEPS = 150000        # 有效 step 数（按“截断次数”算），你可以增大

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

    # 先全部冻结
    for p in model.parameters():
        p.requires_grad = False

    # 只训练 head1
    for p in model.head1.parameters():
        p.requires_grad = True

    # 明确保证 base 模型和 head2 / fusion 都是冻结的
    for p in model.base_model.parameters():
        p.requires_grad = False
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
    
    print("adamw finish")
    num_phases = 1  # 只跑 phase1 => 只用 head1
    global_step = 0  # 统计“有效 step 数”（每次截断算一个 step）
    opt.zero_grad(set_to_none=True)

    # ===== 20-step 平均 loss 统计 =====
    window_size = 20
    recent_losses = deque(maxlen=window_size)
    avg_steps = []   # 记录画图用的 step（对应窗口末尾的 step 编号）
    avg_losses = []  # 对应的 20-step 平均 loss

    # 多位置 distillation 的每个样本采样次数
    NUM_T_PER_SAMPLE = 4

    data_iter = iter(loader)
    while global_step < MAX_STEPS:
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            batch = next(data_iter)

        # 这里假定 batch_size=1，方便处理长度
        input_ids_full = batch["input_ids"].to(device, non_blocking=True)   # [1, L]
        attn_mask_full = batch["attention_mask"].to(device, non_blocking=True)

        # 真实长度（排除 pad）
        seq_len = int(attn_mask_full.sum(dim=-1).item())
        if seq_len <= 2:
            continue  # 太短的不训练

        # 可以采样的位置范围：t ∈ [min_t, seq_len-1]，预测的是 x_{t+1}
        min_t = 4                      # 保证有点上下文
        max_t = seq_len - 1
        if max_t <= min_t:
            continue

        # 实际采样的 t 数量
        num_t = min(NUM_T_PER_SAMPLE, max_t - min_t + 1)

        # 从 [min_t, max_t] 里随机采样不同的 t（长度）
        ts = sorted(random.sample(range(min_t, max_t + 1), num_t))

        for t in ts:
            if global_step >= MAX_STEPS:
                break

            # 截断到前 t 个 token，当成新的序列 x_{1:t}
            input_ids = input_ids_full[:, :t]
            attention_mask = attn_mask_full[:, :t]

            with torch.autocast(device_type="cuda", dtype=autocast_dtype):
                out = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    num_phases=num_phases,
                    temperature=1.0,
                )
                step_loss = out.head1_loss         # KL(student, teacher) @ 位置 t
                loss = step_loss / GRAD_ACCUM

            # 记录原始 loss（不除 grad_accum）
            loss_val = float(step_loss.item())
            recent_losses.append(loss_val)

            # backward & grad_accum
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

            global_step += 1

            # 每到一个 step，就看 recent_losses 是否满 20，满了就算一个平均值
            if len(recent_losses) == window_size:
                avg_loss = sum(recent_losses) / window_size
                avg_steps.append(global_step)
                avg_losses.append(avg_loss)

            
            # if global_step % 10 == 0:
            #     print(
            #         f"[StageA] step={global_step} "
            #         f"h1={out.head1_loss.item():.4f} "
            #     )
            
            # 控制台打印：每 100 个有效 step 打一次当前 20-step 平均
            if global_step % 100 == 0 and len(recent_losses) > 0:
                cur_avg = sum(recent_losses) / len(recent_losses)
                print(f"[StageA] step={global_step}  "
                      f"avg_loss(last {len(recent_losses)} steps)={cur_avg:.4f}")

    # ===== 保存 checkpoint：只存 head1 =====
    os.makedirs("checkpoints", exist_ok=True)
    student_state = {
        "head1": model.head1.state_dict(),
    }
    ckpt_path = "checkpoints/dual_decoder_stageA_head1_student.pt"
    torch.save(student_state, ckpt_path)
    print(f"[StageA] saved student-only checkpoint to {ckpt_path}")

    # ===== 画 20-step 平均 loss 曲线 =====
    os.makedirs("results", exist_ok=True)
    if len(avg_steps) > 0:
        plt.figure()
        plt.plot(avg_steps, avg_losses)
        plt.xlabel("step (effective, with multi-t)")
        plt.ylabel("avg KL loss (window=20)")
        plt.title("StageA head1 KL loss (20-step moving average)")
        plt.grid(True)
        out_png = os.path.join("results", "stageA_head1_loss_avg20.png")
        plt.savefig(out_png, dpi=150, bbox_inches="tight")
        print(f"[StageA] 20-step avg loss curve saved to {out_png}")
    else:
        print("[StageA] no avg loss recorded (too few steps?)")


if __name__ == "__main__":
    main()
