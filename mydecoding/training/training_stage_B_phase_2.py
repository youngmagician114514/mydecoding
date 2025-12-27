# mydecoding/training/training_stage_B.py
"""
Stage B: 训练 fusion + head2（在 head1 已训练好、base 冻结的基础上）

- 加载 Stage A head1 checkpoint（head1_state_dict）
- 冻结 base_model / head1
- 只训练 fusion + head2
- num_phases = 1 + max_speculative_steps (>=2)
- loss 使用 out.head2_loss（或 out.total_loss 也行，但梯度只会流向可训练模块）
- ✅重要：按有效长度裁剪，避免 padding 污染 teacher 对齐
- ✅checkpoint 只保存 fusion + head2，不保存 base（避免 9GB）
- ✅最后强制保存 _last
"""

import os
import random
from collections import deque

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from datasets import load_dataset
from transformers import AutoTokenizer

from mydecoding.models.dual_decoder import DualDecoderModel, DualDecoderConfig


# ===========================
# Utils
# ===========================
def count_trainable_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6


def freeze_all(model):
    for p in model.parameters():
        p.requires_grad = False

def get_last_logits_from_output(out):
    """
    兼容你不同版本 DualDecoderOutput 字段：
      - out.head2_logits
      - out.draft_logits
      - out.logits
    """
    if hasattr(out, "head2_logits") and out.head2_logits is not None:
        return out.head2_logits
    if hasattr(out, "draft_logits") and out.draft_logits is not None:
        return out.draft_logits
    if hasattr(out, "logits") and out.logits is not None:
        return out.logits
    raise AttributeError("Cannot find logits in DualDecoderOutput.")


@torch.no_grad()
def compute_phase_match_and_stats(
    model,
    input_ids,
    attention_mask,
    head2_logits,
    teacher_pos,
):
    """
    head2_logits: (B,V)
    teacher_pos: base_logits[:, teacher_pos, :] predicts token at teacher_pos+1
    """
    base_out = model.base_model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        return_dict=True,
    )
    base_logits = base_out.logits  # (B,T,V)
    T = base_logits.size(1)
    teacher_pos = max(0, min(int(teacher_pos), T - 1))

    teacher_logits = base_logits[:, teacher_pos, :]  # (B,V)

    student_top1 = head2_logits.argmax(dim=-1)
    teacher_top1 = teacher_logits.argmax(dim=-1)
    match = (student_top1 == teacher_top1).float().mean().item()

    # 分布统计
    ps = torch.softmax(head2_logits.float(), dim=-1)
    top1_prob = ps.max(dim=-1).values.mean().item()
    entropy = (-ps * ps.clamp_min(1e-12).log()).sum(dim=-1).mean().item()
    abs_mean = head2_logits.float().abs().mean().item()
    std = head2_logits.float().std().item()

    return {
        "match": match,
        "top1_prob": top1_prob,
        "entropy": entropy,
        "abs_mean": abs_mean,
        "std": std,
    }



def unfreeze_fusion_and_head2(model):
    # fusion
    if hasattr(model, "fusion") and model.fusion is not None:
        for p in model.fusion.parameters():
            p.requires_grad = True
    # head2
    if hasattr(model, "head2") and model.head2 is not None:
        for p in model.head2.parameters():
            p.requires_grad = True

    # ✅保险：lm_head 必须冻结（head2 内部已冻结，但这里再兜底一次）
    if hasattr(model, "base_model"):
        lm_head = model.base_model.get_output_embeddings()
        if lm_head is not None:
            for p in lm_head.parameters():
                p.requires_grad = False


def print_trainable_names(model, max_lines=120):
    print("[StageB] Trainable parameter names (first few):")
    c = 0
    for n, p in model.named_parameters():
        if p.requires_grad:
            print(f"  {n:<70s} {p.numel()/1e6:.6f} M")
            c += 1
            if c >= max_lines:
                break
    if c == 0:
        print("  (none!)")


def load_head1_ckpt_into_model(model, stageA_ckpt_path):
    """
    支持两种 ckpt：
      1) {"head1_state_dict": ...}   （你 StageA 新保存方式）
      2) {"model_state_dict": ...}   （老方式）
    """
    state = torch.load(stageA_ckpt_path, map_location="cpu")

    if "head1_state_dict" in state:
        model.head1.load_state_dict(state["head1_state_dict"], strict=False)
        print(f"[StageB] loaded head1 from {stageA_ckpt_path} (head1_state_dict)")
        return

    if "model_state_dict" in state:
        model.load_state_dict(state["model_state_dict"], strict=False)
        print(f"[StageB] loaded model from {stageA_ckpt_path} (model_state_dict)")
        return

    # 兜底：直接是 state_dict
    model.load_state_dict(state, strict=False)
    print(f"[StageB] loaded model from {stageA_ckpt_path} (raw state_dict)")


def save_stageB_ckpt(model, optimizer, step, out_dir="checkpoints", tag=""):
    """
    只保存 fusion + head2（学生模块）
    """
    os.makedirs(out_dir, exist_ok=True)
    name = f"stageB_student_step{step}{tag}.pt"
    path = os.path.join(out_dir, name)

    torch.save(
        {
            "step": step,
            "fusion_state_dict": model.fusion.state_dict() if hasattr(model, "fusion") else None,
            "head2_state_dict": model.head2.state_dict() if hasattr(model, "head2") else None,
            "optimizer_state_dict": optimizer.state_dict(),
            "config": getattr(model, "config", None),
        },
        path,
    )
    print(f"[StageB] ckpt saved: {path}")


@torch.no_grad()
def compute_teacher_logits_for_pos(model: DualDecoderModel, input_ids: torch.Tensor, attention_mask: torch.Tensor, pos: int):
    """
    teacher logits at position pos:
      base_logits[:, pos, :] predicts token at pos+1
    """
    base_out = model.base_model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        output_hidden_states=False,
        return_dict=True,
    )
    base_logits = base_out.logits  # (B,T,V)
    pos = max(0, min(pos, base_logits.size(1) - 1))
    return base_logits[:, pos, :]


@torch.no_grad()
def debug_metrics(logits: torch.Tensor, teacher_logits: torch.Tensor):
    """
    logits: (B,V), teacher_logits: (B,V)
    """
    s = logits.float()
    t = teacher_logits.float()

    ps = torch.softmax(s, dim=-1)
    top1_prob = ps.max(dim=-1).values.mean().item()

    top1_s = s.argmax(dim=-1)
    top1_t = t.argmax(dim=-1)
    match = (top1_s == top1_t).float().mean().item()

    entropy = (-ps * ps.clamp_min(1e-12).log()).sum(dim=-1).mean().item()
    abs_mean = s.abs().mean().item()
    std = s.std().item()

    return dict(top1_prob=top1_prob, match=match, entropy=entropy, abs_mean=abs_mean, std=std)


# ===========================
# Main
# ===========================
def main():
    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("results", exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[StageB] device={device}")

    # ===========================
    # Hyperparams
    # ===========================
    base_name = "Qwen/Qwen2.5-3B"

    SEQ_LEN = 256
    BATCH_SIZE = 1
    GRAD_ACCUM = 8
    MAX_STEPS = 20000

    LR = 2e-4
    WEIGHT_DECAY = 0.1
    TEMPERATURE = 0.7

    # StageB: 训练多 phase
    MAX_SPEC_STEPS = 1  # 你想 rollout 的步数（phase2..phase(K)）
    NUM_PHASES = 1 + MAX_SPEC_STEPS  # phase1=head1 + 后续 head2 rollout
    assert NUM_PHASES >= 2

    LOG_INTERVAL = 50
    SAVE_INTERVAL = 20000

    # ✅有效长度最小约束：必须 >= NUM_PHASES + 1（否则 teacher 对齐会落在 PAD/越界）
    MIN_EFF_LEN = max(64, NUM_PHASES + 1)

    autocast_dtype = torch.bfloat16 if device == "cuda" else torch.float32
    use_scaler = (device == "cuda" and autocast_dtype == torch.float16)
    scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)

    # ===========================
    # Config & Tokenizer
    # ===========================
    config = DualDecoderConfig(
        base_model_name_or_path=base_name,
        num_draft_candidates=10,
        max_speculative_steps=MAX_SPEC_STEPS,

        draft_hidden_size=1024,
        draft_num_layers=4,

        fusion_hidden_size=1024,
        fusion_num_heads=4,
        fusion_dropout=0.0,
        num_fuion_candidates=10,
        
        decoder_num_layers=2,
        decoder_num_heads=8,
        decoder_dropout=0.0,
    )

    tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ===========================
    # Dataset
    # ===========================
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
    # Model
    # ===========================
    model = DualDecoderModel(config).to(device)
    model.train()

    # ---- load stageA head1
    stageA_ckpt_path = "checkpoints/stageA_head1_step5000_last.pt"  # <<< 改成你的 head1 ckpt
    load_head1_ckpt_into_model(model, stageA_ckpt_path)

    # ---- freeze base + head1; train fusion + head2
    freeze_all(model)
    unfreeze_fusion_and_head2(model)

    trainable_m = count_trainable_params(model)
    print(f"[StageB] trainable params: {trainable_m:.1f} M")
    print_trainable_names(model)

    optimizer = AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=LR,
        betas=(0.9, 0.95),
        weight_decay=WEIGHT_DECAY,
    )

    # ===========================
    # Training loop
    # ===========================
    global_step = 0
    loss_deque = deque(maxlen=20)

    idx = 0
    N = len(tokenized)
    while global_step < MAX_STEPS:
        sample = tokenized[idx % N]
        idx += 1


        input_ids_full = sample["input_ids"].unsqueeze(0).to(device)       # (1,SEQ_LEN)
        attn_full = sample["attention_mask"].unsqueeze(0).to(device)       # (1,SEQ_LEN)

        eff_len = int(attn_full.sum(dim=-1).item())
        if eff_len < MIN_EFF_LEN:
            continue

        # ✅关键：按有效长度裁剪，避免 teacher 对齐落在 PAD 上
        input_ids_full = input_ids_full[:, :eff_len]
        attn_full = attn_full[:, :eff_len]
        T = eff_len

        # 再确保足够长：T >= NUM_PHASES + 1
        if T < (NUM_PHASES + 1):
            continue

        # 训练时我们用 prefix = [: ctx_len + K] 这种方式让多 phase 有 teacher
        # 你 dual_decoder.py 内部规则是：
        #   ctx_len = T - K
        #   head1 预测 pos=ctx_len-1 -> token at ctx_len
        #   head2 phase2 预测 pos=ctx_len -> token at ctx_len+1
        #   ...
        # 所以这里直接把完整序列喂给 model，并指定 num_phases=NUM_PHASES
        with torch.autocast(device_type="cuda", dtype=autocast_dtype, enabled=(device == "cuda")):
            out = model(
                input_ids=input_ids_full,
                attention_mask=attn_full,
                num_phases=NUM_PHASES,
                temperature=TEMPERATURE,
                train_stage="head2",
                teacher_mode="base_greedy",
            )

            # 优先用 head2_loss（如果你 forward 里返回），否则用 total_loss
            if out.head2_loss is None:
                # 这通常发生在 T 不够或你 forward 里 K=1 退化
                continue

            step_loss = out.head2_loss
            loss = step_loss / GRAD_ACCUM

        if use_scaler:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        loss_val = float(step_loss.item())
        loss_deque.append(loss_val)

        if (global_step + 1) % GRAD_ACCUM == 0:
            if use_scaler:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        # =====================
        # Debug logs
        # =====================
        if global_step % LOG_INTERVAL == 0:
            # ==========================
            # ✅ Debug: 分别计算 phase2 / phase3 的 match（严格对齐）
            # ==========================
            # 你当前训练的 NUM_PHASES = 1 + MAX_SPEC_STEPS
            # 若 MAX_SPEC_STEPS=2，则 NUM_PHASES=3 => phase2+phase3 都存在

            model_dtype = next(model.parameters()).dtype
            with torch.autocast(device_type="cuda", dtype=model_dtype, enabled=(device == "cuda")):
                out2 = model(
                    input_ids=input_ids_full,
                    attention_mask=attn_full,
                    num_phases=2,
                    temperature=TEMPERATURE,
                    train_stage="head2",
                )
            logits_p2 = get_last_logits_from_output(out2)  # phase2 logits (B,V)

            # phase2 teacher_pos:
            # ctx_len = T - K, 这里 K=2
            # teacher_pos = ctx_len - 1 + (2-1) = ctx_len
            ctx_len_2 = T - 2
            teacher_pos_2 = ctx_len_2
            m2 = compute_phase_match_and_stats(model, input_ids_full, attn_full, logits_p2, teacher_pos_2)

            # 只有当你训练 phases>=3 时才算 phase3
            m3 = None
            if NUM_PHASES >= 3:
                with torch.autocast(device_type="cuda", dtype=model_dtype, enabled=(device == "cuda")):
                    out3 = model(
                        input_ids=input_ids_full,
                        attention_mask=attn_full,
                        num_phases=3,
                        temperature=TEMPERATURE,
                        train_stage="head2",
                    )
                logits_p3 = get_last_logits_from_output(out3)  # phase3 logits (B,V)

                # phase3 teacher_pos:
                # ctx_len = T - K, 这里 K=3
                # teacher_pos = ctx_len - 1 + (3-1) = ctx_len + 1
                ctx_len_3 = T - 3
                teacher_pos_3 = ctx_len_3 + 1
                m3 = compute_phase_match_and_stats(model, input_ids_full, attn_full, logits_p3, teacher_pos_3)

            # ==========================
            # Print（分别输出 phase2/phase3）
            # ==========================
            if m3 is None:
                print(
                    f"[StageB][step {global_step}] loss={loss_val:.4f} "
                    f"| P2: match={m2['match']:.4f} top1_prob={m2['top1_prob']:.6g} "
                    f"entropy={m2['entropy']:.4f} logits_abs_mean={m2['abs_mean']:.4f} logits_std={m2['std']:.4f} "
                    f"(T={T}, phases={NUM_PHASES})"
                )
            else:
                print(
                    f"[StageB][step {global_step}] loss={loss_val:.4f} "
                    f"| P2: match={m2['match']:.4f} top1_prob={m2['top1_prob']:.6g} "
                    f"entropy={m2['entropy']:.4f} "
                    f"| P3: match={m3['match']:.4f} top1_prob={m3['top1_prob']:.6g} "
                    f"entropy={m3['entropy']:.4f} "
                    f"(T={T}, phases={NUM_PHASES})"
                )


        # =====================
        # Save ckpt (student only)
        # =====================
        if global_step > 0 and global_step % SAVE_INTERVAL == 0:
            save_stageB_ckpt(model, optimizer, global_step, out_dir="checkpoints")

        global_step += 1

    # ✅最后强制保存
    save_stageB_ckpt(model, optimizer, global_step, out_dir="checkpoints", tag="_last")
    print("[StageB] done.")


if __name__ == "__main__":
    main()
