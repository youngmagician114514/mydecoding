# eval_all_phases.py
import random
from collections import defaultdict

import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoTokenizer

from mydecoding.models.dual_decoder import DualDecoderModel
from mydecoding.config.dual_decoder import DualDecoderConfig


# -----------------------
# Loading helpers
# -----------------------
def load_stageA_head1(model, ckpt_path: str):
    st = torch.load(ckpt_path, map_location="cpu")
    if "head1_state_dict" in st:
        model.head1.load_state_dict(st["head1_state_dict"], strict=False)
        print("[load] head1_state_dict loaded")
    elif "model_state_dict" in st:
        model.load_state_dict(st["model_state_dict"], strict=False)
        print("[load] model_state_dict loaded (strict=False)")
    else:
        model.load_state_dict(st, strict=False)
        print("[load] raw state_dict loaded (strict=False)")
    return model


def load_stageB_student(model, ckpt_path: str):
    st = torch.load(ckpt_path, map_location="cpu")
    if "fusion_state_dict" in st and st["fusion_state_dict"] is not None:
        model.fusion.load_state_dict(st["fusion_state_dict"], strict=False)
        print("[load] fusion_state_dict loaded")
    if "head2_state_dict" in st and st["head2_state_dict"] is not None:
        model.head2.load_state_dict(st["head2_state_dict"], strict=False)
        print("[load] head2_state_dict loaded")
    if "model_state_dict" in st:
        model.load_state_dict(st["model_state_dict"], strict=False)
        print("[load] model_state_dict loaded (strict=False)")
    return model


# -----------------------
# Output logits getters
# -----------------------
def get_head1_logits(out):
    """
    phase1 必须拿 head1 的 logits，一般是 out.draft_logits 或 out.head1_logits。
    """
    if hasattr(out, "head1_logits") and out.head1_logits is not None:
        return out.head1_logits
    if hasattr(out, "draft_logits") and out.draft_logits is not None:
        return out.draft_logits
    # 有些版本可能用 logits 存 head1
    if hasattr(out, "logits") and out.logits is not None:
        return out.logits
    raise AttributeError("Cannot find head1 logits in output (head1_logits/draft_logits/logits).")


def get_head2_logits(out):
    """
    phase>=2 的 logits，最好直接用 out.head2_logits。
    如果没有，就 fallback 到 out.draft_logits/out.logits（但你最好在模型里显式返回 head2_logits）。
    """
    if hasattr(out, "head2_logits") and out.head2_logits is not None:
        return out.head2_logits
    if hasattr(out, "draft_logits") and out.draft_logits is not None:
        return out.draft_logits
    if hasattr(out, "logits") and out.logits is not None:
        return out.logits
    raise AttributeError("Cannot find head2 logits in output (head2_logits/draft_logits/logits).")


# -----------------------
# Teacher / metrics
# -----------------------
@torch.no_grad()
def base_teacher_probs(model, input_ids, attention_mask, teacher_pos: int):
    """
    base_logits[:, pos, :] predicts token at pos+1
    返回 teacher_pos 这一步的 next-token 概率分布 (V,)
    """
    out = model.base_model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        return_dict=True,
    )
    base_logits = out.logits  # (B,T,V)
    T = base_logits.size(1)
    teacher_pos = max(0, min(int(teacher_pos), T - 1))
    probs = F.softmax(base_logits[0, teacher_pos, :].float(), dim=-1)
    return probs


def entropy(p: torch.Tensor) -> float:
    return float((-p * p.clamp_min(1e-12).log()).sum().item())


def eval_one_phase(
    model,
    input_ids,
    attention_mask,
    phase: int,
    topk: int,
    temperature: float = 1.0,
):
    """
    phase=1 -> num_phases=1，取 head1_logits，并对齐到 teacher_pos=T-2
    phase=2 -> num_phases=2，取 head2_logits，并对齐到 teacher_pos=ctx_len (=T-2)
    phase=3 -> num_phases=3，取 head2_logits，并对齐到 teacher_pos=ctx_len+1 (=T-2)
    也就是说：phase1/2/3 都是在不同条件下预测“最后一个 token (T-1)”。
    """
    T = input_ids.size(1)
    assert phase in (1, 2, 3)

    # forward
    dtype = next(model.parameters()).dtype
    with torch.autocast(device_type="cuda", dtype=dtype, enabled=(input_ids.is_cuda)):
        out = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            num_phases=phase,
            temperature=temperature,
            train_stage="head2",  # 对 eval 不敏感，你的 forward 里只是控制 loss 分支
        )

    if phase == 1:
        logits = get_head1_logits(out)
        teacher_pos = T - 2  # base_logits[T-2] predicts token at T-1
    elif phase == 2:
        logits = get_head2_logits(out)
        ctx_len = T - 2
        teacher_pos = ctx_len
    else:  # phase == 3
        logits = get_head2_logits(out)
        ctx_len = T - 3
        teacher_pos = ctx_len + 1

    logits = logits[0].float()  # (V,)
    probs = F.softmax(logits, dim=-1)

    # teacher
    teacher_probs = base_teacher_probs(model, input_ids, attention_mask, teacher_pos=teacher_pos)
    teacher_top1 = int(torch.argmax(teacher_probs).item())

    # student topK
    student_topk_ids = torch.topk(probs, k=topk).indices.tolist()
    covered = int(teacher_top1 in student_topk_ids)
    rank = (student_topk_ids.index(teacher_top1) + 1) if covered else None

    student_top1 = int(torch.argmax(probs).item())
    base_prob_student_top1 = float(teacher_probs[student_top1].item())

    return {
        "covered": covered,
        "rank": rank,
        "base_prob_student_top1": base_prob_student_top1,
        "top1_prob": float(probs.max().item()),
        "entropy": entropy(probs),
    }


# -----------------------
# Main
# -----------------------
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("[eval] device =", device)

    # ========== 改成你的路径 ==========
    stageA_head1_ckpt = "training/checkpoints/stageA_head1_step5000_last.pt"
    stageB_student_ckpt = "training/checkpoints/stageB_student_step20000_last.pt"
    # =================================

    base_name = "Qwen/Qwen2.5-3B"
    SEQ_LEN = 256
    NUM_SAMPLES = 200
    TOPK = 3
    temperature = 0.7

    # config
    config = DualDecoderConfig(
        base_model_name_or_path=base_name,
        num_draft_candidates=3,
        max_speculative_steps=2,

        draft_hidden_size=1024,
        draft_num_layers=4,

        fusion_hidden_size=1024,
        fusion_num_heads=4,
        fusion_dropout=0.0,

        decoder_num_layers=2,
        decoder_num_heads=8,
        decoder_dropout=0.0,
    )

    tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # dataset
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")

    def tok(ex):
        return tokenizer(
            ex["text"],
            truncation=True,
            max_length=SEQ_LEN,
            padding="max_length",
        )

    tokenized = dataset.map(tok, batched=False, remove_columns=dataset.column_names)
    tokenized.set_format(type="torch", columns=["input_ids", "attention_mask"])

    # model
    model = DualDecoderModel(config).to(device)
    model.eval()

    # load ckpts
    model = load_stageA_head1(model, stageA_head1_ckpt)
    model = load_stageB_student(model, stageB_student_ckpt)

    # ---- stats ----
    stats = {
        1: defaultdict(list),
        2: defaultdict(list),
        3: defaultdict(list),
    }

    indices = list(range(len(tokenized)))
    random.shuffle(indices)

    used = 0
    for idx in indices:
        if used >= NUM_SAMPLES:
            break

        sample = tokenized[idx]
        input_ids_full = sample["input_ids"].unsqueeze(0).to(device)
        attn_full = sample["attention_mask"].unsqueeze(0).to(device)
        eff_len = int(attn_full.sum().item())

        # 最少需要能预测最后 token：T>=2
        # phase3 需要至少 T>=4（因为 ctx_len=T-3 >=1）
        if eff_len < 8:
            continue

        # ✅裁剪有效长度（避免 PAD）
        input_ids = input_ids_full[:, :eff_len]
        attention_mask = attn_full[:, :eff_len]
        T = eff_len

        # phase1/2/3 都评估
        for phase in (1, 2, 3):
            # 需要保证 forward 的 ctx_len 合法
            if phase == 2 and T < 4:
                continue
            if phase == 3 and T < 5:
                continue

            r = eval_one_phase(
                model=model,
                input_ids=input_ids,
                attention_mask=attention_mask,
                phase=phase,
                topk=TOPK,
                temperature=temperature,
            )
            stats[phase]["covered"].append(r["covered"])
            if r["rank"] is not None:
                stats[phase]["rank"].append(r["rank"])
            stats[phase]["base_prob_student_top1"].append(r["base_prob_student_top1"])
            stats[phase]["top1_prob"].append(r["top1_prob"])
            stats[phase]["entropy"].append(r["entropy"])

        used += 1

    def mean(xs):
        return sum(xs) / max(len(xs), 1)

    print("\n============================")
    print("[Eval Summary - phase1/2/3]")
    print(f"num_samples = {used}")
    print(f"TOPK = {TOPK}")
    print("----------------------------")

    for phase in (1, 2, 3):
        cov = sum(stats[phase]["covered"])
        cov_rate = cov / max(len(stats[phase]["covered"]), 1)
        avg_rank = mean(stats[phase]["rank"]) if len(stats[phase]["rank"]) > 0 else float("nan")
        mean_baseprob = mean(stats[phase]["base_prob_student_top1"])
        mean_top1 = mean(stats[phase]["top1_prob"])
        mean_ent = mean(stats[phase]["entropy"])

        label = "phase1(head1)" if phase == 1 else f"phase{phase}(head2)"
        print(f"{label} Coverage@{TOPK}: {cov}/{used} = {cov_rate:.4f}")
        print(f"{label} AvgRank (when covered): {avg_rank:.2f}")
        print(f"{label} mean base_prob(student_top1): {mean_baseprob:.6f}")
        print(f"{label} mean top1_prob: {mean_top1:.6f}")
        print(f"{label} mean entropy: {mean_ent:.6f}")
        print("----------------------------")

    print("============================\n")


if __name__ == "__main__":
    main()
