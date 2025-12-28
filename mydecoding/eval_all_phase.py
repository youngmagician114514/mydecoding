# eval_all_phases_plus_basedriven.py
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
    # 你的 DualDecoderOutput 一般用 draft_logits 存 head1 logits
    if hasattr(out, "draft_logits") and out.draft_logits is not None:
        return out.draft_logits
    if hasattr(out, "head1_logits") and out.head1_logits is not None:
        return out.head1_logits
    if hasattr(out, "logits") and out.logits is not None:
        return out.logits
    raise AttributeError("Cannot find head1 logits in output (draft_logits/head1_logits/logits).")


def get_head2_logits(out):
    # 你的 DualDecoderOutput 用 head2_logits 存最后一次 head2 logits（num_phases=2 -> phase2, num_phases=3 -> phase3）
    if hasattr(out, "head2_logits") and out.head2_logits is not None:
        return out.head2_logits
    if hasattr(out, "fusion_logits") and out.fusion_logits is not None:
        return out.fusion_logits
    if hasattr(out, "logits") and out.logits is not None:
        return out.logits
    raise AttributeError("Cannot find head2 logits in output (head2_logits/fusion_logits/logits).")


# -----------------------
# Base model helpers
# -----------------------
@torch.no_grad()
def base_next_logits(base_model, input_ids, attention_mask):
    """
    return base next-token logits at last position: (V,)
    """
    out = base_model(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
    return out.logits[0, -1, :].float()


@torch.no_grad()
def base_logits_at_pos(base_model, input_ids, attention_mask, pos: int):
    """
    base_logits[:, pos, :] predicts token at pos+1
    return probs at that pos: (V,)
    """
    out = base_model(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
    logits = out.logits[0, pos, :].float()
    probs = F.softmax(logits, dim=-1)
    return probs


def entropy(p: torch.Tensor) -> float:
    return float((-p * p.clamp_min(1e-12).log()).sum().item())


# -----------------------
# Student helpers (phase forward)
# -----------------------
@torch.no_grad()
def student_topk_for_phase(
    model,
    input_ids,
    attention_mask,
    phase: int,
    topk: int,
    temperature: float = 1.0,
):
    """
    返回该 phase 的 student topk token ids 列表 + probs
    注意：你的 forward(num_phases=phase) 会内部执行 head1 + (head2 rollout)
    - phase=1 -> out.draft_logits
    - phase=2 -> out.head2_logits  (phase2 logits)
    - phase=3 -> out.head2_logits  (phase3 logits)
    """
    dtype = next(model.parameters()).dtype
    with torch.autocast(device_type="cuda", dtype=dtype, enabled=(input_ids.is_cuda)):
        out = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            num_phases=phase,
            temperature=temperature,
            train_stage="head2",
            infer_mode =True,
        )

    if phase == 1:
        logits = get_head1_logits(out)[0].float()  # (V,)
    else:
        logits = get_head2_logits(out)[0].float()

    probs = F.softmax(logits, dim=-1)
    topk_ids = torch.topk(probs, k=topk).indices.tolist()
    top1_prob = float(probs.max().item())
    return topk_ids, probs, top1_prob


# -----------------------
# Eval: teacher-forced (your original style)
# -----------------------
def eval_one_phase_teacher_forced(
    model,
    input_ids,
    attention_mask,
    phase: int,
    topk: int,
    temperature: float = 1.0,
):
    """
    复刻你原来的逻辑：用 base_logits 在指定 teacher_pos 上取 teacher_top1
    然后看 teacher_top1 是否在 student topK
    """
    T = input_ids.size(1)
    assert phase in (1, 2, 3)

    # student
    student_topk_ids, probs, top1_prob = student_topk_for_phase(
        model, input_ids, attention_mask, phase=phase, topk=topk, temperature=temperature
    )

    # teacher_pos 对齐保持你原来的写法（但这套对齐只能反映你当时的“最后token预测”对齐，不对应真实推理）
    if phase == 1:
        teacher_pos = T - 2
    elif phase == 2:
        ctx_len = T - 2
        teacher_pos = ctx_len
    else:
        ctx_len = T - 3
        teacher_pos = ctx_len + 1

    teacher_probs = base_logits_at_pos(model.base_model, input_ids, attention_mask, pos=teacher_pos)
    teacher_top1 = int(torch.argmax(teacher_probs).item())

    covered = int(teacher_top1 in student_topk_ids)
    rank = (student_topk_ids.index(teacher_top1) + 1) if covered else None

    student_top1 = int(torch.argmax(probs).item())
    base_prob_student_top1 = float(teacher_probs[student_top1].item())

    return {
        "covered": covered,
        "rank": rank,
        "base_prob_student_top1": base_prob_student_top1,
        "top1_prob": top1_prob,
        "entropy": entropy(probs),
    }


# -----------------------
# Eval: base-driven (what actually matters for speculative acceptance)
# -----------------------
@torch.no_grad()
def eval_base_driven_tree(
    model,
    input_ids,
    attention_mask,
    topk: int,
    temperature: float = 1.0,
):
    """
    这是关键评估：
    1) 用 base 自己的 argmax 作为 base_top1/base_top2/base_top3（条件化 prefix）
    2) 分别检查：
       base_top1 ∈ C1(topK)    (phase1 student under original prefix)
       base_top2 ∈ C2(topK)    (phase2 student under original prefix)
       base_top3 ∈ C3(topK)    (phase3 student under original prefix)
       以及 TreeCoverage@K^3: (base_top1, base_top2, base_top3) 是否在 C1×C2×C3
    """
    device = input_ids.device

    # ---- base rollout 3 steps (greedy) ----
    # step1
    base_logits1 = base_next_logits(model.base_model, input_ids, attention_mask)
    base_top1 = int(torch.argmax(base_logits1).item())

    ids1 = torch.cat([input_ids, torch.tensor([[base_top1]], device=device, dtype=torch.long)], dim=1)
    m1 = torch.cat([attention_mask, torch.ones((1, 1), device=device, dtype=attention_mask.dtype)], dim=1)

    # step2
    base_logits2 = base_next_logits(model.base_model, ids1, m1)
    base_top2 = int(torch.argmax(base_logits2).item())

    ids2 = torch.cat([ids1, torch.tensor([[base_top2]], device=device, dtype=torch.long)], dim=1)
    m2 = torch.cat([m1, torch.ones((1, 1), device=device, dtype=attention_mask.dtype)], dim=1)

    # step3
    base_logits3 = base_next_logits(model.base_model, ids2, m2)
    base_top3 = int(torch.argmax(base_logits3).item())

    # ---- student candidate sets under ORIGINAL prefix (global rollout sets) ----
    C1, p1, top1prob1 = student_topk_for_phase(model, input_ids, attention_mask, phase=1, topk=topk, temperature=temperature)
    C2, p2, top1prob2 = student_topk_for_phase(model, input_ids, attention_mask, phase=2, topk=topk, temperature=temperature)
    C3, p3, top1prob3 = student_topk_for_phase(model, input_ids, attention_mask, phase=3, topk=topk, temperature=temperature)

    in1 = int(base_top1 in C1)
    in2 = int(base_top2 in C2)
    in3 = int(base_top3 in C3)

    tree_covered = int(in1 and in2 and in3)  # base greedy path is in the cartesian product

    # 额外给一些解释性统计：base 对 student top1 的概率（越高越容易 accept）
    baseprob_student1 = float(F.softmax(base_logits1, dim=-1)[int(torch.argmax(p1).item())].item())
    baseprob_student2 = float(F.softmax(base_logits2, dim=-1)[int(torch.argmax(p2).item())].item())
    baseprob_student3 = float(F.softmax(base_logits3, dim=-1)[int(torch.argmax(p3).item())].item())

    return {
        "base_top_tokens": (base_top1, base_top2, base_top3),
        "in_C1": in1,
        "in_C2": in2,
        "in_C3": in3,
        "tree_covered": tree_covered,
        "baseprob_student_top1_phase1": baseprob_student1,
        "baseprob_student_top1_phase2": baseprob_student2,
        "baseprob_student_top1_phase3": baseprob_student3,
        "entropy1": entropy(p1),
        "entropy2": entropy(p2),
        "entropy3": entropy(p3),
        "top1prob1": top1prob1,
        "top1prob2": top1prob2,
        "top1prob3": top1prob3,
    }


# -----------------------
# Main
# -----------------------
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("[eval] device =", device)

    # ========== 改成你的路径 ==========
    stageA_head1_ckpt = "training/checkpoints/stageA_head1_step5000_last.pt"
    stageB_student_ckpt = "training/checkpoints/stageB_student_step5000_last.pt"
    # =================================

    base_name = "Qwen/Qwen2.5-3B"
    SEQ_LEN = 256
    NUM_SAMPLES = 200
    TOPK = 3
    temperature = 0.7

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

    model = DualDecoderModel(config).to(device)
    model.eval()

    model = load_stageA_head1(model, stageA_head1_ckpt)
    model = load_stageB_student(model, stageB_student_ckpt)

    # ---------------- teacher-forced stats (your original)
    teacher_stats = {
        1: defaultdict(list),
        2: defaultdict(list),
        3: defaultdict(list),
    }

    # ---------------- base-driven stats (what matters)
    base_stats = defaultdict(list)

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

        if eff_len < 8:
            continue

        input_ids = input_ids_full[:, :eff_len]
        attention_mask = attn_full[:, :eff_len]
        T = eff_len

        # ---- teacher-forced phase1/2/3 (as before)
        for phase in (1, 2, 3):
            if phase == 2 and T < 4:
                continue
            if phase == 3 and T < 5:
                continue
            r = eval_one_phase_teacher_forced(
                model=model,
                input_ids=input_ids,
                attention_mask=attention_mask,
                phase=phase,
                topk=TOPK,
                temperature=temperature,
            )
            teacher_stats[phase]["covered"].append(r["covered"])
            if r["rank"] is not None:
                teacher_stats[phase]["rank"].append(r["rank"])
            teacher_stats[phase]["base_prob_student_top1"].append(r["base_prob_student_top1"])
            teacher_stats[phase]["top1_prob"].append(r["top1_prob"])
            teacher_stats[phase]["entropy"].append(r["entropy"])

        # ---- base-driven 3-step greedy path vs C1/C2/C3 sets + TreeCoverage@27
        rb = eval_base_driven_tree(
            model=model,
            input_ids=input_ids,
            attention_mask=attention_mask,
            topk=TOPK,
            temperature=temperature,
        )
        base_stats["in_C1"].append(rb["in_C1"])
        base_stats["in_C2"].append(rb["in_C2"])
        base_stats["in_C3"].append(rb["in_C3"])
        base_stats["tree_covered"].append(rb["tree_covered"])
        base_stats["baseprob_student_top1_phase1"].append(rb["baseprob_student_top1_phase1"])
        base_stats["baseprob_student_top1_phase2"].append(rb["baseprob_student_top1_phase2"])
        base_stats["baseprob_student_top1_phase3"].append(rb["baseprob_student_top1_phase3"])
        base_stats["entropy1"].append(rb["entropy1"])
        base_stats["entropy2"].append(rb["entropy2"])
        base_stats["entropy3"].append(rb["entropy3"])
        base_stats["top1prob1"].append(rb["top1prob1"])
        base_stats["top1prob2"].append(rb["top1prob2"])
        base_stats["top1prob3"].append(rb["top1prob3"])

        used += 1

    def mean(xs):
        return sum(xs) / max(len(xs), 1)

    # ---------------- print teacher-forced summary
    print("\n============================")
    print("[Eval Summary - teacher-forced phase1/2/3]")
    print(f"num_samples = {used}")
    print(f"TOPK = {TOPK}")
    print("----------------------------")

    for phase in (1, 2, 3):
        cov = sum(teacher_stats[phase]["covered"])
        cov_rate = cov / max(len(teacher_stats[phase]["covered"]), 1)
        avg_rank = mean(teacher_stats[phase]["rank"]) if len(teacher_stats[phase]["rank"]) > 0 else float("nan")
        mean_baseprob = mean(teacher_stats[phase]["base_prob_student_top1"])
        mean_top1 = mean(teacher_stats[phase]["top1_prob"])
        mean_ent = mean(teacher_stats[phase]["entropy"])

        label = "phase1(head1)" if phase == 1 else f"phase{phase}(head2)"
        print(f"{label} Coverage@{TOPK}: {cov}/{used} = {cov_rate:.4f}")
        print(f"{label} AvgRank (when covered): {avg_rank:.2f}")
        print(f"{label} mean base_prob(student_top1): {mean_baseprob:.6f}")
        print(f"{label} mean top1_prob: {mean_top1:.6f}")
        print(f"{label} mean entropy: {mean_ent:.6f}")
        print("----------------------------")

    # ---------------- print base-driven summary
    print("\n============================")
    print("[Eval Summary - base-driven (speculative-relevant)]")
    print(f"num_samples = {used}")
    print(f"TOPK = {TOPK}")
    print("----------------------------")
    print(f"base_top1 ∈ C1@{TOPK}: {sum(base_stats['in_C1'])}/{used} = {mean(base_stats['in_C1']):.4f}")
    print(f"base_top2 ∈ C2@{TOPK}: {sum(base_stats['in_C2'])}/{used} = {mean(base_stats['in_C2']):.4f}")
    print(f"base_top3 ∈ C3@{TOPK}: {sum(base_stats['in_C3'])}/{used} = {mean(base_stats['in_C3']):.4f}")
    print(f"TreeCoverage@{TOPK**3} (base 3-step path ∈ C1×C2×C3): {sum(base_stats['tree_covered'])}/{used} = {mean(base_stats['tree_covered']):.4f}")
    print("----------------------------")
    print(f"mean base_prob(student_top1) phase1: {mean(base_stats['baseprob_student_top1_phase1']):.6f}")
    print(f"mean base_prob(student_top1) phase2: {mean(base_stats['baseprob_student_top1_phase2']):.6f}")
    print(f"mean base_prob(student_top1) phase3: {mean(base_stats['baseprob_student_top1_phase3']):.6f}")
    print("----------------------------")
    print(f"mean entropy phase1/2/3: {mean(base_stats['entropy1']):.4f} / {mean(base_stats['entropy2']):.4f} / {mean(base_stats['entropy3']):.4f}")
    print(f"mean top1_prob phase1/2/3: {mean(base_stats['top1prob1']):.4f} / {mean(base_stats['top1prob2']):.4f} / {mean(base_stats['top1prob3']):.4f}")
    print("============================\n")


if __name__ == "__main__":
    main()
