# eval_prefix_phases.py
# Evaluate head1/head2 candidate coverage using a fixed prefix length t (base-driven, speculative-relevant).
# phase1: g1 = argmax P_base(.|x<=t)  check g1 ∈ C1@K (head1 topK)
# phase2: g2 = argmax P_base(.|x<=t, g1) check g2 ∈ C2@K (head2 phase2 topK conditioned on (Z1, x<=t))
# phase3 (optional): g3 = argmax P_base(.|x<=t, g1, g2) check g3 ∈ C3@K (head2 phase3 topK)

import argparse
import random
from collections import defaultdict

import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoTokenizer

from mydecoding.models.dual_decoder import DualDecoderModel
from mydecoding.config.dual_decoder import DualDecoderConfig


# -----------------------
# Loading helpers (same style as your existing eval scripts)
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
    elif "model_state_dict" in st:
        model.load_state_dict(st["model_state_dict"], strict=False)
        print("[load] model_state_dict loaded (strict=False)")
    else:
        # allow loading raw state dicts too
        try:
            model.load_state_dict(st, strict=False)
            print("[load] raw state_dict loaded (strict=False)")
        except Exception:
            pass
    return model


@torch.no_grad()
def base_greedy_next(base_model, input_ids: torch.Tensor, attention_mask: torch.Tensor, temperature: float = 1.0):
    """
    input_ids: (B, T)
    return:
      next_id: (B,)
      probs:   (B, V) softmax over last-position logits
      logits:  (B, V)
    """
    out = base_model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        use_cache=False,
        output_hidden_states=False,
    )
    logits = out.logits[:, -1, :].float()  # (B,V)
    probs = F.softmax(logits / max(float(temperature), 1e-6), dim=-1)
    next_id = probs.argmax(dim=-1)
    return next_id, probs, logits


def entropy_from_logits(logits: torch.Tensor, temperature: float = 1.0):
    p = F.softmax(logits.float() / max(float(temperature), 1e-6), dim=-1)
    # avoid nan
    ent = -(p * torch.log(p.clamp_min(1e-12))).sum(dim=-1)
    return ent


def top1_prob_from_logits(logits: torch.Tensor, temperature: float = 1.0):
    p = F.softmax(logits.float() / max(float(temperature), 1e-6), dim=-1)
    return p.max(dim=-1).values


def rank_in_topk(cand_ids: torch.Tensor, target_id: torch.Tensor):
    """
    cand_ids: (B, K) sorted by student prob descending
    target_id: (B,)
    return rank (B,) where 1..K if found else 0
    """
    # (B,K)
    eq = (cand_ids == target_id.unsqueeze(1))
    found = eq.any(dim=1)
    # position = first True index
    # use argmax on int mask; but argmax returns 0 if all zeros -> gate by found
    pos0 = eq.int().argmax(dim=1)  # 0..K-1
    rank = torch.where(found, pos0 + 1, torch.zeros_like(pos0))
    return rank, found


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stageA_head1_ckpt", type=str, required=True)
    ap.add_argument("--stageB_student_ckpt", type=str, required=True)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--num_samples", type=int, default=200)
    ap.add_argument("--topk", type=int, default=3)
    ap.add_argument("--seq_len", type=int, default=256)
    ap.add_argument("--min_t", type=int, default=32, help="minimum prefix length t")
    ap.add_argument("--num_phases", type=int, default=3, choices=[2, 3], help="2: phase1+phase2; 3: phase1+phase2+phase3")
    ap.add_argument("--student_temperature", type=float, default=1.0)
    ap.add_argument("--base_temperature", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=12)
    args = ap.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device(args.device)

    config = DualDecoderConfig()
    tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")

    def tok(ex):
        return tokenizer(
            ex["text"],
            truncation=True,
            max_length=args.seq_len,
            padding="max_length",
        )

    tokenized = dataset.map(tok, batched=False, remove_columns=dataset.column_names)
    tokenized.set_format(type="torch", columns=["input_ids", "attention_mask"])

    model = DualDecoderModel(config).to(device)
    model.eval()
    model = load_stageA_head1(model, args.stageA_head1_ckpt)
    model = load_stageB_student(model, args.stageB_student_ckpt)
    model.eval()

    base_model = model.base_model
    base_model.eval()

    dtype = next(model.parameters()).dtype

    stats = {1: defaultdict(list), 2: defaultdict(list), 3: defaultdict(list)}

    indices = list(range(len(tokenized)))
    random.shuffle(indices)

    used = 0
    for idx in indices:
        if used >= args.num_samples:
            break

        sample = tokenized[idx]
        input_ids_full = sample["input_ids"].unsqueeze(0).to(device)
        attn_full = sample["attention_mask"].unsqueeze(0).to(device)
        eff_len = int(attn_full.sum().item())
        if eff_len < max(args.min_t + 2, 8):
            continue

        # choose a random prefix length t, ensure we can do at least two base greedy steps
        # (we don't need ground-truth future tokens, but avoid too-short contexts)
        t_max = eff_len - 1
        t = random.randint(args.min_t, t_max)

        prefix_ids = input_ids_full[:, :t]
        prefix_attn = attn_full[:, :t]

        # ---- base greedy g1/g2/(g3) ----
        with torch.autocast(device_type="cuda", dtype=dtype, enabled=prefix_ids.is_cuda):
            g1, base_probs1, _ = base_greedy_next(base_model, prefix_ids, prefix_attn, temperature=args.base_temperature)

            ids2 = torch.cat([prefix_ids, g1.unsqueeze(1)], dim=1)
            attn2 = torch.cat([prefix_attn, torch.ones_like(g1.unsqueeze(1))], dim=1)
            g2, base_probs2, _ = base_greedy_next(base_model, ids2, attn2, temperature=args.base_temperature)

            if args.num_phases >= 3:
                ids3 = torch.cat([ids2, g2.unsqueeze(1)], dim=1)
                attn3 = torch.cat([attn2, torch.ones_like(g2.unsqueeze(1))], dim=1)
                g3, base_probs3, _ = base_greedy_next(base_model, ids3, attn3, temperature=args.base_temperature)
            else:
                g3 = None
                base_probs3 = None

            # ---- student one forward (infer_mode=True) ----
            out = model(
                input_ids=prefix_ids,
                attention_mask=prefix_attn,
                num_phases=args.num_phases,
                temperature=args.student_temperature,
                train_stage="head2",
                infer_mode=True,
            )

        # candidate groups: (B, K, max_k)
        cand = out.candidate_ids
        k = args.topk

        # logits
        head1_logits = out.draft_logits  # (B,V)
        head2_logits_all = getattr(out, "head2_logits_all", None)  # (B, K-1, V) or None

        # ---------------- phase1(head1): g1 ∈ C1@K ----------------
        C1 = cand[:, 0, :k]
        rank1, hit1 = rank_in_topk(C1, g1)
        stats[1]["hit"].append(hit1.item())
        if hit1.item():
            stats[1]["rank"].append(rank1.item())

        # student top1 prob / entropy
        stats[1]["top1_prob"].append(top1_prob_from_logits(head1_logits, args.student_temperature).item())
        stats[1]["entropy"].append(entropy_from_logits(head1_logits, args.student_temperature).item())

        # base_prob(student_top1)
        y1 = head1_logits.float().argmax(dim=-1)  # (B,)
        stats[1]["base_prob_student_top1"].append(base_probs1.gather(-1, y1.unsqueeze(1)).squeeze(1).item())

        # ---------------- phase2(head2): g2 ∈ C2@K ----------------
        if args.num_phases >= 2 and cand.size(1) >= 2:
            C2 = cand[:, 1, :k]
            # get phase2 logits
            if head2_logits_all is not None and head2_logits_all.size(1) >= 1:
                head2_p2_logits = head2_logits_all[:, 0, :]
            else:
                head2_p2_logits = out.head2_logits  # fallback: last logits

            rank2, hit2 = rank_in_topk(C2, g2)
            stats[2]["hit"].append(hit2.item())
            if hit2.item():
                stats[2]["rank"].append(rank2.item())

            stats[2]["top1_prob"].append(top1_prob_from_logits(head2_p2_logits, args.student_temperature).item())
            stats[2]["entropy"].append(entropy_from_logits(head2_p2_logits, args.student_temperature).item())

            y2 = head2_p2_logits.float().argmax(dim=-1)
            stats[2]["base_prob_student_top1"].append(base_probs2.gather(-1, y2.unsqueeze(1)).squeeze(1).item())

        # ---------------- phase3(head2): g3 ∈ C3@K ----------------
        if args.num_phases >= 3 and cand.size(1) >= 3 and g3 is not None:
            C3 = cand[:, 2, :k]
            if head2_logits_all is not None and head2_logits_all.size(1) >= 2:
                head2_p3_logits = head2_logits_all[:, 1, :]
            else:
                head2_p3_logits = out.head2_logits  # fallback

            rank3, hit3 = rank_in_topk(C3, g3)
            stats[3]["hit"].append(hit3.item())
            if hit3.item():
                stats[3]["rank"].append(rank3.item())

            stats[3]["top1_prob"].append(top1_prob_from_logits(head2_p3_logits, args.student_temperature).item())
            stats[3]["entropy"].append(entropy_from_logits(head2_p3_logits, args.student_temperature).item())

            y3 = head2_p3_logits.float().argmax(dim=-1)
            stats[3]["base_prob_student_top1"].append(base_probs3.gather(-1, y3.unsqueeze(1)).squeeze(1).item())

        used += 1

    # ---------------- summary ----------------
    print("=" * 28)
    print("[Eval Summary - base-driven prefix=t]")
    print(f"num_samples = {used}")
    print(f"TOPK = {args.topk}")
    print(f"num_phases = {args.num_phases}")
    print("-" * 28)

    def summarize(phase: int, name: str):
        hits = stats[phase]["hit"]
        if len(hits) == 0:
            print(f"{name}: no samples")
            return
        cov = sum(hits) / len(hits)
        ranks = stats[phase]["rank"]
        avg_rank = (sum(ranks) / len(ranks)) if len(ranks) else float("nan")
        base_prob = stats[phase]["base_prob_student_top1"]
        top1_prob = stats[phase]["top1_prob"]
        ent = stats[phase]["entropy"]
        print(f"{name} Coverage@{args.topk}: {sum(hits)}/{len(hits)} = {cov:.4f}")
        if len(ranks):
            print(f"{name} AvgRank (when covered): {avg_rank:.2f}")
        else:
            print(f"{name} AvgRank (when covered): N/A")
        print(f"{name} mean base_prob(student_top1): {sum(base_prob)/len(base_prob):.6f}")
        print(f"{name} mean top1_prob: {sum(top1_prob)/len(top1_prob):.6f}")
        print(f"{name} mean entropy: {sum(ent)/len(ent):.6f}")
        print("-" * 28)

    summarize(1, "phase1(head1)")
    summarize(2, "phase2(head2)")
    if args.num_phases >= 3:
        summarize(3, "phase3(head2)")


if __name__ == "__main__":
    main()
