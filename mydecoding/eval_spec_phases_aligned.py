# eval_spec_phases_aligned.py
# Evaluate phase1(head1) and phase2/3(head2) candidates against BASE greedy rollout:
# g1 = base_top1 given x<=t
# g2 = base_top1 given x<=t,g1
# g3 = base_top1 given x<=t,g1,g2
#
# Metrics:
# - Coverage@K: whether g_phase is inside C_phase@K
# - AvgRank: rank of g_phase inside C_phase@K when covered
# - mean base_prob(student_top1): P_base(student_top1 | prefix + g<phase)
# - mean top1_prob: P_student(student_top1)
# - mean entropy: entropy(student distribution)

import argparse
import random
from collections import defaultdict

import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoTokenizer

from mydecoding.models.dual_decoder import DualDecoderModel
from mydecoding.config.dual_decoder import DualDecoderConfig


def load_ckpt(model: DualDecoderModel, ckpt_head1: str = None, ckpt_head2: str = None):
    if ckpt_head1:
        st = torch.load(ckpt_head1, map_location="cpu")
        sd = st.get("head1_state_dict", st.get("model_state_dict", st))
        model.head1.load_state_dict(sd, strict=False)
        print(f"[load] head1 loaded from {ckpt_head1}")
    if ckpt_head2:
        st = torch.load(ckpt_head2, map_location="cpu")
        sd = st.get("head2_state_dict", st.get("model_state_dict", st))
        model.head2.load_state_dict(sd, strict=False)
        print(f"[load] head2 loaded from {ckpt_head2}")
    return model


@torch.no_grad()
def get_base_logits_next(model: DualDecoderModel, input_ids: torch.Tensor, attention_mask: torch.Tensor):
    # logits for NEXT token after the provided prefix
    out = model.base_model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        output_hidden_states=False,
        use_cache=False,
    )
    return out.logits[:, -1, :]


def entropy_from_logits(logits: torch.Tensor) -> torch.Tensor:
    p = F.softmax(logits, dim=-1)
    logp = torch.log(p.clamp_min(1e-12))
    return -(p * logp).sum(dim=-1)


def topk_ids_from_logits(logits: torch.Tensor, k: int):
    probs = F.softmax(logits, dim=-1)
    topv, topi = torch.topk(probs, k=k, dim=-1)
    return topi, topv


def rank_in_candidates(cands: torch.Tensor, target: torch.Tensor):
    # cands: (K,), target: scalar
    eq = (cands == target).nonzero(as_tuple=False)
    if eq.numel() == 0:
        return None
    return int(eq[0].item()) + 1  # 1-indexed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_model", type=str, required=True)
    ap.add_argument("--ckpt_head1", type=str, default=None)
    ap.add_argument("--ckpt_head2", type=str, default=None)
    ap.add_argument("--num_samples", type=int, default=200)
    ap.add_argument("--topk", type=int, default=3)
    ap.add_argument("--num_phases", type=int, default=3)  # 1(head1)+2(head2 steps)
    ap.add_argument("--max_len", type=int, default=256)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--device", type=str, default="cuda")
    args = ap.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    cfg = DualDecoderConfig(base_model_name=args.base_model)
    model = DualDecoderModel(cfg)
    model = load_ckpt(model, args.ckpt_head1, args.ckpt_head2)
    model.to(args.device)
    model.eval()

    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")

    stats = {phase: defaultdict(float) for phase in range(1, args.num_phases + 1)}
    counts = {phase: defaultdict(int) for phase in range(1, args.num_phases + 1)}
    tree_hit = 0

    for n in range(args.num_samples):
        text = dataset[random.randrange(len(dataset))]["text"]
        if not text or len(text.strip()) < 5:
            continue

        enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=args.max_len)
        input_ids = enc["input_ids"].to(args.device)
        attn = enc["attention_mask"].to(args.device)

        T = input_ids.shape[1]
        K = args.num_phases
        if T < (K + 5):
            continue

        # pick a random prefix length ctx_len so that we have enough room for phases
        # We'll feed the model a "full" sequence of length ctx_len + K (to match its ctx_len = T-K logic)
        ctx_len = random.randint(5, min(T - K, args.max_len - K))
        full_ids = input_ids[:, : ctx_len + K].contiguous()
        full_attn = attn[:, : ctx_len + K].contiguous()

        # Model forward ONCE to get C1..CK and head2 logits list
        dtype = next(model.parameters()).dtype
        with torch.autocast(device_type="cuda", dtype=dtype, enabled=full_ids.is_cuda):
            out = model(
                input_ids=full_ids,
                attention_mask=full_attn,
                num_phases=K,
                temperature=args.temperature,
                train_stage="head2",
                teacher_mode="base_greedy",
            )

        # candidates tensor: (B, K, max_k)
        cand = out.candidate_ids[0]  # (K, max_k)
        # base greedy tokens: (B,K) -> g1..gK
        g = out.base_greedy_tokens[0]  # (K,)

        # teacher logits:
        # phase1: compute from base on prefix (ctx_len)
        prefix_ids = full_ids[:, :ctx_len]
        prefix_attn = full_attn[:, :ctx_len]
        base_logits_p1 = get_base_logits_next(model, prefix_ids, prefix_attn)  # (1,V)
        teacher_logits_phase = {1: base_logits_p1[0]}
        if out.teacher_logits_list is not None:
            # list for phase2..K
            for j in range(2, K + 1):
                teacher_logits_phase[j] = out.teacher_logits_list[j - 2][0]

        # student logits:
        # phase1 uses out.draft_logits
        student_logits_phase = {1: out.draft_logits[0]}
        if out.head2_logits_list is not None:
            for j in range(2, K + 1):
                student_logits_phase[j] = out.head2_logits_list[j - 2][0]
        else:
            # fallback: old behavior - only last head2 logits in out.fusion_logits
            student_logits_phase[K] = out.fusion_logits[0]

        # per-phase metrics
        per_phase_hit = []
        for phase in range(1, K + 1):
            C = cand[phase - 1, : args.topk]
            target = g[phase - 1]  # g_phase

            hit = bool((C == target).any().item())
            per_phase_hit.append(hit)

            counts[phase]["n"] += 1
            counts[phase]["hit"] += int(hit)

            if hit:
                r = rank_in_candidates(C, target)
                stats[phase]["rank_sum"] += float(r)
                counts[phase]["rank_n"] += 1

            # probs
            slogits = student_logits_phase[phase]
            tlogits = teacher_logits_phase[phase]

            s_top1 = int(torch.argmax(slogits).item())
            s_top1_prob = float(F.softmax(slogits, dim=-1)[s_top1].item())
            t_prob_of_s_top1 = float(F.softmax(tlogits, dim=-1)[s_top1].item())
            ent = float(entropy_from_logits(slogits.unsqueeze(0))[0].item())

            stats[phase]["top1_prob_sum"] += s_top1_prob
            stats[phase]["baseprob_sum"] += t_prob_of_s_top1
            stats[phase]["entropy_sum"] += ent

        # tree coverage: base path inside C1×C2×C3
        if all(per_phase_hit):
            tree_hit += 1

    # report
    print("============================")
    print("[Eval Summary - base-greedy aligned]")
    print(f"num_samples = {args.num_samples}")
    print(f"TOPK = {args.topk}")
    print("----------------------------")
    for phase in range(1, args.num_phases + 1):
        n = counts[phase]["n"]
        hit = counts[phase]["hit"]
        cov = hit / max(n, 1)

        rank_n = counts[phase]["rank_n"]
        avg_rank = stats[phase]["rank_sum"] / max(rank_n, 1)

        baseprob = stats[phase]["baseprob_sum"] / max(n, 1)
        top1prob = stats[phase]["top1_prob_sum"] / max(n, 1)
        ent = stats[phase]["entropy_sum"] / max(n, 1)

        print(f"phase{phase} Coverage@{args.topk}: {hit}/{n} = {cov:.4f}")
        if rank_n > 0:
            print(f"phase{phase} AvgRank (when covered): {avg_rank:.2f}")
        print(f"phase{phase} mean base_prob(student_top1): {baseprob:.6f}")
        print(f"phase{phase} mean top1_prob: {top1prob:.6f}")
        print(f"phase{phase} mean entropy: {ent:.6f}")
        print("----------------------------")

    tree_cov = tree_hit / max(args.num_samples, 1)
    if args.num_phases >= 3:
        print(f"TreeCoverage@{args.topk**args.num_phases}: {tree_hit}/{args.num_samples} = {tree_cov:.4f}")
    print("============================")


if __name__ == "__main__":
    main()
