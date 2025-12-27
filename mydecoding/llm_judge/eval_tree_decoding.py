# eval_tree40_layerwise.py
# ------------------------------------------------------------
# Tree speculative decoding (40-node tree) using *layerwise global rollout*
# - C1 from head1 (top3)
# - C2 from head2 phase2 (top3)
# - C3 from head2 phase3 (top3)
# Paths = C1 x C2 x C3 = 27
# Base verifies all 27 in ONE forward
# ------------------------------------------------------------

import os, json, time, random, argparse
from dataclasses import dataclass
from typing import List, Tuple, Dict

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from mydecoding.models.dual_decoder import DualDecoderModel
from mydecoding.config.dual_decoder import DualDecoderConfig


# -----------------------------
# IO
# -----------------------------
def load_questions(question_file: str) -> List[dict]:
    qs = []
    with open(question_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                qs.append(json.loads(line))
    return qs

def save_answers_jsonl(path: str, records: List[dict]):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def build_prompt_from_question(q: dict) -> str:
    if "turns" in q and isinstance(q["turns"], list):
        return q["turns"][0]
    if "text" in q:
        return q["text"]
    if "question" in q:
        return q["question"]
    return json.dumps(q, ensure_ascii=False)


# -----------------------------
# Sampling
# -----------------------------
def sample_from_logits(logits: torch.Tensor, temperature=1.0, top_p=1.0, top_k=0) -> int:
    if temperature <= 0:
        return int(torch.argmax(logits).item())
    logits = logits / max(float(temperature), 1e-6)

    if top_k and top_k > 0:
        _, idx = torch.topk(logits, k=top_k)
        mask = torch.full_like(logits, float("-inf"))
        mask[idx] = logits[idx]
        logits = mask

    probs = F.softmax(logits, dim=-1)

    if top_p < 1.0:
        sorted_probs, sorted_idx = torch.sort(probs, descending=True)
        cum = torch.cumsum(sorted_probs, dim=-1)
        keep = cum <= top_p
        keep[..., 0] = True
        filtered_idx = sorted_idx[keep]
        filtered_probs = probs[filtered_idx]
        filtered_probs = filtered_probs / filtered_probs.sum()
        return int(filtered_idx[torch.multinomial(filtered_probs, 1)].item())

    return int(torch.multinomial(probs, 1).item())


# -----------------------------
# Output field adapters (EDIT HERE if needed)
# -----------------------------
def get_head1_logits(out) -> torch.Tensor:
    # expect (B,V)
    if hasattr(out, "draft_logits") and out.draft_logits is not None:
        return out.draft_logits
    if hasattr(out, "head1_logits") and out.head1_logits is not None:
        return out.head1_logits
    raise RuntimeError("Cannot find head1 logits in output (draft_logits/head1_logits).")

def get_head2_logits_phase(out, phase: int) -> torch.Tensor:
    """
    For phase2/phase3 logits.
    Your model may store them separately.
    Common patterns:
      - out.head2_logits is already "current phase" logits
      - out.phase2_logits / out.phase3_logits
    """
    # easiest: out.head2_logits is "last phase logits"
    if hasattr(out, "head2_logits") and out.head2_logits is not None:
        return out.head2_logits

    # optional explicit names
    name = f"phase{phase}_logits"
    if hasattr(out, name) and getattr(out, name) is not None:
        return getattr(out, name)

    if hasattr(out, "fusion_logits") and out.fusion_logits is not None:
        return out.fusion_logits

    raise RuntimeError("Cannot find head2 logits in output (head2_logits/phase*_logits/fusion_logits).")


# -----------------------------
# Layerwise proposal: 3 forwards only
# -----------------------------
@torch.no_grad()
def propose_layerwise_tree(
    model: DualDecoderModel,
    input_ids: torch.Tensor,         # (1,T)
    attention_mask: torch.Tensor,    # (1,T)
    branch: int = 3,
) -> Tuple[List[int], List[int], List[int], List[Tuple[int,int,int]]]:
    """
    Returns:
      C1 (head1 top3), C2 (head2 phase2 top3), C3 (head2 phase3 top3),
      paths = cartesian product C1xC2xC3 (27)
    """
    device = input_ids.device

    # --- C1 from head1 ---
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        out1 = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            num_phases=1,
            temperature=1.0,
            train_stage="head2",  # doesn't matter, but keep consistent
        )
    logits1 = get_head1_logits(out1)[0].float()
    C1 = torch.topk(logits1, k=branch).indices.tolist()

    # --- C2 from head2 phase2 ---
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        out2 = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            num_phases=2,
            temperature=1.0,
            train_stage="head2",
        )
    logits2 = get_head2_logits_phase(out2, phase=2)[0].float()
    C2 = torch.topk(logits2, k=branch).indices.tolist()

    # --- C3 from head2 phase3 ---
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        out3 = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            num_phases=3,
            temperature=1.0,
            train_stage="head2",
        )
    logits3 = get_head2_logits_phase(out3, phase=3)[0].float()
    C3 = torch.topk(logits3, k=branch).indices.tolist()

    # cartesian product => 27 leaves
    paths: List[Tuple[int,int,int]] = []
    for t1 in C1:
        for t2 in C2:
            for t3 in C3:
                paths.append((t1, t2, t3))

    assert len(paths) == branch**3
    return C1, C2, C3, paths


# -----------------------------
# Base verify 27 paths in ONE forward (len=3)
# -----------------------------
@torch.no_grad()
def base_verify_paths_len3(
    base_model,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    paths: List[Tuple[int,int,int]],
    temperature: float,
    top_p: float,
    top_k: int,
    mode: str = "greedy",
) -> Tuple[List[int], int, int]:
    """
    Returns:
      accepted_tokens: length 1..3
      base_forwards: 1
      accepted_from_draft: number of accepted draft tokens
    """
    device = input_ids.device
    T = input_ids.size(1)
    B = len(paths)

    t1 = torch.tensor([[p[0]] for p in paths], device=device, dtype=torch.long)
    t2 = torch.tensor([[p[1]] for p in paths], device=device, dtype=torch.long)
    t3 = torch.tensor([[p[2]] for p in paths], device=device, dtype=torch.long)

    ctx = input_ids.expand(B, -1)
    msk = attention_mask.expand(B, -1)

    full_ids = torch.cat([ctx, t1, t2, t3], dim=1)      # (B,T+3)
    full_msk = torch.cat([msk, torch.ones((B,3), device=device, dtype=msk.dtype)], dim=1)

    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        out = base_model(input_ids=full_ids, attention_mask=full_msk, return_dict=True)
    logits = out.logits.float()  # (B,T+3,V)

    # step1 predicted at pos T-1, step2 at pos T, step3 at pos T+1
    step1_logits = logits[:, T-1, :]
    step2_logits = logits[:, T, :]
    step3_logits = logits[:, T+1, :]

    if mode == "greedy":
        base1 = torch.argmax(step1_logits, dim=-1)
        base2 = torch.argmax(step2_logits, dim=-1)
        base3 = torch.argmax(step3_logits, dim=-1)
    else:
        base1 = torch.tensor([sample_from_logits(step1_logits[i], temperature, top_p, top_k) for i in range(B)],
                             device=device, dtype=torch.long)
        base2 = torch.tensor([sample_from_logits(step2_logits[i], temperature, top_p, top_k) for i in range(B)],
                             device=device, dtype=torch.long)
        base3 = torch.tensor([sample_from_logits(step3_logits[i], temperature, top_p, top_k) for i in range(B)],
                             device=device, dtype=torch.long)

    t1_ids, t2_ids, t3_ids = t1.squeeze(1), t2.squeeze(1), t3.squeeze(1)

    ok1 = (base1 == t1_ids)
    ok2 = ok1 & (base2 == t2_ids)
    ok3 = ok2 & (base3 == t3_ids)

    # tie-break by base logprob of accepted prefix
    logp1 = F.log_softmax(step1_logits, dim=-1).gather(-1, t1_ids.unsqueeze(-1)).squeeze(-1)
    logp2 = F.log_softmax(step2_logits, dim=-1).gather(-1, t2_ids.unsqueeze(-1)).squeeze(-1)
    logp3 = F.log_softmax(step3_logits, dim=-1).gather(-1, t3_ids.unsqueeze(-1)).squeeze(-1)

    best_len, best_score, best_idx = -1, -1e30, 0
    for i in range(B):
        if ok3[i]:
            cand_len = 3
            cand_score = (logp1[i] + logp2[i] + logp3[i]).item()
        elif ok2[i]:
            cand_len = 2
            cand_score = (logp1[i] + logp2[i]).item()
        elif ok1[i]:
            cand_len = 1
            cand_score = logp1[i].item()
        else:
            cand_len = 0
            cand_score = -1e30

        if cand_len > best_len or (cand_len == best_len and cand_score > best_score):
            best_len, best_score, best_idx = cand_len, cand_score, i

    if best_len == 3:
        return [int(t1_ids[best_idx]), int(t2_ids[best_idx]), int(t3_ids[best_idx])], 1, 3
    if best_len == 2:
        return [int(t1_ids[best_idx]), int(t2_ids[best_idx])], 1, 2
    if best_len == 1:
        return [int(t1_ids[best_idx])], 1, 1

    # all failed at step1 -> fallback to base token (row0)
    return [int(base1[0].item())], 1, 0


# -----------------------------
# Metrics
# -----------------------------
@dataclass
class SpecMetrics:
    total_base_forwards: int = 0
    total_draft_forwards: int = 0
    total_generated: int = 0
    total_accepted_from_draft: int = 0
    wall_time_sec: float = 0.0

    def report(self) -> Dict[str, float]:
        vtpbf = self.total_generated / max(self.total_base_forwards, 1)
        accept_rate = self.total_accepted_from_draft / max(self.total_generated, 1)
        tps = self.total_generated / max(self.wall_time_sec, 1e-9)
        return {
            "accept_rate": accept_rate,
            "verified_tokens_per_base_forward": vtpbf,
            "total_base_forwards": float(self.total_base_forwards),
            "total_draft_forwards": float(self.total_draft_forwards),
            "total_generated_tokens": float(self.total_generated),
            "tokens_per_sec": tps,
        }


# -----------------------------
# Baseline generation
# -----------------------------
@torch.no_grad()
def generate_baseline(base_model, tokenizer, prompt, max_new_tokens, temperature, top_p, top_k, device):
    t0 = time.time()
    ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    mask = torch.ones_like(ids)
    out_tokens = []
    m = SpecMetrics()

    for _ in range(max_new_tokens):
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            out = base_model(input_ids=ids, attention_mask=mask, return_dict=True)
        m.total_base_forwards += 1
        step_logits = out.logits[0, -1].float()
        tok = sample_from_logits(step_logits, temperature, top_p, top_k)
        out_tokens.append(tok)

        ids = torch.cat([ids, torch.tensor([[tok]], device=device, dtype=torch.long)], dim=1)
        mask = torch.cat([mask, torch.ones((1,1), device=device, dtype=mask.dtype)], dim=1)

        if tokenizer.eos_token_id is not None and tok == tokenizer.eos_token_id:
            break

    m.total_generated = len(out_tokens)
    m.wall_time_sec = time.time() - t0
    return tokenizer.decode(out_tokens, skip_special_tokens=True), m


# -----------------------------
# Tree speculative generation (3 draft forwards per round)
# -----------------------------
@torch.no_grad()
def generate_tree40_layerwise(
    model: DualDecoderModel,
    tokenizer,
    prompt: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int,
    base_mode: str,
    device: str,
    branch: int = 3,
):
    t0 = time.time()
    ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    mask = torch.ones_like(ids)

    out_tokens: List[int] = []
    m = SpecMetrics()

    while len(out_tokens) < max_new_tokens:
        # 3 forward proposals only
        C1, C2, C3, paths = propose_layerwise_tree(model, ids, mask, branch=branch)
        m.total_draft_forwards += 3  # head1 + head2(phase2) + head2(phase3)

        accepted, base_fwds, acc_from_draft = base_verify_paths_len3(
            base_model=model.base_model,
            input_ids=ids,
            attention_mask=mask,
            paths=paths,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            mode=base_mode,
        )
        m.total_base_forwards += base_fwds
        m.total_accepted_from_draft += acc_from_draft

        out_tokens.extend(accepted)
        m.total_generated = len(out_tokens)

        # append to context
        new_ids = torch.tensor([accepted], device=device, dtype=torch.long)
        ids = torch.cat([ids, new_ids], dim=1)
        mask = torch.cat([mask, torch.ones((1, new_ids.size(1)), device=device, dtype=mask.dtype)], dim=1)

        if tokenizer.eos_token_id is not None and out_tokens and out_tokens[-1] == tokenizer.eos_token_id:
            break

    m.wall_time_sec = time.time() - t0
    return tokenizer.decode(out_tokens, skip_special_tokens=True), m


# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--question_file", type=str, required=True)
    ap.add_argument("--out_dir", type=str, default="tree40_layerwise_out")
    ap.add_argument("--base_model", type=str, default="Qwen/Qwen2.5-3B")
    ap.add_argument("--ckpt_head1", type=str, required=True)
    ap.add_argument("--ckpt_head2", type=str, required=True)

    ap.add_argument("--num_samples", type=int, default=50)
    ap.add_argument("--max_new_tokens", type=int, default=128)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top_p", type=float, default=1.0)
    ap.add_argument("--top_k", type=int, default=0)
    ap.add_argument("--base_mode", type=str, default="greedy", choices=["greedy", "sample"])
    ap.add_argument("--branch", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("[device]", device)

    config = DualDecoderConfig(
        base_model_name_or_path=args.base_model,
        num_draft_candidates=10,
        max_speculative_steps=2,
        draft_hidden_size=1024,
        draft_num_layers=4,
        fusion_hidden_size=1024,
        fusion_num_heads=4,
        fusion_dropout=0.0,
        decoder_num_layers=2,
        decoder_num_heads=8,
        decoder_dropout=0.0,
        head2_teacher_topk=10,
        head2_soft_ce_weight=1.0,
        head2_kl_weight=0.3,
    )

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = DualDecoderModel(config).to(device).eval()
    model = model.to(dtype=torch.bfloat16)  # IMPORTANT

    # load checkpoints
    stA = torch.load(args.ckpt_head1, map_location="cpu")
    if "head1_state_dict" in stA:
        model.head1.load_state_dict(stA["head1_state_dict"], strict=False)
    elif "model_state_dict" in stA:
        model.load_state_dict(stA["model_state_dict"], strict=False)
    else:
        model.load_state_dict(stA, strict=False)

    stB = torch.load(args.ckpt_head2, map_location="cpu")
    if "fusion_state_dict" in stB and stB["fusion_state_dict"] is not None:
        model.fusion.load_state_dict(stB["fusion_state_dict"], strict=False)
    if "head2_state_dict" in stB and stB["head2_state_dict"] is not None:
        model.head2.load_state_dict(stB["head2_state_dict"], strict=False)
    if "model_state_dict" in stB:
        model.load_state_dict(stB["model_state_dict"], strict=False)

    questions = load_questions(args.question_file)[: args.num_samples]

    base_records, tree_records = [], []
    base_all, tree_all = SpecMetrics(), SpecMetrics()

    for i, q in enumerate(questions):
        qid = q.get("question_id", i)
        prompt = build_prompt_from_question(q)

        base_text, m_base = generate_baseline(
            base_model=model.base_model,
            tokenizer=tokenizer,
            prompt=prompt,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            device=device,
        )
        tree_text, m_tree = generate_tree40_layerwise(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            base_mode=args.base_mode,
            device=device,
            branch=args.branch,
        )

        base_records.append({
            "question_id": qid,
            "model_id": "base_only",
            "answer_id": f"{qid}_base",
            "choices": [{"index": 0, "turns": [base_text]}],
        })
        tree_records.append({
            "question_id": qid,
            "model_id": "tree40_layerwise_b3",
            "answer_id": f"{qid}_tree40",
            "choices": [{"index": 0, "turns": [tree_text]}],
        })

        for k, v in m_base.__dict__.items():
            setattr(base_all, k, getattr(base_all, k) + v)
        for k, v in m_tree.__dict__.items():
            setattr(tree_all, k, getattr(tree_all, k) + v)

        if (i + 1) % 5 == 0:
            print(f"[{i+1}/{len(questions)}] "
                  f"base tps={m_base.report()['tokens_per_sec']:.2f} | "
                  f"tree40 tps={m_tree.report()['tokens_per_sec']:.2f} "
                  f"accept={m_tree.report()['accept_rate']:.3f} vtpbf={m_tree.report()['verified_tokens_per_base_forward']:.3f}")

    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)
    base_path = os.path.join(out_dir, "answers_base_only.jsonl")
    tree_path = os.path.join(out_dir, "answers_tree40_layerwise.jsonl")
    save_answers_jsonl(base_path, base_records)
    save_answers_jsonl(tree_path, tree_records)

    base_rep = base_all.report()
    tree_rep = tree_all.report()

    base_fwd_per_tok = base_rep["total_base_forwards"] / max(base_rep["total_generated_tokens"], 1.0)
    tree_fwd_per_tok = tree_rep["total_base_forwards"] / max(tree_rep["total_generated_tokens"], 1.0)
    theoretical_speedup = base_fwd_per_tok / max(tree_fwd_per_tok, 1e-9)
    real_speedup = tree_rep["tokens_per_sec"] / max(base_rep["tokens_per_sec"], 1e-9)

    print("\n========================")
    print("[Baseline metrics]")
    print(json.dumps(base_rep, indent=2))
    print("[Tree40 layerwise speculative metrics]")
    print(json.dumps(tree_rep, indent=2))
    print(f"[Speedup] theoretical_by_base_forwards = {theoretical_speedup:.3f}")
    print(f"[Speedup] real_by_tokens_per_sec        = {real_speedup:.3f}")
    print(f"[Outputs] {base_path}")
    print(f"[Outputs] {tree_path}")
    print("========================\n")


if __name__ == "__main__":
    main()
