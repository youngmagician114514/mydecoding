#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Linear (no-tree) speculative decoding eval for ECNU-EDD / Faster-SD style Effective_Draft_Decoder.

What "linear" means here:
- Draft produces ONE chain of K tokens each round (no branching tree).
- Target verifies that chain in ONE forward pass (standard speculative decoding).

Important alignment choice (matches the seed-first EDD decoding you discussed):
- We maintain:
    ctx_ids   : accepted prefix tokens (already committed)
    ctx_hidden: target hidden states (enc_layer_index) for ctx_ids
    seed_id   : target greedy next token AFTER ctx_ids
- Each round:
    1) Draft proposes K tokens AFTER seed, conditioned on (ctx_hidden + emb(seed))
    2) Target verifies on [ctx_ids, seed, props] in one forward
    3) Accept the longest prefix of props that match target greedy
    4) Commit: append (seed + accepted props) into ctx_ids
    5) Next seed := target greedy next token after the last committed token
       (taken from the same verification logits)

Metrics printed:
- acceptance rate α  = accepted_draft_tokens / proposed_draft_tokens
- average acceptance length τ = average number of NEW tokens committed per round
  (here τ = avg(1 + accepted_draft_tokens) because seed is always committed)

This script can evaluate ONE MT-Bench question (by --question_id) or the first one (default).
"""

import argparse, json, os, sys, random
from typing import Dict, Optional, List, Tuple

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


_THIS = os.path.abspath(__file__)
_ROOT = os.path.dirname(_THIS)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

try:
    from models.edd_draft_model import Effective_Draft_Decoder  # type: ignore
except Exception:
    # fallback for different project layout
    from mydecoding.models.edd_draft_model import Effective_Draft_Decoder  # type: ignore


def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_one_mtbench_question(path: str, question_id: Optional[int]) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        items = [json.loads(line) for line in f if line.strip()]
    if question_id is None:
        return items[0]
    for it in items:
        if int(it.get("question_id", -1)) == int(question_id):
            return it
    raise ValueError(f"question_id={question_id} not found in {path}")


def build_prompt_text(tokenizer, question: Dict, turn: int = 0, use_chat_template: bool = True) -> str:
    user_text = question["turns"][turn]
    msgs = [{"role": "user", "content": user_text}]
    if use_chat_template and hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    return f"USER: {user_text}\nASSISTANT:"


@torch.no_grad()
def target_forward_hidden_logits(target, input_ids, attention_mask, enc_layer_index: int):
    out = target(
        input_ids=input_ids,
        attention_mask=attention_mask,
        output_hidden_states=True,
        use_cache=False,
        return_dict=True,
    )
    return out.hidden_states[enc_layer_index], out.logits


@torch.no_grad()
def greedy_id(logits_1v: torch.Tensor) -> int:
    return int(torch.argmax(logits_1v, dim=-1).item())


@torch.no_grad()
def draft_next_token(draft: Effective_Draft_Decoder,
                     ctx_hidden: torch.Tensor,     # (1, T, H)
                     tokens: List[int]) -> int:
    """
    tokens: current token-stream tokens that are ALREADY in the "decoder stream"
            for EDD, we use [seed, d1, d2, ...] (length M>=1).
    returns: next token id (greedy) after the last token in `tokens`.
    """
    device = ctx_hidden.device
    B, T, H = ctx_hidden.shape
    M = len(tokens)
    assert B == 1, "this eval assumes batch_size=1"

    tok = torch.tensor([tokens], device=device, dtype=torch.long)  # (1,M)

    # embedding layer name differs across implementations
    if hasattr(draft, "embedding_layer"):
        tok_emb = draft.embedding_layer(tok)
    else:
        tok_emb = draft.get_input_embeddings()(tok)

    hidden_states = torch.cat([ctx_hidden, tok_emb], dim=1)  # (1, T+M, H)

    # positions: enc part [0..T-1], token stream [T..T+M-1]
    pos = torch.arange(T + M, device=device).unsqueeze(0)  # (1, T+M)

    # Dual-block attention mask from the paper implementation
    attn = draft._build_dual_block_mask(
        enc_len=T,
        total_len=T + M,
        inp_len=T,
        pred_len=M,
        device=device,
        dtype=hidden_states.dtype,
    )
    if attn.size(0) != B:
        attn = attn.expand(B, -1, -1, -1)

    # RoPE / position embeddings helper (varies by transformers version)
    position_embeddings = draft._compute_position_embeddings(hidden_states, pos)

    x = hidden_states
    for layer in draft.decoder_layers:
        try:
            out = layer(x, attention_mask=attn, position_ids=pos, position_embeddings=position_embeddings)
        except TypeError:
            out = layer(x, attention_mask=attn, position_ids=pos)
        x = out[0] if isinstance(out, (tuple, list)) else out

    x = draft.norm(x)
    logits = draft.lm_head(x)  # (1, T+M, V)
    last_logits = logits[:, -1, :]  # next-token distribution after last token-stream token
    return greedy_id(last_logits)


@torch.no_grad()
def propose_k_after_seed(draft: Effective_Draft_Decoder,
                         ctx_hidden: torch.Tensor,
                         seed_id: int,
                         K: int) -> List[int]:
    tokens = [seed_id]
    props: List[int] = []
    for _ in range(K):
        nxt = draft_next_token(draft, ctx_hidden, tokens)
        props.append(nxt)
        tokens.append(nxt)
    return props


@torch.no_grad()
def verify_linear(target, ctx_ids, seed_id, props, enc_layer_index: int) -> Tuple[int, int, torch.Tensor]:
    """
    Verify in one forward:
      full_ids = [ctx_ids, seed, props]

    Returns:
      accept_m  : number of accepted proposal tokens (0..K)
      next_seed : target greedy next token after the last committed token
                 (computed from logits at index = ctx_len + accept_m)
      h_all     : hidden_states[enc_layer_index] for full_ids
    """
    device = ctx_ids.device
    ctx_len = ctx_ids.size(1)
    K = len(props)

    seed_tensor = torch.tensor([[seed_id]], device=device, dtype=ctx_ids.dtype)
    prop_tensor = torch.tensor([props], device=device, dtype=ctx_ids.dtype) if K > 0 else None
    if prop_tensor is None:
        full_ids = torch.cat([ctx_ids, seed_tensor], dim=1)
    else:
        full_ids = torch.cat([ctx_ids, seed_tensor, prop_tensor], dim=1)

    full_mask = torch.ones_like(full_ids, device=device)

    h_all, logits_all = target_forward_hidden_logits(target, full_ids, full_mask, enc_layer_index)

    # Check proposals left-to-right: proposal j should match target greedy next after prefix ending at:
    # index (ctx_len + j - 1) where j starts at 1 and ctx_len points to seed position - 1? (seed is at ctx_len)
    accept_m = 0
    for j in range(1, K + 1):
        pred = greedy_id(logits_all[:, ctx_len + (j - 1), :])   # target next after seed / after accepted proposals
        gold = int(full_ids[0, ctx_len + j].item())             # proposal id at that position
        if pred == gold:
            accept_m += 1
        else:
            break

    # Next seed = target greedy next token after last committed token:
    # - if accept_m == 0: after seed -> logits index ctx_len
    # - if accept_m == k: after proposal k -> logits index ctx_len + k
    next_seed = greedy_id(logits_all[:, ctx_len + accept_m, :])

    return accept_m, next_seed, h_all


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target_model", required=True)
    ap.add_argument("--edd_dir", required=True)
    ap.add_argument("--draft_ckpt", default=None, help="defaults to {edd_dir}/draft_decoder_final.pt")
    ap.add_argument("--mtbench_questions", required=True)
    ap.add_argument("--question_id", type=int, default=None)
    ap.add_argument("--use_chat_template", action="store_true")
    ap.add_argument("--enc_layer_index", type=int, default=-4)
    ap.add_argument("--num_layers", type=int, default=1, help="must match training --num_layers")
    ap.add_argument("--K", type=int, default=5)
    ap.add_argument("--max_new_tokens", type=int, default=128)
    ap.add_argument("--bf16", action="store_true")
    ap.add_argument("--fp16", action="store_true")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--print_rounds", type=int, default=8)
    args = ap.parse_args()

    if args.bf16 and args.fp16:
        raise ValueError("choose at most one of --bf16/--fp16")
    dtype = torch.bfloat16 if args.bf16 else (torch.float16 if args.fp16 else torch.float32)
    device = torch.device(args.device)
    set_seed(args.seed)

    ckpt = args.draft_ckpt or os.path.join(args.edd_dir, "draft_decoder_final.pt")
    assert os.path.exists(ckpt), f"not found: {ckpt}"

    # Tokenizer from edd_dir to match training chat template / special tokens
    tok = AutoTokenizer.from_pretrained(args.edd_dir, trust_remote_code=True, use_fast=True)
    if tok.pad_token is None and tok.eos_token is not None:
        tok.pad_token = tok.eos_token

    q = load_one_mtbench_question(args.mtbench_questions, args.question_id)
    prompt_text = build_prompt_text(tok, q, 0, use_chat_template=args.use_chat_template)
    enc = tok(prompt_text, return_tensors="pt")
    ctx_ids = enc["input_ids"].to(device)
    ctx_mask = enc.get("attention_mask", torch.ones_like(ctx_ids)).to(device)

    # Target model (frozen)
    target = AutoModelForCausalLM.from_pretrained(
        args.target_model, torch_dtype=dtype, trust_remote_code=True
    ).to(device).eval()
    for p in target.parameters():
        p.requires_grad_(False)

    # Draft decoder initialized from teacher (copies emb + lm_head, etc.)
    draft = Effective_Draft_Decoder.from_teacher(target, num_layers=args.num_layers)
    draft.to(device).eval()

    # load EDD checkpoint (.pt)
    sd = torch.load(ckpt, map_location="cpu")
    if isinstance(sd, dict) and any(k.startswith("module.") for k in sd.keys()):
        sd = {k.replace("module.", "", 1): v for k, v in sd.items()}
    draft.load_state_dict(sd, strict=True)

    # ensure dtype alignment (fixes bf16/float mismatch errors)
    draft = draft.to(device=device, dtype=dtype).eval()

    # Initial encoder hidden and initial seed (target greedy next after prompt)
    ctx_hidden, ctx_logits = target_forward_hidden_logits(target, ctx_ids, ctx_mask, args.enc_layer_index)
    seed_id = greedy_id(ctx_logits[:, -1, :])

    total_proposed = 0
    total_accepted = 0
    rounds = 0
    total_committed_new = 0  # counts how many new tokens we appended to ctx_ids

    print("=" * 80)
    print(f"MT-Bench question_id={q.get('question_id')}  category={q.get('category')}")
    print("- Prompt -")
    print(prompt_text)
    print("=" * 80)

    while total_committed_new < args.max_new_tokens:
        rounds += 1

        # 1) Draft proposes K tokens AFTER seed
        props = propose_k_after_seed(draft, ctx_hidden, seed_id, args.K)
        total_proposed += len(props)

        # 2) Target verifies in one forward on [ctx, seed, props]
        accept_m, next_seed, h_all = verify_linear(
            target, ctx_ids, seed_id, props, args.enc_layer_index
        )
        total_accepted += accept_m

        # 3) Commit seed + accepted proposals into ctx_ids (these tokens are now part of the accepted prefix)
        ctx_len = ctx_ids.size(1)
        new_ctx_len = ctx_len + 1 + accept_m  # seed + accept_m proposals
        # Rebuild full_ids to slice (cheap)
        seed_tensor = torch.tensor([[seed_id]], device=device, dtype=ctx_ids.dtype)
        prop_tensor = torch.tensor([props], device=device, dtype=ctx_ids.dtype)
        full_ids = torch.cat([ctx_ids, seed_tensor, prop_tensor], dim=1)

        ctx_ids = full_ids[:, :new_ctx_len]
        ctx_hidden = h_all[:, :new_ctx_len, :].detach()

        # 4) Next round seed comes from target greedy after last committed token
        seed_id = next_seed

        # Metrics / stopping
        committed_now = 1 + accept_m
        total_committed_new += committed_now

        if rounds <= args.print_rounds:
            seed_txt = tok.decode([seed_id], skip_special_tokens=False)
            prop_txt = tok.decode(props, skip_special_tokens=False)
            print(f"[round {rounds:03d}] accept={accept_m}/{args.K}  committed={committed_now}  next_seed={seed_txt!r}  props={prop_txt!r}")

    alpha = total_accepted / max(1, total_proposed)
    tau = total_committed_new / max(1, rounds)

    print(f"Rounds={rounds} proposed={total_proposed} accepted={total_accepted}")
    print(f"alpha(accept_rate)={alpha:.4f}   tau(avg_accept_len)={tau:.4f}")


if __name__ == "__main__":
    main()
