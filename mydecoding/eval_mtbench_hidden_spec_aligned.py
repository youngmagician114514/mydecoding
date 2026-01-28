#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import os
import random
from typing import Dict, Optional, Tuple, List

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

NEG_INF = -1e9


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_one_mtbench_question(path: str, question_id: Optional[int] = None) -> Dict:
    assert os.path.exists(path), f"Not found: {path}"
    with open(path, "r", encoding="utf-8") as f:
        items = [json.loads(line) for line in f if line.strip()]
    if not items:
        raise RuntimeError(f"Empty mtbench file: {path}")

    if question_id is None:
        return items[0]

    for it in items:
        if int(it.get("question_id", -1)) == int(question_id):
            return it
    raise ValueError(f"question_id={question_id} not found in {path}")


def build_prompt_text(tokenizer, question: Dict, use_turn: int = 0) -> str:
    turns = question.get("turns", [])
    if not turns:
        raise ValueError("Question has no turns.")
    user_text = turns[use_turn]
    messages = [{"role": "user", "content": user_text}]

    if hasattr(tokenizer, "apply_chat_template"):
        try:
            return tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        except Exception:
            pass

    return f"USER: {user_text}\nASSISTANT:"


@torch.no_grad()
def target_hidden_logits(
    target,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    enc_layer_index: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    out = target(
        input_ids=input_ids,
        attention_mask=attention_mask,
        output_hidden_states=True,
        use_cache=False,
    )
    return out.hidden_states[enc_layer_index], out.logits


def prefix_plus_causal_attn_bias(prefix_len: int, total_len: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    # 对 bf16/fp16 推荐用一个足够小的负数（-1e4）避免极值带来不稳定
    neg = torch.tensor(-1e4, device=device, dtype=dtype)
    bias = torch.full((1, 1, total_len, total_len), neg, dtype=dtype, device=device)

    # prefix causal
    for i in range(prefix_len):
        bias[:, :, i, : i + 1] = 0

    # token stream queries: attend prefix + causal inside stream
    for i in range(prefix_len, total_len):
        bias[:, :, i, :prefix_len] = 0
        bias[:, :, i, prefix_len : i + 1] = 0

    return bias


@torch.no_grad()
def draft_propose_k(
    draft,
    ctx_hidden: torch.Tensor,      # (1, Tc, H)  hidden(ctx_ids), NOT trimmed
    seed_id: torch.Tensor,         # (1, 1)
    K: int,
) -> torch.Tensor:
    """
    inputs_embeds = [ctx_hidden ; emb(seed) ; emb(generated...)]
    generate K tokens by argmax
    """
    device = seed_id.device
    Tc = ctx_hidden.shape[1]
    tok_stream = seed_id.clone()  # (1,1)

    for _ in range(K):
        tok_emb = draft.get_input_embeddings()(tok_stream)              # (1,S,H)
        inputs_embeds = torch.cat([ctx_hidden, tok_emb], dim=1)         # (1,Tc+S,H)
        attn_bias = prefix_plus_causal_attn_bias(Tc, inputs_embeds.shape[1], device, dtype=inputs_embeds.dtype,)

        out = draft.model(inputs_embeds=inputs_embeds, attention_mask=attn_bias, use_cache=False)
        h_last = out.last_hidden_state[:, -1, :]                        # (1,H)
        logits = draft.lm_head(h_last)                                  # (1,V)
        next_id = torch.argmax(logits, dim=-1, keepdim=True)            # (1,1)
        tok_stream = torch.cat([tok_stream, next_id], dim=1)

    return tok_stream[:, 1:]  # (1,K)

def show_token_list(tokenizer, ids: List[int]) -> str:
    pieces = tokenizer.convert_ids_to_tokens(ids)
    # piece级别（带▁） + 每个id单独decode（看空格/换行更直观）
    singles = [repr(tokenizer.decode([i], skip_special_tokens=False)) for i in ids]
    return " | ".join([f"{i}:{p}:{s}" for i, p, s in zip(ids, pieces, singles)])



@torch.no_grad()
def verify_once_and_slice_hidden(
    target,
    ctx_ids: torch.Tensor,          # (1, Tc)
    seed_id: torch.Tensor,          # (1, 1)
    proposal: torch.Tensor,         # (1, K)
    enc_layer_index: int,
) -> Tuple[int, int, torch.Tensor]:
    """
    Run ONE forward on [ctx, seed, proposal].
    Verify proposal tokens against target greedy.

    Returns:
      accept_len: number of proposal tokens accepted from the start
      correction_token: target greedy next token at the first proposal position (used if accept_len==0)
      enc_hidden_full: hidden_states[enc_layer_index] for the full sequence (1, Tc+1+K, H)
    """
    device = ctx_ids.device
    ids = torch.cat([ctx_ids, seed_id, proposal], dim=1)   # (1, Tc+1+K)
    attn = torch.ones_like(ids, device=device)

    enc_hidden_full, logits_full = target_hidden_logits(target, ids, attn, enc_layer_index)

    Tc = ctx_ids.shape[1]
    K = proposal.shape[1]

    # proposal[0] should equal target greedy next at position = Tc (seed position)
    correction = int(torch.argmax(logits_full[:, Tc, :], dim=-1).item())

    accept = 0
    for i in range(K):
        pos = Tc + i  # logits at pos predicts token at pos+1, which is proposal[i]
        greedy_next = int(torch.argmax(logits_full[:, pos, :], dim=-1).item())
        if greedy_next == int(proposal[:, i].item()):
            accept += 1
        else:
            break

    return accept, correction, enc_hidden_full


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target_model", type=str, required=True)
    ap.add_argument("--edd_dir", type=str, required=True)

    ap.add_argument("--mtbench_questions", type=str, required=True)
    ap.add_argument("--question_id", type=int, default=None)
    ap.add_argument("--use_turn", type=int, default=0)

    ap.add_argument("--bf16", action="store_true")
    ap.add_argument("--fp16", action="store_true")

    ap.add_argument("--enc_layer_index", type=int, default=-4)
    ap.add_argument("--K", type=int, default=5)
    ap.add_argument("--max_new_tokens", type=int, default=128)
    ap.add_argument("--max_rounds", type=int, default=10_000)
    ap.add_argument("--seed", type=int, default=42)

    args = ap.parse_args()
    if args.bf16 and args.fp16:
        raise ValueError("Choose only one of --bf16 / --fp16")
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if args.bf16 else (torch.float16 if args.fp16 else torch.float32)

    q = load_one_mtbench_question(args.mtbench_questions, args.question_id)

    tokenizer = AutoTokenizer.from_pretrained(args.target_model, use_fast=True, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    prompt_text = build_prompt_text(tokenizer, q, args.use_turn)
    toks = tokenizer(prompt_text, return_tensors="pt", add_special_tokens=False)
    prompt_ids = toks["input_ids"].to(device)          # (1, Tprompt)
    prompt_attn = toks["attention_mask"].to(device)

    target = AutoModelForCausalLM.from_pretrained(args.target_model, torch_dtype=dtype, trust_remote_code=True).to(device)
    target.eval()
    for p in target.parameters():
        p.requires_grad_(False)

    draft = AutoModelForCausalLM.from_pretrained(args.edd_dir, torch_dtype=dtype, trust_remote_code=True).to(device)
    draft.eval()

    print("=" * 80)
    print(f"MT-Bench question_id={q.get('question_id')}  category={q.get('category')}")
    print("- Prompt -")
    print(prompt_text)
    print("=" * 80)

    # -------------------------
    # State invariant:
    #   ctx_ids = committed prefix EXCLUDING seed token
    #   seed_id = last committed token (used as embedding input to draft)
    # draft input each round:
    #   ctx_hidden = hidden(ctx_ids)  (NOT trimmed)
    #   seed embedding = emb(seed_id)
    # -------------------------

    # Init:
    # ctx = prompt, seed = target greedy next(prompt)
    with torch.no_grad():
        ctx_hidden, logits = target_hidden_logits(target, prompt_ids, prompt_attn, args.enc_layer_index)
        seed_id = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)  # (1,1)

    ctx_ids = prompt_ids  # (1,Tc)
    generated: List[int] = [int(seed_id.item())]  # output tokens after prompt (seed is first token)

    total_proposed = 0
    total_accepted = 0

    rounds = 0
    while len(generated) < args.max_new_tokens and rounds < args.max_rounds:
        rounds += 1

        # 1) draft propose
        proposal = draft_propose_k(draft, ctx_hidden, seed_id, args.K)  # (1,K)
        total_proposed += proposal.shape[1]

        # 2) target verify once, also get full hidden to slice for next ctx_hidden
        accept_len, correction, enc_hidden_full = verify_once_and_slice_hidden(
            target=target,
            ctx_ids=ctx_ids,
            seed_id=seed_id,
            proposal=proposal,
            enc_layer_index=args.enc_layer_index,
        )

        prop_ids = proposal[0].tolist()
        print(f"[Round {rounds}] seed_id={int(seed_id.item())} seed_tok={repr(tokenizer.decode([int(seed_id.item())], skip_special_tokens=False))}")
        print("  proposal_ids:", prop_ids)
        print("  proposal_tok:", show_token_list(tokenizer, prop_ids))
        print("  proposal_text:", repr(tokenizer.decode(prop_ids, skip_special_tokens=False)))

        Tc = ctx_ids.shape[1]

        if accept_len > 0:
            accepted = proposal[:, :accept_len]  # (1,m)
            generated.extend(accepted[0].tolist())
            total_accepted += accept_len

            # update state:
            # new_seed = last accepted token
            new_seed = accepted[:, -1:]  # (1,1)
            acc_ids = proposal[0, :accept_len].tolist()
            print("  accepted_tok:", show_token_list(tokenizer, acc_ids))
            # new_ctx = old_ctx + [old_seed] + accepted[:-1]
            if accept_len > 1:
                accepted_except_last = accepted[:, :-1]  # (1,m-1)
                ctx_ids = torch.cat([ctx_ids, seed_id, accepted_except_last], dim=1)
            else:
                ctx_ids = torch.cat([ctx_ids, seed_id], dim=1)

            # slice hidden for new_ctx from enc_hidden_full:
            # enc_hidden_full corresponds to [old_ctx, old_seed, proposal...]
            # new_ctx length = Tc + accept_len  (old_ctx + old_seed + (accept_len-1 proposals))
            new_ctx_len = Tc + accept_len
            ctx_hidden = enc_hidden_full[:, :new_ctx_len, :]

            seed_id = new_seed

        else:
            # no proposal accepted -> commit correction token (target greedy next after seed)
            generated.append(int(correction))

            # new_ctx = old_ctx + [old_seed]
            ctx_ids = torch.cat([ctx_ids, seed_id], dim=1)

            # hidden for new_ctx is first Tc+1 positions of enc_hidden_full
            ctx_hidden = enc_hidden_full[:, :Tc + 1, :]

            # new_seed = correction token
            seed_id = torch.tensor([[correction]], device=device, dtype=torch.long)
            print("  correction_id:", correction, " correction_tok:", repr(tokenizer.decode([correction], skip_special_tokens=False)))

        # EOS stop
        if tokenizer.eos_token_id is not None and generated and generated[-1] == tokenizer.eos_token_id:
            print("[stop] EOS.")
            break

        partial = tokenizer.decode(generated, skip_special_tokens=False)
        print("partial_output:", partial)
        print("-" * 80)

    final_text = tokenizer.decode(generated, skip_special_tokens=False)
    print("\n" + "=" * 80)
    print("Final output (tokens after prompt):")
    print(final_text)
    print("=" * 80)
    print(f"Rounds={rounds} total_proposed={total_proposed} total_accepted={total_accepted}")
    if total_proposed > 0:
        print(f"Accepted/proposed={total_accepted / total_proposed:.4f}")


if __name__ == "__main__":
    main()
