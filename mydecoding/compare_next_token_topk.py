import argparse
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM

NEG = -1e4  # safe additive bias for bf16/fp16


def prefix_plus_causal_attn_bias(prefix_len: int, total_len: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """
    4D additive attention bias for draft:
      - prefix positions: causal inside prefix
      - token-stream (after prefix): can attend all prefix + causal within stream
    dtype MUST match query dtype for SDPA.
    Shape: (1, 1, L, L)
    """
    L = total_len
    bias = torch.full((1, 1, L, L), NEG, device=device, dtype=dtype)

    # prefix causal
    for i in range(prefix_len):
        bias[:, :, i, : i + 1] = 0

    # stream causal + can attend prefix
    for i in range(prefix_len, L):
        bias[:, :, i, :prefix_len] = 0
        bias[:, :, i, prefix_len : i + 1] = 0

    return bias


def fmt_table(tokenizer, ids, probs):
    rows = []
    toks = tokenizer.convert_ids_to_tokens(ids)
    for i, (tid, piece, p) in enumerate(zip(ids, toks, probs), 1):
        single = tokenizer.decode([int(tid)], skip_special_tokens=False)
        disp = single.replace("\n", "\\n").replace("\t", "\\t")
        rows.append(f"{i:2d}. id={int(tid):6d} piece={piece:<12} dec={disp:<18}  p={p:.6f}")
    return "\n".join(rows)


@torch.no_grad()
def target_topk_next(model, input_ids, attention_mask, topk=10):
    """Top-k for next token (logits at last position)."""
    out = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        use_cache=False,
        return_dict=True,
    )
    logits = out.logits[:, -1, :]  # (1, V)
    probs = F.softmax(logits.float(), dim=-1)  # (1, V)
    topv, topi = torch.topk(probs, k=topk, dim=-1)  # (1, K)
    return topi[0].tolist(), topv[0].tolist()


@torch.no_grad()
def edd_draft_topk_first_token_seeded(
    target_model,
    draft_model,
    input_ids,
    attention_mask,
    enc_layer_index=-4,
    topk=10,
    seed_strategy="greedy",
):
    """
    Compare following your seed-first inference:

    Step A (target on prompt):
      - enc_hidden = hidden_states[enc_layer_index] over prompt tokens
      - seed = target predicted next token after prompt (greedy by default)

    Step B (draft on [enc_hidden(prompt), emb(seed)]):
      - run draft.model on inputs_embeds (prefix hidden + seed embedding)
      - take LAST position hidden -> lm_head -> distribution over the FIRST draft token d1

    Returns:
      seed_id (int)
      topk_ids, topk_probs for d1 distribution
    """
    # A1) target encode prompt and compute seed
    out_t = target_model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        output_hidden_states=True,
        use_cache=False,
        return_dict=True,
    )
    enc_hidden = out_t.hidden_states[enc_layer_index]  # (1, T, H)

    if seed_strategy != "greedy":
        raise ValueError("Only greedy seed supported in this script.")
    seed_id = int(torch.argmax(out_t.logits[:, -1, :], dim=-1).item())  # token after prompt

    # B1) build inputs_embeds = [enc_hidden(prompt), emb(seed)]
    seed_tensor = torch.tensor([[seed_id]], device=input_ids.device, dtype=torch.long)
    seed_emb = draft_model.get_input_embeddings()(seed_tensor)  # (1,1,H)
    inputs_embeds = torch.cat([enc_hidden, seed_emb], dim=1)    # (1, T+1, H)

    # B2) 4D additive bias mask, dtype aligned to inputs_embeds
    T = enc_hidden.shape[1]
    attn_bias = prefix_plus_causal_attn_bias(prefix_len=T, total_len=T + 1, device=input_ids.device, dtype=inputs_embeds.dtype)

    out_d = draft_model.model(
        inputs_embeds=inputs_embeds,
        attention_mask=attn_bias,
        use_cache=False,
        return_dict=True,
    )
    h_last = out_d.last_hidden_state[:, -1, :]  # hidden at seed position, used to predict next (d1)
    logits = draft_model.lm_head(h_last)        # (1, V)
    probs = F.softmax(logits.float(), dim=-1)
    topv, topi = torch.topk(probs, k=topk, dim=-1)

    return seed_id, topi[0].tolist(), topv[0].tolist()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target_model", required=True)
    ap.add_argument("--edd_dir", required=True)
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--enc_layer_index", type=int, default=-4)
    ap.add_argument("--topk", type=int, default=10)
    ap.add_argument("--bf16", action="store_true")
    ap.add_argument("--fp16", action="store_true")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    if args.bf16 and args.fp16:
        raise ValueError("Choose at most one of --bf16 / --fp16")

    if args.bf16:
        dtype = torch.bfloat16
    elif args.fp16:
        dtype = torch.float16
    else:
        dtype = torch.float32

    tokenizer = AutoTokenizer.from_pretrained(args.target_model, trust_remote_code=True)
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token

    target = AutoModelForCausalLM.from_pretrained(
        args.target_model,
        trust_remote_code=True,
        torch_dtype=dtype if args.device.startswith("cuda") else None,
    ).to(args.device).eval()

    draft = AutoModelForCausalLM.from_pretrained(
        args.edd_dir,
        trust_remote_code=True,
        torch_dtype=dtype if args.device.startswith("cuda") else None,
    ).to(args.device).eval()

    enc = tokenizer(args.prompt, return_tensors="pt", add_special_tokens=False)
    input_ids = enc["input_ids"].to(args.device)
    attention_mask = enc.get("attention_mask", torch.ones_like(input_ids)).to(args.device)

    # 1) seed from target(prompt)
    # Also show target topk for t_{i+1}
    tgt_next_ids, tgt_next_probs = target_topk_next(target, input_ids, attention_mask, topk=args.topk)
    seed_id = tgt_next_ids[0]  # greedy == top1

    # 2) compare after-seed: target([prompt, seed]) predicts t_{i+2}
    seed_tensor = torch.tensor([[int(seed_id)]], device=args.device, dtype=torch.long)
    ids2 = torch.cat([input_ids, seed_tensor], dim=1)
    mask2 = torch.cat([attention_mask, torch.ones_like(seed_tensor, dtype=attention_mask.dtype)], dim=1)
    tgt_after_seed_ids, tgt_after_seed_probs = target_topk_next(target, ids2, mask2, topk=args.topk)

    # 3) draft first token distribution d1 using seed-first inference
    seed_id2, dr_ids, dr_probs = edd_draft_topk_first_token_seeded(
        target_model=target,
        draft_model=draft,
        input_ids=input_ids,
        attention_mask=attention_mask,
        enc_layer_index=args.enc_layer_index,
        topk=args.topk,
        seed_strategy="greedy",
    )
    assert int(seed_id2) == int(seed_id), "Seed mismatch: target greedy and draft seed should match"

    print("=" * 80)
    print("PROMPT:")
    print(args.prompt)
    print("=" * 80)
    print(f"Seed (target greedy next after prompt): id={int(seed_id)} piece={tokenizer.convert_ids_to_tokens([int(seed_id)])[0]} dec={repr(tokenizer.decode([int(seed_id)], skip_special_tokens=False))}")
    print("-" * 80)
    print(f"[Target] top{args.topk} for t(i+1) (next after prompt)")
    print(fmt_table(tokenizer, tgt_next_ids, tgt_next_probs))
    print("-" * 80)
    print(f"[Target] top{args.topk} for t(i+2) (next after seed), i.e. target on [prompt, seed]")
    print(fmt_table(tokenizer, tgt_after_seed_ids, tgt_after_seed_probs))
    print("-" * 80)
    print(f"[EDD Draft] top{args.topk} for d1 (first draft token after seed) using enc_layer_index={args.enc_layer_index}")
    print(fmt_table(tokenizer, dr_ids, dr_probs))
    print("=" * 80)

    # overlap diagnostics
    tgt2_set = {int(i) for i in tgt_after_seed_ids}
    overlap = [int(i) for i in dr_ids if int(i) in tgt2_set]
    print(f"Overlap token-ids between Target(t(i+2)) top{args.topk} and Draft(d1) top{args.topk}: {overlap}")


if __name__ == "__main__":
    main()
