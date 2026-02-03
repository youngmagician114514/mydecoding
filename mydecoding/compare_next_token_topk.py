#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compare Target next-token distributions vs trained Effective_Draft_Decoder checkpoint (.pt).

Prints:
- Seed token = target greedy next after prompt
- Target topK for t(i+2) given [prompt, seed]
- Draft topK for d1 given (enc_hidden(prompt), emb(seed))

This is the quickest sanity-check for whether the trained draft is aligned.
"""

import argparse, os, sys, json
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM

# make sure we can import your local models package when running from mydecoding/training
_THIS = os.path.abspath(__file__)
_ROOT = os.path.dirname(os.path.dirname(_THIS))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

try:
    from models.edd_draft_model import Effective_Draft_Decoder, load_teacher  # type: ignore
except Exception:
    # fallback for other package layouts
    from mydecoding.models.edd_draft_model import Effective_Draft_Decoder, load_teacher  # type: ignore


@torch.no_grad()
def greedy_next_id(model, input_ids, attn_mask):
    out = model(input_ids=input_ids, attention_mask=attn_mask, use_cache=False, return_dict=True)
    return int(out.logits[0, -1].argmax().item())


@torch.no_grad()
def topk_from_logits(logits, k, tokenizer):
    probs = F.softmax(logits.float(), dim=-1)
    topv, topi = torch.topk(probs, k=k, dim=-1)
    ids = topi[0].tolist()
    return [(i, tokenizer.decode([i]), float(p)) for i, p in zip(ids, topv[0].tolist())]


def fmt(rows, title):
    print(title)
    for rank, (tid, piece, p) in enumerate(rows, 1):
        disp = piece.replace("\n", "\\n").replace("\t", "\\t")
        print(f"{rank:2d}. id={tid:6d} piece={disp:<12} p={p:.6f}")


def build_prompt(tokenizer, text, use_chat_template: bool):
    if not use_chat_template:
        return text
    if hasattr(tokenizer, "apply_chat_template"):
        msgs = [{"role": "user", "content": text}]
        return tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    return f"USER: {text}\nASSISTANT:"


@torch.no_grad()
def target_hidden(target, input_ids, attn_mask, enc_layer_index):
    out = target(
        input_ids=input_ids,
        attention_mask=attn_mask,
        output_hidden_states=True,
        use_cache=False,
        return_dict=True,
    )
    return out.hidden_states[enc_layer_index], out.logits


@torch.no_grad()
def draft_d1_topk(draft: Effective_Draft_Decoder,
                 target: AutoModelForCausalLM,
                 tokenizer,
                 prompt_ids,
                 prompt_mask,
                 seed_id: int,
                 enc_layer_index: int,
                 topk: int):
    # enc_hidden for prompt
    enc_hidden, _ = target_hidden(target, prompt_ids, prompt_mask, enc_layer_index)

    # Build a minimal token stream of length 1 (seed only).
    # We feed enc_hidden + tok_emb(seed) into the draft decoder and read the first token-stream logit.
    B, T, H = enc_hidden.shape
    seed = torch.tensor([[seed_id]], device=enc_hidden.device, dtype=prompt_ids.dtype)  # (1,1)
    tok_emb = draft.embedding_layer(seed) if hasattr(draft, "embedding_layer") else draft.get_input_embeddings()(seed)
    hidden_states = torch.cat([enc_hidden, tok_emb], dim=1)  # (1, T+1, H)

    # position_ids: [0..T-1] for enc, then [T] for seed token stream
    pos = torch.arange(T + 1, device=enc_hidden.device).unsqueeze(0)  # (1, T+1)

    # Build dual-block mask the same way the model does internally
    attn = draft._build_dual_block_mask(
        enc_len=T,
        total_len=T + 1,
        inp_len=T,      # prompt length
        pred_len=1,     # token stream length
        device=enc_hidden.device,
        dtype=enc_hidden.dtype,
    )
    if attn.size(0) != B:
        attn = attn.expand(B, -1, -1, -1)

    # Run draft layers
    position_embeddings = draft._compute_position_embeddings(hidden_states, pos)
    x = hidden_states

        
    for layer in draft.decoder_layers:
        # --- dtype align (fix float vs bf16 mismatch) ---
        layer_dtype = next(layer.parameters()).dtype
        x = x.to(layer_dtype)
        attn = attn.to(layer_dtype)

        if position_embeddings is not None:
            cos, sin = position_embeddings
            position_embeddings = (cos.to(layer_dtype), sin.to(layer_dtype))
        try:
            out = layer(x, attention_mask=attn, position_ids=pos, position_embeddings=position_embeddings)
        except TypeError:
            out = layer(x, attention_mask=attn, position_ids=pos)
        x = out[0] if isinstance(out, (tuple, list)) else out
    x = draft.norm(x)
    logits = draft.lm_head(x)  # (1, T+1, V)

    # token-stream is last pred_len positions => last position corresponds to seed position.
    # We want d1: next token AFTER seed, which in Faster-SD alignment is the token-stream logit at that position.
    # Practically: use the last position logits.
    d1_logits = logits[:, -1, :]  # (1,V)
    return topk_from_logits(d1_logits, topk, tokenizer)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target_model", required=True)
    ap.add_argument("--edd_dir", required=True, help="checkpoint dir containing tokenizer files + draft_decoder_final.pt")
    ap.add_argument("--draft_ckpt", default=None, help="path to .pt (default: <edd_dir>/draft_decoder_final.pt)")
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--use_chat_template", action="store_true", help="wrap prompt as a single user turn via chat template")
    ap.add_argument("--enc_layer_index", type=int, default=-4)
    ap.add_argument("--num_layers", type=int, default=1, help="must match training --num_layers")
    ap.add_argument("--topk", type=int, default=20)
    ap.add_argument("--bf16", action="store_true")
    ap.add_argument("--fp16", action="store_true")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    if args.bf16 and args.fp16:
        raise ValueError("choose at most one of --bf16/--fp16")

    dtype = torch.bfloat16 if args.bf16 else (torch.float16 if args.fp16 else torch.float32)
    device = torch.device(args.device)

    ckpt = args.draft_ckpt or os.path.join(args.edd_dir, "draft_decoder_final.pt")
    assert os.path.exists(ckpt), f"not found: {ckpt}"

    # IMPORTANT: load tokenizer from edd_dir so chat template/special tokens match training
    tok = AutoTokenizer.from_pretrained(args.edd_dir, trust_remote_code=True, use_fast=True)
    if tok.pad_token is None and tok.eos_token is not None:
        tok.pad_token = tok.eos_token

    prompt_text = build_prompt(tok, args.prompt, args.use_chat_template)
    enc = tok(prompt_text, return_tensors="pt")
    input_ids = enc["input_ids"].to(device)
    attn_mask = enc.get("attention_mask", torch.ones_like(input_ids)).to(device)

    # teacher/target
    target = AutoModelForCausalLM.from_pretrained(args.target_model, torch_dtype=dtype, trust_remote_code=True).to(device).eval()
    for p in target.parameters():
        p.requires_grad_(False)

    # draft: build from teacher then load weights
    draft = Effective_Draft_Decoder.from_teacher(target, num_layers=args.num_layers)  # num_layers doesn't matter for loading if ckpt matches
    draft.to(device).eval()
    sd = torch.load(ckpt, map_location="cpu")
    if isinstance(sd, dict) and any(k.startswith("module.") for k in sd.keys()):
        sd = {k.replace("module.", "", 1): v for k, v in sd.items()}
    draft.load_state_dict(sd, strict=True)
    
    draft.eval()
    if args.bf16:
        draft = draft.to(device=device, dtype=torch.bfloat16)
    else:
        draft = draft.to(device=device)
        # seed
    seed_id = greedy_next_id(target, input_ids, attn_mask)
    seed_piece = tok.decode([seed_id])

    # target topk for t(i+2) on [prompt, seed]
    seed_tensor = torch.tensor([[seed_id]], device=device, dtype=input_ids.dtype)
    ids2 = torch.cat([input_ids, seed_tensor], dim=1)
    mask2 = torch.cat([attn_mask, torch.ones_like(seed_tensor)], dim=1)
    out2 = target(input_ids=ids2, attention_mask=mask2, use_cache=False, return_dict=True)
    t_i2_logits = out2.logits[:, -1, :]  # next after seed
    tgt_rows = topk_from_logits(t_i2_logits, args.topk, tok)

    # draft topk for d1
    dr_rows = draft_d1_topk(draft, target, tok, input_ids, attn_mask, seed_id, args.enc_layer_index, args.topk)

    print("=" * 80)
    print("PROMPT:")
    print(prompt_text)
    print("=" * 80)
    print(f"Seed (target greedy next after prompt): id={seed_id} piece={seed_piece!r}")
    print("-" * 80)
    fmt(tgt_rows, f"[Target] top{args.topk} for t(i+2) (next after seed), i.e. target on [prompt, seed]")
    print("-" * 80)
    fmt(dr_rows, f"[Draft] top{args.topk} for d1 (first draft token after seed) using enc_layer_index={args.enc_layer_index}")
    print("=" * 80)

    tgt_ids = {tid for tid, _, _ in tgt_rows}
    dr_ids = {tid for tid, _, _ in dr_rows}
    overlap = sorted(list(tgt_ids & dr_ids))
    print(f"Overlap token-ids between Target top{args.topk} and Draft top{args.topk}: {overlap}")


if __name__ == "__main__":
    main()
