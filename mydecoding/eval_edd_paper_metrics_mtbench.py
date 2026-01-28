import argparse
import random
import time
from typing import List, Tuple

import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM


def build_chat_prompt(tokenizer, user_text: str) -> str:
    msgs = [{"role": "user", "content": user_text}]
    return tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


def to_cache(past):
    if past is None:
        return None
    if hasattr(past, "get_seq_length"):
        return past
    from transformers.cache_utils import DynamicCache
    return DynamicCache.from_legacy_cache(past)


@torch.no_grad()
def target_init_with_cache_and_enchidden(target, input_ids, attention_mask, enc_layer_index: int):
    """
    Run target on the prompt once, return:
      - past cache
      - next greedy token (1,1)
      - enc_hidden buffer for the prompt (1, T, H) at layer enc_layer_index
    """
    out = target(
        input_ids=input_ids,
        attention_mask=attention_mask,
        output_hidden_states=True,
        use_cache=True,
        return_dict=True,
    )
    past = to_cache(out.past_key_values)
    next_tok = torch.argmax(out.logits[:, -1, :], dim=-1, keepdim=True)  # (1,1)
    enc_hidden = out.hidden_states[enc_layer_index]  # (1,T,H)
    return past, next_tok, enc_hidden


@torch.no_grad()
def target_step(target, past, token_id: torch.Tensor, enc_layer_index: int):
    """
    Feed ONE token with cache, return updated:
      - past
      - next greedy token (1,1)
      - enc_hidden for THIS token only (1,1,H) at enc_layer_index
    """
    out = target(
        input_ids=token_id,  # (1,1)
        attention_mask=torch.ones_like(token_id),
        past_key_values=past,
        output_hidden_states=True,
        use_cache=True,
        return_dict=True,
    )
    past = to_cache(out.past_key_values)
    next_tok = torch.argmax(out.logits[:, -1, :], dim=-1, keepdim=True)  # (1,1)
    h_new = out.hidden_states[enc_layer_index]  # (1,1,H) for the new token
    return past, next_tok, h_new


@torch.no_grad()
def draft_propose_k(draft, enc_hidden_buf: torch.Tensor, K: int) -> torch.Tensor:
    """
    Propose K tokens greedily using the same soft-prompt idea as your eval_edd_speedup:
    inputs_embeds = [enc_hidden_buf ; tok_embeds(block_ids)]
    We keep K small (5), so recomputing is fine and still fast enough.
    """
    device = enc_hidden_buf.device
    block_ids = None
    gen_tokens = []
    for _ in range(K):
        if block_ids is None:
            inputs_embeds = enc_hidden_buf
        else:
            tok_embeds = draft.get_input_embeddings()(block_ids)
            inputs_embeds = torch.cat([enc_hidden_buf, tok_embeds], dim=1)

        attn_mask = torch.ones((1, inputs_embeds.size(1)), device=device, dtype=torch.long)
        out = draft.model(
            inputs_embeds=inputs_embeds,
            attention_mask=attn_mask,
            use_cache=False,
            return_dict=True,
        )
        h_last = out.last_hidden_state[:, -1, :]
        logits = draft.lm_head(h_last)
        next_id = torch.argmax(logits, dim=-1, keepdim=True)  # (1,1)
        gen_tokens.append(next_id)
        block_ids = next_id if block_ids is None else torch.cat([block_ids, next_id], dim=1)

    return torch.cat(gen_tokens, dim=1)  # (1,K)


def load_mtbench_turns(num_samples: int, seed: int) -> List[str]:
    random.seed(seed)
    turns: List[str] = []
    try:
        ds = load_dataset("lmsys/mt_bench", "question", split="train")
        items = list(ds)
        random.shuffle(items)
        for ex in items:
            t = ex.get("turns", None)
            if isinstance(t, (list, tuple)):
                for s in t:
                    if isinstance(s, str) and s.strip():
                        turns.append(s.strip())
            elif isinstance(t, str) and t.strip():
                turns.append(t.strip())
            if len(turns) >= num_samples:
                break
    except Exception:
        ds = load_dataset("HuggingFaceH4/mt_bench_prompts", split="train")
        items = list(ds)
        random.shuffle(items)
        for ex in items:
            p = ex.get("prompt", None)
            if isinstance(p, str) and p.strip():
                turns.append(p.strip())
            elif isinstance(p, (list, tuple)):
                for s in p:
                    if isinstance(s, str) and s.strip():
                        turns.append(s.strip())
            if len(turns) >= num_samples:
                break
    return turns[:num_samples]


@torch.no_grad()
def eval_tau_alpha_fast(
    target, draft, tokenizer, prompts: List[str],
    K: int, enc_layer_index: int, gen_len: int, device: str
) -> Tuple[float, float, List[int], int, float]:
    """
    Paper-style:
      - each iteration propose K (draft greedy)
      - verify with target greedy (using cache, token-by-token)
      - accept m in [0..K], then (if m<K) take 1 target token
    Report:
      tau = mean(m)
      alpha = tau/K
      accept@j = P(m >= j)
    """
    accept_ge = [0] * (K + 1)
    m_list: List[int] = []
    t0 = time.time()

    for p in prompts:
        prompt = build_chat_prompt(tokenizer, p)
        enc = tokenizer(prompt, return_tensors="pt", padding=False)
        ctx_ids = enc["input_ids"].to(device)
        ctx_attn = enc.get("attention_mask", torch.ones_like(ctx_ids)).to(device)

        past, next_tok, enc_hidden_buf = target_init_with_cache_and_enchidden(
            target, ctx_ids, ctx_attn, enc_layer_index
        )
        produced = 0

        while produced < gen_len:
            # 1) draft propose K based on CURRENT enc_hidden_buf
            student_k = draft_propose_k(draft, enc_hidden_buf, K=K)  # (1,K)

            # 2) verify token-by-token using target greedy next_tok and cache
            m = 0
            cur_next = next_tok  # (1,1)

            # We will always consume at least 1 token per iter (either accepted or mismatch target token)
            for i in range(K):
                t_i = int(cur_next.item())
                if int(student_k[0, i].item()) == t_i:
                    m += 1
                    chosen = cur_next  # accept
                else:
                    chosen = torch.tensor([[t_i]], device=device, dtype=torch.long)  # take target token
                    # mismatch => break after consuming this target token
                    pass

                # advance target cache by chosen token, and append its enc_hidden
                past, cur_next, h_new = target_step(target, past, chosen, enc_layer_index)
                enc_hidden_buf = torch.cat([enc_hidden_buf, h_new], dim=1)

                produced += 1
                if produced >= gen_len:
                    break

                # mismatch: stop verifying remaining draft tokens
                if int(student_k[0, i].item()) != t_i:
                    break

            m_list.append(m)
            for j in range(1, K + 1):
                if m >= j:
                    accept_ge[j] += 1

            next_tok = cur_next

    wall = time.time() - t0
    iters = len(m_list)
    tau = sum(m_list) / max(iters, 1)
    alpha = tau / K
    return tau, alpha, accept_ge, iters, wall


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target_model", required=True)
    ap.add_argument("--edd_dir", required=True)
    ap.add_argument("--bf16", action="store_true")
    ap.add_argument("--K", type=int, default=5)
    ap.add_argument("--enc_layer_index", type=int, default=-4)
    ap.add_argument("--num_samples", type=int, default=80)
    ap.add_argument("--gen_len", type=int, default=128)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if args.bf16 else torch.float16

    tokenizer = AutoTokenizer.from_pretrained(args.target_model, use_fast=True, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    target = AutoModelForCausalLM.from_pretrained(
        args.target_model, torch_dtype=dtype, device_map=None, trust_remote_code=True
    ).to(device).eval()
    for p in target.parameters():
        p.requires_grad_(False)

    draft = AutoModelForCausalLM.from_pretrained(
        args.edd_dir, torch_dtype=dtype, device_map=None, trust_remote_code=True
    ).to(device).eval()

    prompts = load_mtbench_turns(args.num_samples, seed=args.seed)
    if not prompts:
        raise RuntimeError("No MT-bench prompts loaded.")

    tau, alpha, accept_ge, iters, wall = eval_tau_alpha_fast(
        target=target,
        draft=draft,
        tokenizer=tokenizer,
        prompts=prompts,
        K=args.K,
        enc_layer_index=args.enc_layer_index,
        gen_len=args.gen_len,
        device=device,
    )

    print("==================================")
    print("[EDD Paper-metrics (tau/alpha) on MT-bench prompts] FAST (cache + incremental enc_hidden)")
    print(f"num_samples(turns) = {len(prompts)}")
    print(f"gen_len per prompt = {args.gen_len}")
    print(f"K = {args.K}")
    print(f"enc_layer_index = {args.enc_layer_index}")
    print(f"tau (avg accepted tokens/iter) = {tau:.3f}")
    print(f"alpha (acceptance rate)        = {alpha:.3f}")
    for j in range(1, args.K + 1):
        print(f"accept@{j} (per-iter) = {accept_ge[j]}/{iters} = {accept_ge[j]/max(iters,1):.4f}")
    print(f"wall_time = {wall:.1f}s")
    print("==================================")


if __name__ == "__main__":
    main()
