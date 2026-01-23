import argparse
import random
import time

import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM


@torch.no_grad()
def get_enc_hidden(target, input_ids, attention_mask, enc_layer_index: int):
    out = target(
        input_ids=input_ids,
        attention_mask=attention_mask,
        output_hidden_states=True,
        use_cache=False,
        return_dict=True,
    )
    return out.hidden_states[enc_layer_index]  # (1,T,H)


@torch.no_grad()
def greedy_next_k(target, input_ids, attention_mask, k: int):
    gen = target.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        do_sample=False,
        temperature=0.0,
        max_new_tokens=k,
        use_cache=True,
    )
    return gen[:, input_ids.shape[1]:]  # (1,k)


@torch.no_grad()
def edd_propose_k(draft, enc_hidden, input_ids, attention_mask, k: int):
    """
    EDD-style propose:
      inputs_embeds = [enc_hidden ; embed(tokens_so_far)]
      greedy next token each step
    """
    device = input_ids.device
    prompt_embeds = enc_hidden  # (1, T, H)

    cur_ids = input_ids
    cur_mask = attention_mask

    out_ids = []
    for _ in range(k):
        tok_embeds = draft.get_input_embeddings()(cur_ids)  # (1,t,H)
        inputs_embeds = torch.cat([prompt_embeds, tok_embeds], dim=1)  # (1,T+t,H)
        attn_mask = torch.cat([cur_mask, cur_mask], dim=1)            # (1,T+t)

        out = draft.model(
            inputs_embeds=inputs_embeds,
            attention_mask=attn_mask,
            use_cache=False,
            return_dict=True,
        )
        last_h = out.last_hidden_state[:, -1:, :]  # (1,1,H)
        logits = draft.lm_head(last_h)             # (1,1,V)
        next_id = torch.argmax(logits, dim=-1)     # (1,1)

        out_ids.append(next_id.item())
        cur_ids = torch.cat([cur_ids, next_id.to(device)], dim=1)
        cur_mask = torch.cat([cur_mask, torch.ones_like(next_id, device=device)], dim=1)

    return out_ids


def build_chat_prompt(tokenizer, user_text: str) -> str:
    # MT-bench prompt 已经是“指令式”文本的话，也可以直接用 user role 包起来
    msgs = [{"role": "user", "content": user_text}]
    return tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target_model", required=True)
    ap.add_argument("--edd_dir", required=True)
    ap.add_argument("--bf16", action="store_true")
    ap.add_argument("--enc_layer_index", type=int, default=-4)
    ap.add_argument("--K", type=int, default=5)
    ap.add_argument("--num_samples", type=int, default=80)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    dtype = torch.bfloat16 if args.bf16 else torch.float16
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(args.target_model, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    target = AutoModelForCausalLM.from_pretrained(
        args.target_model, torch_dtype=dtype, trust_remote_code=True
    ).to(device).eval()

    draft = AutoModelForCausalLM.from_pretrained(
        args.edd_dir, torch_dtype=dtype, trust_remote_code=True
    ).to(device).eval()

    # Your schema has: category, prompt, reference, prompt_id
    ds = load_dataset("HuggingFaceH4/mt_bench_prompts", split="train")
    items = list(ds)
    random.shuffle(items)
    items = items[: min(args.num_samples, len(items))]

    print("[debug] first item keys:", list(items[0].keys()))
    p0 = items[0].get("prompt", "")
    if isinstance(p0, str):
        s0 = p0
    elif isinstance(p0, (list, tuple)):
        # list 里可能是两轮文本，或 list of dict
        if len(p0) > 0 and isinstance(p0[0], str):
            s0 = " || ".join(p0)
        else:
            s0 = str(p0)
    else:
        s0 = str(p0)

    print("[debug] example prompt (first 120 chars):", s0[:120].replace("\n", "\\n"))

    accept_counts = [0] * args.K
    sum_accepted = 0.0
    n = 0

    t0 = time.time()

    for ex in items:
        p = ex.get("prompt", None)
        if p is None:
            continue

        # prompt 可能是 str，也可能是 [turn1, turn2]
        if isinstance(p, str):
            prompt_text = p
        elif isinstance(p, (list, tuple)):
            # 默认用 turn1 做评测（最接近方案B的“单轮”）
            if len(p) == 0:
                continue
            if isinstance(p[0], str):
                prompt_text = p[0]
            else:
                prompt_text = str(p[0])
        else:
            prompt_text = str(p)

        prompt = build_chat_prompt(tokenizer, prompt_text)
        enc = tokenizer(prompt, return_tensors="pt", padding=False)
        input_ids = enc["input_ids"].to(device)
        attn = enc.get("attention_mask", torch.ones_like(input_ids)).to(device)

        enc_hidden = get_enc_hidden(target, input_ids, attn, args.enc_layer_index)
        t_next = greedy_next_k(target, input_ids, attn, args.K).squeeze(0).tolist()
        d_next = edd_propose_k(draft, enc_hidden, input_ids, attn, args.K)

        acc = 0
        for i in range(args.K):
            if d_next[i] == t_next[i]:
                acc += 1
            else:
                break

        sum_accepted += acc / args.K
        for j in range(args.K):
            if acc >= (j + 1):
                accept_counts[j] += 1
        n += 1

    wall = time.time() - t0
    if n == 0:
        raise RuntimeError("No samples evaluated (no 'prompt' field found).")

    avg_accepted_len = sum_accepted / n
    print("==================================")
    print("[EDD Speedup Eval on MT-bench prompts] (target greedy as reference)")
    print(f"num_samples = {n}")
    print(f"K = {args.K}")
    print(f"enc_layer_index = {args.enc_layer_index}")
    print(f"avg_accepted_len = {avg_accepted_len:.3f}")
    for j in range(args.K):
        print(f"accept@{j+1}: {accept_counts[j]}/{n} = {accept_counts[j]/n:.4f}")
    print(f"rough_speedup_proxy ≈ 1 + avg_accepted_len = {1.0 + avg_accepted_len:.3f}")
    print(f"wall_time = {wall:.1f}s")
    print("==================================")


if __name__ == "__main__":
    main()
