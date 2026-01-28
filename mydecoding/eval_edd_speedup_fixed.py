# eval_edd_speedup_fixed.py
import argparse, random
from dataclasses import dataclass
from typing import List

import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM

from mydecoding.models.edd_draft_model import EDDDraftModel  # 按你的项目路径改一下


@dataclass
class Batch:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor


class ShareGPTDataset(Dataset):
    def __init__(self, path: str, tokenizer, max_length: int = 256):
        import json
        self.samples = []
        self.tokenizer = tokenizer
        self.max_length = max_length

        def load_jsonl(p):
            items = []
            with open(p, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        items.append(json.loads(line))
            return items

        items = load_jsonl(path)
        for sample in items:
            conv = sample.get("conversations", [])
            msgs = []
            for turn in conv:
                frm = turn.get("from")
                val = turn.get("value", "")
                if frm in ("human", "user"):
                    role = "user"
                elif frm in ("gpt", "assistant"):
                    role = "assistant"
                else:
                    continue
                msgs.append({"role": role, "content": val})
            if not msgs:
                continue
            try:
                text = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
            except Exception:
                text = ""
                for m in msgs:
                    text += f"{m['role'].upper()}: {m['content']}\n"
            self.samples.append(text)

    def __len__(self):
        return len(self.samples)

    def get_tokenized(self, idx: int):
        enc = self.tokenizer(
            self.samples[idx],
            truncation=True,
            max_length=self.max_length,
            padding=False,
            return_tensors="pt",
        )
        return {"input_ids": enc["input_ids"][0], "attention_mask": enc["attention_mask"][0]}


def longest_prefix_match(a: torch.Tensor, b: torch.Tensor) -> int:
    m = 0
    for i in range(min(a.numel(), b.numel())):
        if int(a[i].item()) == int(b[i].item()):
            m += 1
        else:
            break
    return m


@torch.no_grad()
def target_greedy_k(target, ctx_ids, ctx_attn, K: int) -> torch.Tensor:
    gen = target.generate(
        input_ids=ctx_ids,
        attention_mask=ctx_attn,
        max_new_tokens=K,
        do_sample=False,
        use_cache=True,
    )
    return gen[:, -K:]


@torch.no_grad()
def edd_next_token_logits(
    edd: EDDDraftModel,
    target: AutoModelForCausalLM,
    ctx_ids: torch.Tensor,      # (1,t)
    ctx_attn: torch.Tensor,     # (1,t)
    gen_ids: torch.Tensor,      # (1,s)  已经生成的draft tokens
    enc_layer_index: int,
    block_len: int,
) -> torch.Tensor:
    """
    返回对“下一个 token”的 logits: (1,V)
    关键：构造训练同款 [enc_hidden, token_embeds] 的 2T 输入，并从 token-half 取最后一位 logits。
    """
    device = ctx_ids.device
    dtype = next(edd.parameters()).dtype

    # 1) 用 target 编码得到 enc_hidden（只对 ctx 这段）
    out_t = target(
        input_ids=ctx_ids,
        attention_mask=ctx_attn,
        output_hidden_states=True,
        use_cache=False,
        return_dict=True,
    )
    enc_hidden = out_t.hidden_states[enc_layer_index].to(dtype=dtype)  # (1,t,H)

    # 2) token-half：ctx + gen
    if gen_ids is None:
        tok_ids = ctx_ids
    else:
        tok_ids = torch.cat([ctx_ids, gen_ids], dim=1)  # (1, t+s)
    T = tok_ids.size(1)

    # 3) enc-half：已知的 ctx enc_hidden + 对 gen 的占位 0（因为训练的 dual-block 会避免同块使用未来 enc）
    if T > enc_hidden.size(1):
        pad_len = T - enc_hidden.size(1)
        pad = torch.zeros((1, pad_len, enc_hidden.size(-1)), device=device, dtype=dtype)
        enc_half = torch.cat([enc_hidden, pad], dim=1)  # (1,T,H)
    else:
        enc_half = enc_hidden[:, :T, :]

    tok_embeds = edd.draft_lm.get_input_embeddings()(tok_ids).to(dtype=dtype)  # (1,T,H)
    inputs_embeds = torch.cat([enc_half, tok_embeds], dim=1)  # (1,2T,H)

    # 4) dual-block mask（训练同款）
    attn_1d = torch.ones((1, T), device=device, dtype=ctx_attn.dtype)
    attn_bias = edd._dual_block_attn_bias(attn_1d, block_len, dtype=inputs_embeds.dtype)

    out_d = edd.draft_lm.model(
        inputs_embeds=inputs_embeds,
        attention_mask=attn_bias,   # 4D additive mask
        use_cache=False,
        return_dict=True,
    )
    hidden = out_d.last_hidden_state               # (1,2T,H)
    hidden_tok = hidden[:, T:, :]                  # (1,T,H)
    logits_tok = edd.draft_lm.lm_head(hidden_tok)  # (1,T,V)

    return logits_tok[:, -1, :]  # “下一 token”分布


@torch.no_grad()
def edd_propose_k(
    edd: EDDDraftModel,
    target: AutoModelForCausalLM,
    ctx_ids: torch.Tensor,
    ctx_attn: torch.Tensor,
    K: int,
    enc_layer_index: int,
    block_len: int,
) -> torch.Tensor:
    gen_ids = None
    outs = []
    for _ in range(K):
        logits = edd_next_token_logits(
            edd=edd,
            target=target,
            ctx_ids=ctx_ids,
            ctx_attn=ctx_attn,
            gen_ids=gen_ids,
            enc_layer_index=enc_layer_index,
            block_len=block_len,
        )
        next_id = torch.argmax(logits, dim=-1, keepdim=True)  # (1,1)
        outs.append(next_id)
        gen_ids = next_id if gen_ids is None else torch.cat([gen_ids, next_id], dim=1)
    return torch.cat(outs, dim=1)  # (1,K)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target_model", type=str, required=True)
    ap.add_argument("--edd_dir", type=str, required=True)
    ap.add_argument("--data_jsonl", type=str, required=True)
    ap.add_argument("--num_samples", type=int, default=200)
    ap.add_argument("--seq_len", type=int, default=256)
    ap.add_argument("--min_t", type=int, default=32)
    ap.add_argument("--K", type=int, default=5)
    ap.add_argument("--enc_layer_index", type=int, default=-4)
    ap.add_argument("--block_len", type=int, default=10)  # 论文里 depth=10；你也可以设成 5
    ap.add_argument("--bf16", action="store_true")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if args.bf16 else torch.float16

    tokenizer = AutoTokenizer.from_pretrained(args.target_model, use_fast=True, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    target = AutoModelForCausalLM.from_pretrained(
        args.target_model,
        torch_dtype=dtype,
        device_map=None,
        trust_remote_code=True,
    ).to(device)
    target.eval()
    for p in target.parameters():
        p.requires_grad_(False)

    # 用 wrapper 载入 draft，这样能拿到 dual-block mask + 正确的 token-half logits 逻辑
    edd = EDDDraftModel.from_pretrained(args.edd_dir, trust_remote_code=True).to(device)
    edd.eval()

    ds = ShareGPTDataset(args.data_jsonl, tokenizer, max_length=args.seq_len)

    accepted_lens: List[int] = []
    accept_at = [0 for _ in range(args.K + 1)]

    n = 0
    idx = 0
    while n < args.num_samples and idx < len(ds):
        ex = ds.get_tokenized(idx)
        idx += 1
        input_ids = ex["input_ids"].unsqueeze(0).to(device)
        attn = ex["attention_mask"].unsqueeze(0).to(device)

        valid_len = int(attn.sum().item())
        if valid_len <= args.min_t + args.K + 1:
            continue

        ctx_len = random.randint(args.min_t, min(valid_len - args.K - 1, args.seq_len - args.K - 1))
        ctx_ids = input_ids[:, :ctx_len]
        ctx_attn = attn[:, :ctx_len]

        teacher_tokens = target_greedy_k(target, ctx_ids, ctx_attn, args.K)
        student_tokens = edd_propose_k(edd, target, ctx_ids, ctx_attn, args.K, args.enc_layer_index, args.block_len)

        m = longest_prefix_match(student_tokens[0], teacher_tokens[0])
        accepted_lens.append(m)
        for j in range(1, args.K + 1):
            if m >= j:
                accept_at[j] += 1
        n += 1

    avg_accept = sum(accepted_lens) / max(n, 1)
    print("==================================")
    print("[EDD Speedup Eval] (target greedy as reference) FIXED")
    print(f"num_samples = {n}")
    print(f"K = {args.K}")
    print(f"block_len = {args.block_len}")
    print(f"enc_layer_index = {args.enc_layer_index}")
    print(f"avg_accepted_len = {avg_accept:.3f}")
    for j in range(1, args.K + 1):
        print(f"accept@{j}: {accept_at[j]}/{n} = {accept_at[j]/n:.4f}")
    print(f"rough_speedup_proxy ≈ 1 + avg_accepted_len = {1.0 + avg_accept:.3f}")
    print("==================================")


if __name__ == "__main__":
    main()
