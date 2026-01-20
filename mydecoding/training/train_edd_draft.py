# train_edd_draft.py
# Train EDD draft model (only draft part; no PCT/tree)
#
# Paper-aligned essentials:
# - draft = same structure as target LLM but only 1 Transformer layer
# - copy embedding + lm_head from target for initialization; update all draft params
# - frozen target provides hidden states H and teacher distribution Pt
# - dual-block mask training; randomly sample block length each step
# - loss = KL(Pt || Pd)
# - impl details: block_len in [5,10], enc layer = 4th-to-last, bs=8, lr=1e-4, AdamW, 1 epoch, keep final ckpt

from __future__ import annotations

import argparse
import json
import os
import random
from dataclasses import dataclass
from typing import Any, Dict, List

import torch
from torch.utils.data import Dataset, DataLoader

from transformers import AutoTokenizer, get_linear_schedule_with_warmup

from mydecoding.models.edd_draft_model import EDDDraftModel


@dataclass
class Batch:
    input_ids: torch.Tensor        # (B, T)
    attention_mask: torch.Tensor   # (B, T)


class ShareGPTDataset(Dataset):
    """
    Supports:
      - .jsonl: each line is one sample json
      - .json: a list of samples, OR {"data": [...]} / {"train": [...]} etc.
    Each sample should contain:
      {"conversations":[{"from":"human","value":"..."}, {"from":"gpt","value":"..."}, ...]}
    """
    def __init__(self, path: str, tokenizer, max_length: int = 2048, json_root_keys=("data", "train", "val")):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples: List[str] = []

        path_lower = path.lower()
        if path_lower.endswith(".jsonl"):
            items = []
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    items.append(json.loads(line))
        elif path_lower.endswith(".json"):
            with open(path, "r", encoding="utf-8") as f:
                obj = json.load(f)
            # obj could be list or dict
            if isinstance(obj, list):
                items = obj
            elif isinstance(obj, dict):
                # try common roots
                items = None
                for k in json_root_keys:
                    if k in obj and isinstance(obj[k], list):
                        items = obj[k]
                        break
                if items is None:
                    # fallback: if dict values contain the list
                    # (keeps it safe; you can hardcode if you know exact structure)
                    for v in obj.values():
                        if isinstance(v, list):
                            items = v
                            break
                if items is None:
                    raise ValueError(f"Unsupported JSON dict structure in {path}. Keys: {list(obj.keys())[:20]}")
            else:
                raise ValueError(f"Unsupported JSON type: {type(obj)}")
        else:
            raise ValueError("data file must end with .json or .jsonl")

        # Convert ShareGPT items -> rendered text
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
                text = tokenizer.apply_chat_template(
                    msgs,
                    tokenize=False,
                    add_generation_prompt=False,
                )
            except Exception:
                text = ""
                for m in msgs:
                    text += f"{m['role'].upper()}: {m['content']}\n"

            self.samples.append(text)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        text = self.samples[idx]
        enc = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding=False,
            return_tensors=None,
        )
        return {"input_ids": enc["input_ids"], "attention_mask": enc["attention_mask"]}


class PadCollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, items: List[Dict[str, Any]]) -> Batch:
        batch = self.tokenizer.pad(
            items,
            padding=True,
            return_tensors="pt",
        )
        return Batch(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
        )


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--target_model", type=str, required=True,
                   help="e.g. Qwen/Qwen2.5-3B-Instruct (or your Qwen 3B path)")
    p.add_argument("--data_jsonl", type=str, required=True,
                   help="ShareGPT-format jsonl")
    p.add_argument("--output_dir", type=str, required=True)

    p.add_argument("--max_length", type=int, default=2048)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--warmup_ratio", type=float, default=0.03)

    p.add_argument("--block_len_min", type=int, default=5)
    p.add_argument("--block_len_max", type=int, default=10)
    p.add_argument("--enc_layer_index", type=int, default=-4,
                   help="4th-to-last layer hidden state (default -4)")

    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--bf16", action="store_true")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--num_workers", type=int, default=2)
    return p.parse_args()


def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    set_seed(args.seed)

    dtype = torch.bfloat16 if args.bf16 else torch.float16

    tokenizer = AutoTokenizer.from_pretrained(args.target_model, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        # common safe fallback
        tokenizer.pad_token = tokenizer.eos_token

    # Build (draft, frozen_target)
    edd, target = EDDDraftModel.from_target(
        args.target_model,
        torch_dtype=dtype,
        device_map=None,
        trust_remote_code=True,
    )

    device = torch.device(args.device)
    edd = edd.to(device)
    target = target.to(device)

    dataset = ShareGPTDataset(args.data_jsonl, tokenizer, max_length=args.max_length)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=PadCollator(tokenizer),
        pin_memory=True,
    )
    
    print("Loaded samples:", len(dataset))
    optim = torch.optim.AdamW(edd.parameters(), lr=args.lr)

    total_steps = args.epochs * len(loader)
    warmup_steps = int(total_steps * args.warmup_ratio)
    sched = get_linear_schedule_with_warmup(optim, warmup_steps, total_steps)

    edd.train()
    target.eval()

    global_step = 0
    for epoch in range(args.epochs):
        for batch in loader:
            global_step += 1
            input_ids = batch.input_ids.to(device)
            attention_mask = batch.attention_mask.to(device)

            # 1) Encode with frozen target to get H and Pt
            with torch.no_grad():
                enc = edd.encode_with_target(
                    target_model=target,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    enc_layer_index=args.enc_layer_index,
                )

            # 2) Random block length in [5,10]
            L = random.randint(args.block_len_min, args.block_len_max)

            optim.zero_grad(set_to_none=True)

            # 先算一遍总 weight（只依赖 attention_mask，不需要梯度）
            B, T = input_ids.shape
            total_weight = 0.0
            for s in range(0, T, L):
                e = min(s + L, T)
                if s >= T - 1:
                    break
                cur_valid = attention_mask[:, s:e]
                if e < T:
                    nxt_valid = attention_mask[:, s+1:e+1]
                    valid = (cur_valid * nxt_valid).float()
                else:
                    valid = cur_valid.float()
                    valid[:, -1] = 0.0
                total_weight += valid.sum().item()
            total_weight = max(total_weight, 1.0)

            # 逐 block backward：每个 block 的图用完就释放，不会攒爆显存
            loss_value = 0.0
            for s in range(0, T, L):
                e = min(s + L, T)
                if s >= T - 1:
                    break

                loss_sum, weight_sum = edd.loss_one_block(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    teacher_logits=enc.teacher_logits,
                    enc_hidden=enc.enc_hidden,
                    s=s, e=e,
                    topk=32,
                )

                # 等价于 overall (sum loss_sum)/(sum weight_sum) 的梯度，因为分母与模型无关
                (loss_sum / total_weight).backward()
                loss_value += loss_sum.detach().item()

            torch.nn.utils.clip_grad_norm_(edd.parameters(), 1.0)
            optim.step()
            sched.step()

            # 方便打印：归一化后的 loss
            loss = torch.tensor(loss_value / total_weight, device=device)

            if global_step % 50 == 0:
                print(f"epoch={epoch} step={global_step}/{total_steps} L={L} loss={loss.item():.6f}")

    # Save final checkpoint (paper keeps final ckpt)
    edd.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"Saved EDD draft model to: {args.output_dir}")


if __name__ == "__main__":
    main()
