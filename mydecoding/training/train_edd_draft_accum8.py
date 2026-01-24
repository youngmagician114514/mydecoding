# train_edd_draft_masked_accum8.py
# Faster EDD draft training: single forward with dual-block attention mask + grad accumulation (1 -> 8) + timing.

from __future__ import annotations

import argparse
import json
import os
import random
import math
import time
from dataclasses import dataclass
from typing import Any, Dict, List

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

from mydecoding.models.edd_draft_model import EDDDraftModel


@dataclass
class Batch:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor


class ShareGPTDataset(Dataset):
    def __init__(self, path: str, tokenizer, max_length: int = 1024, json_root_keys=("data", "train", "val")):
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
            if isinstance(obj, list):
                items = obj
            elif isinstance(obj, dict):
                items = None
                for k in json_root_keys:
                    if k in obj and isinstance(obj[k], list):
                        items = obj[k]
                        break
                if items is None:
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
        batch = self.tokenizer.pad(items, padding=True, return_tensors="pt")
        return Batch(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--target_model", type=str, required=True)
    p.add_argument("--data_jsonl", type=str, required=True)
    p.add_argument("--output_dir", type=str, required=True)

    p.add_argument("--max_length", type=int, default=1024)   # recommend 1024 first (2T attention!)
    p.add_argument("--micro_batch_size", type=int, default=1)
    p.add_argument("--grad_accum", type=int, default=8)

    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--warmup_ratio", type=float, default=0.03)

    p.add_argument("--block_len_min", type=int, default=5)
    p.add_argument("--block_len_max", type=int, default=10)
    p.add_argument("--enc_layer_index", type=int, default=-4)
    p.add_argument("--kl_mode", type=str, default="full", choices=["full", "topk"])
    p.add_argument("--topk", type=int, default=32, help="Only used when --kl_mode topk")
    p.add_argument("--kl_chunk_size", type=int, default=4096, help="Only used when --kl_mode full")

    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--bf16", action="store_true")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--num_workers", type=int, default=2)

    p.add_argument("--log_every_updates", type=int, default=10)
    return p.parse_args()


def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def cuda_sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    set_seed(args.seed)

    dtype = torch.bfloat16 if args.bf16 else torch.float16

    tokenizer = AutoTokenizer.from_pretrained(args.target_model, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

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
        batch_size=args.micro_batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=PadCollator(tokenizer),
        pin_memory=True,
    )
    if args.micro_batch_size != 1:
        raise ValueError("This script is intended for micro_batch_size=1. Set --micro_batch_size 1.")

    print("Loaded samples:", len(dataset))

    optim = torch.optim.AdamW(edd.parameters(), lr=args.lr)

    total_micro_steps = args.epochs * len(loader)
    total_updates = math.ceil(total_micro_steps / args.grad_accum)
    warmup_updates = int(total_updates * args.warmup_ratio)
    sched = get_linear_schedule_with_warmup(optim, warmup_updates, total_updates)

    edd.train()
    target.eval()

    micro_step = 0
    update_step = 0

    win_teacher = 0.0
    win_draft = 0.0
    win_total = 0.0

    optim.zero_grad(set_to_none=True)
    cuda_sync()
    win_start_t = time.perf_counter()

    for epoch in range(args.epochs):
        for batch in loader:
            micro_step += 1

            cuda_sync()
            t0 = time.perf_counter()

            input_ids = batch.input_ids.to(device, non_blocking=True)
            attention_mask = batch.attention_mask.to(device, non_blocking=True)

            # teacher encode (frozen)
            with torch.no_grad():
                cuda_sync()
                t_enc0 = time.perf_counter()
                enc = edd.encode_with_target(
                    target_model=target,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    enc_layer_index=args.enc_layer_index,
                )
                cuda_sync()
                t_enc1 = time.perf_counter()

            # random block length
            L = random.randint(args.block_len_min, args.block_len_max)

            # draft single-forward loss
            cuda_sync()
            t_d0 = time.perf_counter()

            loss = edd.kl_loss_dual_block_masked(
                input_ids=input_ids,
                attention_mask=attention_mask,
                teacher_logits=enc.teacher_logits,
                enc_hidden=enc.enc_hidden,
                block_len=L,
                kl_mode=args.kl_mode,
                topk=args.topk,
                kl_chunk_size=args.kl_chunk_size,
            )

            (loss / args.grad_accum).backward()

            cuda_sync()
            t_d1 = time.perf_counter()

            cuda_sync()
            t1 = time.perf_counter()

            micro_teacher = t_enc1 - t_enc0
            micro_draft = t_d1 - t_d0
            micro_total = t1 - t0

            win_teacher += micro_teacher
            win_draft += micro_draft
            win_total += micro_total

            if micro_step % 20 == 0:
                lr_now = sched.get_last_lr()[0] if update_step > 0 else optim.param_groups[0]["lr"]
                print(
                    f"[micro] epoch={epoch} micro={micro_step}/{total_micro_steps} "
                    f"L={L} loss={loss.item():.6f} "
                    f"t_total={micro_total:.3f}s (teacher={micro_teacher:.3f}s draft={micro_draft:.3f}s) "
                    f"lr={lr_now:.2e}"
                )

            do_update = (micro_step % args.grad_accum == 0) or (micro_step == total_micro_steps)
            if do_update:
                update_step += 1
                torch.nn.utils.clip_grad_norm_(edd.parameters(), 1.0)
                optim.step()
                sched.step()
                optim.zero_grad(set_to_none=True)

                cuda_sync()
                win_end_t = time.perf_counter()
                wall = win_end_t - win_start_t

                if (update_step % args.log_every_updates == 0) or (update_step <= 3):
                    print(
                        f"[update] epoch={epoch} update={update_step}/{total_updates} "
                        f"window_wall={wall:.2f}s "
                        f"sum_teacher={win_teacher:.2f}s sum_draft={win_draft:.2f}s sum_total={win_total:.2f}s "
                        f"lr={sched.get_last_lr()[0]:.2e}"
                    )

                win_teacher = 0.0
                win_draft = 0.0
                win_total = 0.0
                win_start_t = time.perf_counter()

    edd.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"Saved EDD draft model to: {args.output_dir}")


if __name__ == "__main__":
    main()
