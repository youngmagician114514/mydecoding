# train_edd_draft_effective_single_gpu.py
# Single-GPU training script aligned with Faster-SD train.py logic:
# - Teacher: AutoModelForCausalLM (Qwen/Llama3.2 supported via --target_model)
# - Draft: EffectiveDraftDecoder (mask + position_ids aligned with Faster-SD)
# - Data: jsonl either in Faster-SD format {"prompt":..., "pred":...} or ShareGPT raw with "conversations"
# - Training objective: KL only (same as Faster-SD train.py)
# - Single epoch + gradient accumulation (micro_batch_size=1)
#
# Example:
#   python train_edd_draft_effective_single_gpu.py \
#     --target_model /path/to/Qwen2.5-3B-Instruct \
#     --data_jsonl /path/to/train.jsonl \
#     --output_dir ./checkpoints/draft_qwen2_3b_v1 \
#     --bf16 \
#     --max_length 2048 \
#     --min_length 128 \
#     --micro_batch_size 1 \
#     --grad_accum 8 \
#     --hidden_layer -3 \
#     --lr 1e-4 \
#     --warmup_updates 1000 \
#     --save_every_updates 1000 \
#     --num_layers 1 \
#     --block_len_min 5 --block_len_max 10

from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM

# Import Effective_Draft_Decoder from repo (matches eval scripts)
try:
    from models.edd_draft_model import Effective_Draft_Decoder  # type: ignore
except Exception:
    from mydecoding.models.edd_draft_model import Effective_Draft_Decoder  # type: ignore

def load_teacher(model_name: str, dtype: torch.dtype, device: torch.device, trust_remote_code: bool = True):
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        device_map=None,
        trust_remote_code=trust_remote_code,
    )
    model.to(device)
    model.eval()
    return model




@dataclass
class Item:
    input_ids: torch.Tensor  # (L,)
    labels: torch.Tensor     # (P,) pred tokens only


def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _safe_apply_chat_template(tok, messages: List[Dict[str, str]], add_generation_prompt: bool) -> str:
    """
    Use tokenizer.apply_chat_template if available; fallback to a simple role format.
    """
    if hasattr(tok, "apply_chat_template"):
        try:
            return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=add_generation_prompt)
        except Exception:
            pass
    # fallback
    s = ""
    for m in messages:
        s += f"{m['role'].upper()}: {m['content']}\n"
    if add_generation_prompt:
        s += "ASSISTANT: "
    return s


def _parse_sharegpt_to_prompt_pred(sample: Dict[str, Any], tok) -> Optional[Tuple[str, str]]:
    """
    Convert ShareGPT raw sample with 'conversations' into (prompt_text, pred_text) where:
      prompt_text = chat_template(up to last user, add_generation_prompt=True)
      pred_text    = last assistant content
    """
    conv = sample.get("conversations")
    if not isinstance(conv, list) or len(conv) < 2:
        return None

    turns: List[Dict[str, str]] = []
    for t in conv:
        frm = t.get("from")
        val = t.get("value", "")
        if frm in ("human", "user"):
            role = "user"
        elif frm in ("gpt", "assistant"):
            role = "assistant"
        else:
            continue
        turns.append({"role": role, "content": val})

    # Need a final assistant answer to be pred
    # Find last assistant turn
    last_ass_idx = None
    for i in range(len(turns) - 1, -1, -1):
        if turns[i]["role"] == "assistant":
            last_ass_idx = i
            break
    if last_ass_idx is None or last_ass_idx == 0:
        return None

    pred_text = turns[last_ass_idx]["content"]

    # prompt messages: everything up to the last user before that assistant
    # (common: ... user, assistant, user, assistant ... ; we want prompt ending at that user)
    # Find nearest user turn before last_ass_idx
    last_user_idx = None
    for i in range(last_ass_idx - 1, -1, -1):
        if turns[i]["role"] == "user":
            last_user_idx = i
            break
    if last_user_idx is None:
        return None

    prompt_msgs = turns[: last_user_idx + 1]  # up to last user
    prompt_text = _safe_apply_chat_template(tok, prompt_msgs, add_generation_prompt=True)

    return prompt_text, pred_text


class PromptPredDataset(Dataset):
    """
    Supports two input formats in jsonl/json:
      A) Faster-SD style jsonl: {"prompt": "...", "pred": "..."}
         - prompt is expected to already include assistant prefix (generation prompt)
         - we will build full_text = prompt + pred
      B) ShareGPT raw: {"conversations":[{"from":"human","value":...}, {"from":"gpt","value":...}, ...]}
         - we convert to (prompt_text, pred_text) using tokenizer chat template

    The dataset returns:
      input_ids: tokenize(full_text)   (prompt + pred)
      labels:    tokenize(pred_text, add_special_tokens=False)
    """
    def __init__(self, path: str, tokenizer, min_length: int, max_length: int):
        self.tok = tokenizer
        self.min_length = int(min_length)
        self.max_length = int(max_length)
        self.items: List[Item] = []

        # Load items
        path_lower = path.lower()
        raw_items: List[Dict[str, Any]] = []
        if path_lower.endswith(".jsonl"):
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    raw_items.append(json.loads(line))
        elif path_lower.endswith(".json"):
            with open(path, "r", encoding="utf-8") as f:
                obj = json.load(f)
            if isinstance(obj, list):
                raw_items = obj
            elif isinstance(obj, dict):
                # common keys
                for k in ("data", "train", "val"):
                    if k in obj and isinstance(obj[k], list):
                        raw_items = obj[k]
                        break
                if not raw_items:
                    # fallback to any list value
                    for v in obj.values():
                        if isinstance(v, list):
                            raw_items = v
                            break
            else:
                raise ValueError(f"Unsupported JSON structure: {type(obj)}")
        else:
            raise ValueError("data file must end with .json or .jsonl")

        # Convert + tokenize
        kept = 0
        for s in raw_items:
            prompt_text = None
            pred_text = None

            if "prompt" in s and "pred" in s:
                prompt_text = s["prompt"]
                pred_text = s["pred"]
            elif "conversations" in s:
                pp = _parse_sharegpt_to_prompt_pred(s, self.tok)
                if pp is not None:
                    prompt_text, pred_text = pp

            if not prompt_text or pred_text is None:
                continue

            full_text = prompt_text + pred_text

            enc_full = self.tok(full_text, add_special_tokens=True, truncation=True, max_length=self.max_length)
            enc_pred = self.tok(pred_text, add_special_tokens=False)

            input_ids = enc_full["input_ids"]
            labels = enc_pred["input_ids"]

            if len(labels) <= 10:
                continue
            if len(input_ids) < self.min_length or len(input_ids) > self.max_length:
                continue

            self.items.append(Item(
                input_ids=torch.tensor(input_ids, dtype=torch.long),
                labels=torch.tensor(labels, dtype=torch.long),
            ))
            kept += 1

        if kept == 0:
            raise RuntimeError(
                "No samples kept after filtering. "
                "Check your data format and length filters."
            )

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        it = self.items[idx]
        return {"input_ids": it.input_ids, "labels": it.labels}


def collate_batch(batch: List[Dict[str, torch.Tensor]], pad_id: int) -> Dict[str, torch.Tensor]:
    """
    Pad to max length in batch. You can keep batch_size=1 to avoid padding, but this supports >1 too.
    """
    assert len(batch) >= 1
    max_len = max(x["input_ids"].numel() for x in batch)
    max_pred = max(x["labels"].numel() for x in batch)

    input_ids = torch.full((len(batch), max_len), pad_id, dtype=torch.long)
    attn_mask = torch.zeros((len(batch), max_len), dtype=torch.long)
    labels = torch.full((len(batch), max_pred), pad_id, dtype=torch.long)

    for i, x in enumerate(batch):
        L = x["input_ids"].numel()
        P = x["labels"].numel()
        input_ids[i, :L] = x["input_ids"]
        attn_mask[i, :L] = 1
        labels[i, :P] = x["labels"]
    return {"input_ids": input_ids, "attention_mask": attn_mask, "labels": labels}


def set_lr(optimizer: torch.optim.Optimizer, lr: float):
    for g in optimizer.param_groups:
        g["lr"] = lr


def lr_schedule_cosine_with_warmup(update_idx: int, total_updates: int, warmup_updates: int, lr_max: float) -> float:
    if total_updates <= 0:
        return lr_max
    if warmup_updates < 1:
        warmup_updates = 1
    if update_idx < warmup_updates:
        return lr_max * float(update_idx + 1) / float(warmup_updates)

    # cosine decay to 0
    if total_updates == warmup_updates:
        return lr_max
    t = (update_idx - warmup_updates) / float(total_updates - warmup_updates)
    t = min(max(t, 0.0), 1.0)
    return lr_max * 0.5 * (1.0 + math.cos(math.pi * t))


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--target_model", type=str, required=True)
    p.add_argument("--data_jsonl", type=str, required=True)
    p.add_argument("--output_dir", type=str, required=True)

    p.add_argument("--epoch", type=int, default=1)
    p.add_argument("--min_length", type=int, default=128)
    p.add_argument("--max_length", type=int, default=2048)

    p.add_argument("--micro_batch_size", type=int, default=1)
    p.add_argument("--grad_accum", type=int, default=8)

    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--warmup_updates", type=int, default=1000)
    p.add_argument("--save_every_updates", type=int, default=1000)

    p.add_argument("--hidden_layer", type=int, default=-3)
    p.add_argument("--num_layers", type=int, default=1)
    p.add_argument("--block_len_min", type=int, default=5)
    p.add_argument("--block_len_max", type=int, default=10)

    p.add_argument("--bf16", action="store_true")
    p.add_argument("--seed", type=int, default=888)
    p.add_argument("--num_workers", type=int, default=0)

    p.add_argument("--device", type=str, default="cuda")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    set_seed(args.seed)

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA not available, but --device cuda was set")

    dtype = torch.bfloat16 if args.bf16 else torch.float16

    # Tokenizer (generic)
    tok = AutoTokenizer.from_pretrained(args.target_model, trust_remote_code=True, use_fast=True)
    if tok.pad_token_id is None:
        # Common for decoder-only LMs
        tok.pad_token = tok.eos_token
    # Match Faster-SD setting (left pad). For batch_size=1 doesn't matter much.
    tok.padding_side = "left"

    # Teacher
    teacher = load_teacher(args.target_model, dtype=dtype, device=device, trust_remote_code=True)

    # Draft (Effective) — init aligned with paper train.py (copy embedding + lm_head from teacher)
    if hasattr(Effective_Draft_Decoder, "from_teacher"):
        draft = Effective_Draft_Decoder.from_teacher(
            teacher,
            num_layers=args.num_layers,
            block_len_min=args.block_len_min,
            block_len_max=args.block_len_max,
        )
    else:
        cfg = teacher.config
        # Paper signature: Effective_Draft_Decoder(hidden_size, dim_feedforward, head_num, num_layers, config)
        try:
            draft = Effective_Draft_Decoder(
                cfg.hidden_size,
                cfg.hidden_size * 2,
                getattr(cfg, "num_attention_heads", getattr(cfg, "num_key_value_heads", 32)),
                args.num_layers,
                cfg,
            )
        except Exception:
            # Fallback for wrappers with different __init__
            draft = Effective_Draft_Decoder(cfg)

        # Copy LM head weights (paper does this)
        try:
            draft.lm_head.load_state_dict(teacher.lm_head.state_dict())
        except Exception:
            pass

        # Copy embedding weights (paper does this)
        try:
            emb = teacher.get_input_embeddings()
            if hasattr(draft, "embedding_layer"):
                draft.embedding_layer.load_state_dict(emb.state_dict())
            elif hasattr(draft, "embed_tokens"):
                draft.embed_tokens.load_state_dict(emb.state_dict())
            else:
                draft.get_input_embeddings().load_state_dict(emb.state_dict())
        except Exception:
            pass

        # If draft supports configuring block_len range, set it
        if hasattr(draft, "block_len_min"):
            try:
                draft.block_len_min = args.block_len_min
            except Exception:
                pass
        if hasattr(draft, "block_len_max"):
            try:
                draft.block_len_max = args.block_len_max
            except Exception:
                pass

    draft = draft.to(device=device, dtype=dtype)
    draft.train()

    # Dataset + loader
    ds = PromptPredDataset(args.data_jsonl, tok, min_length=args.min_length, max_length=args.max_length)
    print(f"Loaded {len(ds)} samples after filtering")

    loader = DataLoader(
        ds,
        batch_size=args.micro_batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=lambda b: collate_batch(b, pad_id=int(tok.pad_token_id)),
        pin_memory=(device.type == "cuda"),
    )

    # Optimizer
    optim = torch.optim.AdamW(filter(lambda p: p.requires_grad, draft.parameters()), lr=args.lr)

    total_micro = args.epoch * len(loader)
    total_updates = math.ceil(total_micro / args.grad_accum)

    print(f"total_micro_steps={total_micro} total_updates={total_updates} warmup_updates={args.warmup_updates}")

    optim.zero_grad(set_to_none=True)
    micro_step = 0
    update_step = 0

    t0_wall = time.time()
    run_kl = 0.0

    for ep in range(args.epoch):
        for batch in loader:
            micro_step += 1
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            attn_mask = batch["attention_mask"].to(device, non_blocking=True)
            pred_ids = batch["labels"].to(device, non_blocking=True)

            # pred_len is actual (unpadded) pred length per sample.
            # Because we pad labels, compute pred_len by trimming pad_id on the right.
            pad_id = int(tok.pad_token_id)
            # (B, Pmax) -> lengths
            with torch.no_grad():
                pred_lens = (pred_ids != pad_id).sum(dim=1)  # (B,)
                # for now we require uniform pred_len within batch (keep micro_batch=1 for safety)
                if pred_lens.min().item() != pred_lens.max().item():
                    raise ValueError("Different pred lengths in same batch; set --micro_batch_size 1.")
                pred_len = int(pred_lens[0].item())

            # Trim to true lengths (avoid teacher/draft seeing pad)
            input_ids = input_ids[:, : int(attn_mask.sum(dim=1).max().item())]
            attn_mask = attn_mask[:, : input_ids.shape[1]]
            pred_len = min(pred_len, input_ids.shape[1])
            if pred_len <= 1:
                continue

            # Teacher forward: we only need logits + hidden_states
            with torch.no_grad():
                out = teacher(
                    input_ids=input_ids,
                    attention_mask=attn_mask,
                    output_hidden_states=True,
                    use_cache=False,
                    return_dict=True,
                )
                enc = out.hidden_states[args.hidden_layer]  # (B, seq_len, H)
                llm_logits = out.logits[:, -pred_len:, :]   # (B, pred_len, V)

            # Draft loss (KL)
            _, kl_loss = draft(enc, input_ids, llm_logits=llm_logits)
            loss = kl_loss / args.grad_accum
            loss.backward()
            run_kl += kl_loss.item()

            do_update = (micro_step % args.grad_accum == 0) or (micro_step == total_micro)
            if do_update:
                # LR schedule (update-based)
                lr_now = lr_schedule_cosine_with_warmup(update_step, total_updates, args.warmup_updates, args.lr)
                set_lr(optim, lr_now)

                torch.nn.utils.clip_grad_norm_(draft.parameters(), 1.0)
                optim.step()
                optim.zero_grad(set_to_none=True)
                update_step += 1

                if update_step % 10 == 0 or update_step <= 3:
                    wall = time.time() - t0_wall
                    avg_kl = run_kl / float(max(1, (10 if update_step % 10 == 0 else update_step)))
                    print(
                        f"[update] ep={ep} upd={update_step}/{total_updates} "
                        f"micro={micro_step}/{total_micro} lr={lr_now:.2e} "
                        f"kl={avg_kl:.4f} wall={wall:.1f}s"
                    )
                    run_kl = 0.0

                if args.save_every_updates > 0 and (update_step % args.save_every_updates == 0):
                    ckpt_path = os.path.join(args.output_dir, f"draft_decoder_{update_step}.pt")
                    torch.save(draft.state_dict(), ckpt_path)
                    print(f"Saved: {ckpt_path}")

    # final save
    final_path = os.path.join(args.output_dir, "draft_decoder_final.pt")
    torch.save(draft.state_dict(), final_path)
    tok.save_pretrained(args.output_dir)
    print(f"Saved final: {final_path}")
    print(f"Tokenizer saved to: {args.output_dir}")


if __name__ == "__main__":
    main()