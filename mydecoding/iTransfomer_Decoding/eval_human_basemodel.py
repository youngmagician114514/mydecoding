import argparse
import os
import random
import time

import jsonlines
import numpy as np
import torch
from datasets import Dataset
from tqdm import tqdm
from transformers import AutoTokenizer

from modeling_llama import LlamaForCausalLM

os.environ["NCCL_P2P_DISABLE"] = "1"
os.environ["NCCL_IB_DISABLE"] = "1"

seed_val = 888
np.random.seed(seed_val)
torch.manual_seed(seed_val)
torch.cuda.manual_seed_all(seed_val)
random.seed(seed_val)


def _resolve_dtype(name: str):
    if name == "bf16":
        return torch.bfloat16
    if name == "fp16":
        return torch.float16
    return torch.float32


def _load_tokenizer(model_ckpt: str, tok_dir: str):
    tokenizer_kwargs = {"use_fast": True, "padding_side": "left", "trust_remote_code": True}
    if tok_dir and os.path.isdir(tok_dir):
        tokenizer = AutoTokenizer.from_pretrained(tok_dir, **tokenizer_kwargs)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_ckpt, **tokenizer_kwargs)
        if tok_dir:
            os.makedirs(tok_dir, exist_ok=True)
            tokenizer.save_pretrained(tok_dir)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    return tokenizer


def load_data(tokenizer: AutoTokenizer, data_path: str, num_proc: int):
    def preprocess_function(examples):
        tokenizer.padding_side = "left"
        prompt = tokenizer(examples["prompt"])
        tokenizer.padding_side = "right"
        pred = tokenizer(examples["pred"], add_special_tokens=False)
        return {
            "input_ids": prompt["input_ids"],
            "input_attention_mask": prompt["attention_mask"],
            "labels_attention_mask": pred["attention_mask"],
        }

    test_data = []
    with jsonlines.open(data_path) as f:
        for line in f:
            test_data.append({"prompt": line["prompt"], "pred": line["pred"]})

    dataset_test = Dataset.from_list(test_data)
    print(dataset_test)
    tokenized = dataset_test.map(
        preprocess_function,
        batched=True,
        num_proc=num_proc,
        remove_columns=["prompt", "pred"],
    )
    tokenized.set_format("torch")
    return tokenized


@torch.no_grad()
def eval_basemodel_speed(batch, llm_model, eos_id: int, max_examples: int = -1, use_kv_cache: bool = True):
    total_tokens = 0
    decode_tokens = 0
    target_tokens = 0
    eos_stop = 0
    decode_time_s = 0.0

    n = len(batch) if max_examples <= 0 else min(len(batch), max_examples)

    torch.cuda.synchronize()
    t0 = time.time()

    for i in tqdm(range(n), desc="Testing Progress"):
        input_ids = batch[i]["input_ids"].unsqueeze(0).cuda()
        attn_mask = batch[i]["input_attention_mask"].unsqueeze(0).cuda()
        label_num = int(batch[i]["labels_attention_mask"].sum().item())
        target_tokens += label_num

        if label_num <= 0:
            continue

        if use_kv_cache:
            out = llm_model(
                input_ids,
                attention_mask=attn_mask,
                use_cache=True,
            )
            past_key_values = out.past_key_values
            next_token = out.logits[:, -1:, :].argmax(dim=-1)
            total_tokens += 1
            attn_mask = torch.cat(
                [attn_mask, torch.ones(1, 1, device=attn_mask.device, dtype=attn_mask.dtype)],
                dim=1,
            )

            if int(next_token.item()) == eos_id:
                eos_stop += 1
                continue

            if label_num > 1:
                torch.cuda.synchronize()
                sample_decode_start = time.perf_counter()

            for _ in range(label_num - 1):
                out = llm_model(
                    next_token,
                    attention_mask=attn_mask,
                    past_key_values=past_key_values,
                    use_cache=True,
                )
                past_key_values = out.past_key_values
                next_token = out.logits[:, -1:, :].argmax(dim=-1)

                attn_mask = torch.cat(
                    [attn_mask, torch.ones(1, 1, device=attn_mask.device, dtype=attn_mask.dtype)],
                    dim=1,
                )
                total_tokens += 1
                decode_tokens += 1

                if int(next_token.item()) == eos_id:
                    eos_stop += 1
                    break

            if label_num > 1:
                torch.cuda.synchronize()
                decode_time_s += time.perf_counter() - sample_decode_start
        else:
            gen_ids = input_ids
            gen_mask = attn_mask

            out = llm_model(
                gen_ids,
                attention_mask=gen_mask,
                use_cache=False,
            )
            next_token = out.logits[:, -1:, :].argmax(dim=-1)
            gen_ids = torch.cat([gen_ids, next_token], dim=1)
            gen_mask = torch.cat(
                [gen_mask, torch.ones(1, 1, device=gen_mask.device, dtype=gen_mask.dtype)],
                dim=1,
            )
            total_tokens += 1

            if int(next_token.item()) == eos_id:
                eos_stop += 1
                continue

            if label_num > 1:
                torch.cuda.synchronize()
                sample_decode_start = time.perf_counter()

            for _ in range(label_num - 1):
                out = llm_model(
                    gen_ids,
                    attention_mask=gen_mask,
                    use_cache=False,
                )
                next_token = out.logits[:, -1:, :].argmax(dim=-1)
                gen_ids = torch.cat([gen_ids, next_token], dim=1)
                gen_mask = torch.cat(
                    [gen_mask, torch.ones(1, 1, device=gen_mask.device, dtype=gen_mask.dtype)],
                    dim=1,
                )
                total_tokens += 1
                decode_tokens += 1

                if int(next_token.item()) == eos_id:
                    eos_stop += 1
                    break

            if label_num > 1:
                torch.cuda.synchronize()
                decode_time_s += time.perf_counter() - sample_decode_start

    torch.cuda.synchronize()
    t1 = time.time()
    total_time = t1 - t0
    tps = total_tokens / max(total_time, 1e-6)
    decode_tps = decode_tokens / max(decode_time_s, 1e-6)

    print("\n==== BASEMODEL RESULTS ====")
    print(f"examples          = {n}")
    print(f"target_tokens     = {target_tokens}")
    print(f"generated_tokens  = {total_tokens}")
    print(f"decode_tokens     = {decode_tokens}")
    print(f"eos_stop_examples = {eos_stop}")
    print(f"time(s)           = {total_time:.4f}")
    print(f"tokens/s          = {tps:.4f}")
    print(f"decode_time(s)    = {decode_time_s:.4f}")
    print(f"decode_tokens/s   = {decode_tps:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="HumanEval base-model speed benchmark")
    parser.add_argument("--model_checkpoint", type=str, required=True, help="Target LLM checkpoint")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to humaneval.jsonl (prompt/pred)")
    parser.add_argument("--dtype", type=str, default="bf16", choices=["bf16", "fp16", "fp32"], help="Inference dtype")
    parser.add_argument("--tokenizer_dir", type=str, default="", help="Optional tokenizer cache directory")
    parser.add_argument("--num_proc", type=int, default=1, help="Num workers for datasets.map")
    parser.add_argument("--max_examples", type=int, default=-1, help="Limit evaluated examples; -1 means all")
    parser.add_argument("--no_kv_cache", action="store_true", help="Disable KV cache and decode with full sequence")
    args = parser.parse_args()

    tokenizer = _load_tokenizer(args.model_checkpoint, args.tokenizer_dir.strip())
    torch_dtype = _resolve_dtype(args.dtype)

    model = LlamaForCausalLM.from_pretrained(
        args.model_checkpoint,
        torch_dtype=torch_dtype,
        device_map="auto",
    )
    model = model.cuda().eval()

    test_tokenized_datasets = load_data(tokenizer, args.data_dir, args.num_proc)
    eval_basemodel_speed(
        batch=test_tokenized_datasets,
        llm_model=model,
        eos_id=tokenizer.eos_token_id,
        max_examples=args.max_examples,
        use_kv_cache=(not args.no_kv_cache),
    )
