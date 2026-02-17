import argparse
import gc
import os
import random

import numpy as np
import torch
from accelerate import Accelerator
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import AutoConfig, AutoTokenizer

from model import Effective_Draft_Decoder
from modeling_llama import LlamaForCausalLM

os.environ["NCCL_P2P_DISABLE"] = "1"
os.environ["NCCL_IB_DISABLE"] = "1"
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

seed_val = 888
np.random.seed(seed_val)
torch.manual_seed(seed_val)
torch.cuda.manual_seed_all(seed_val)
random.seed(seed_val)


def parse_layers(text: str):
    if text is None:
        return None
    text = text.strip()
    if not text:
        return None
    text = text.replace("[", "").replace("]", "")
    layers = [int(x) for x in text.split(",") if x.strip()]
    return layers if layers else None


def resolve_dtype(name: str):
    if name == "bf16":
        return torch.bfloat16
    if name == "fp16":
        return torch.float16
    return torch.float32


def load_tokenizer(model_checkpoint: str, tokenizer_dir: str):
    tokenizer_kwargs = {"use_fast": True, "padding_side": "left", "trust_remote_code": True}
    if tokenizer_dir and os.path.isdir(tokenizer_dir):
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir, **tokenizer_kwargs)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, **tokenizer_kwargs)
        if tokenizer_dir:
            os.makedirs(tokenizer_dir, exist_ok=True)
            tokenizer.save_pretrained(tokenizer_dir)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    return tokenizer


parser = argparse.ArgumentParser(description="Train EDD draft model")
parser.add_argument("--min_length", type=int, default=128, help="Minimum sequence length")
parser.add_argument("--max_length", type=int, default=2048, help="Maximum sequence length")
parser.add_argument("--model_checkpoint", type=str, default="./llama-2-7b-chat-hf", help="Target LLM checkpoint")
parser.add_argument("--epoch", type=int, default=1, help="Training epochs")
parser.add_argument("--dir", type=str, default="./data/sharegpt_llama.jsonl", help="Path to training jsonl")
parser.add_argument("--gradient_accumulation_steps", type=int, default=8, help="Gradient accumulation steps")
parser.add_argument("--warm_up_iter", type=int, default=1000, help="Warmup steps")
parser.add_argument("--save_step", type=int, default=5000, help="Save interval in optimizer steps")
parser.add_argument("--hidden_layer", type=int, default=-4, help="Single hidden layer index")
parser.add_argument("--lr_max", type=float, default=1e-4, help="Max learning rate")
parser.add_argument("--num_layers", type=int, default=1, help="Draft decoder layers")
parser.add_argument("--batch_size", type=int, default=1, help="Batch size")
parser.add_argument("--fusion_layers", type=str, default="", help="Layer indices for fusion, e.g. -25,-15,-1")
parser.add_argument("--dtype", type=str, default="bf16", choices=["bf16", "fp16", "fp32"], help="Training dtype")
parser.add_argument("--tokenizer_dir", type=str, default="", help="Optional tokenizer cache directory")
parser.add_argument("--tokenized_cache_file", type=str, default="", help="Optional datasets cache file (.arrow)")
parser.add_argument("--num_proc", type=int, default=1, help="Num workers for datasets.map")
parser.add_argument("--map_batch_size", type=int, default=64, help="Batch size used by datasets.map tokenization")
parser.add_argument("--dataset_cache_dir", type=str, default="", help="Optional HuggingFace datasets cache dir")
args = parser.parse_args()

min_length = args.min_length
max_length = args.max_length
model_checkpoint = args.model_checkpoint
epoch = args.epoch
data_path = args.dir
gradient_accumulation_steps = args.gradient_accumulation_steps
warm_up_iter = args.warm_up_iter
save_step = args.save_step
hidden_layer = args.hidden_layer
lr_max = args.lr_max
num_layers = args.num_layers
batch_size = args.batch_size
layer_indices = parse_layers(args.fusion_layers)
torch_dtype = resolve_dtype(args.dtype)

tokenizer = load_tokenizer(model_checkpoint, args.tokenizer_dir.strip())


def preprocess_function(examples):
    joined_prompt = [p + r for p, r in zip(examples["prompt"], examples["pred"])]
    prompt = tokenizer(joined_prompt, add_special_tokens=False)
    pred = tokenizer(examples["pred"], add_special_tokens=False)
    return {"input_ids": prompt["input_ids"], "labels": pred["input_ids"]}


dataset_cache_dir = args.dataset_cache_dir.strip() or None
dataset_train = load_dataset(
    "json",
    data_files=data_path,
    split="train",
    cache_dir=dataset_cache_dir,
)
map_kwargs = {
    "batched": True,
    "remove_columns": dataset_train.column_names,
    "load_from_cache_file": True,
    "batch_size": args.map_batch_size,
    "writer_batch_size": max(16, min(args.map_batch_size, 128)),
}
if args.num_proc and args.num_proc > 1:
    map_kwargs["num_proc"] = args.num_proc
else:
    print("[INFO] datasets.map uses main process (num_proc disabled).")
tokenized_cache_file = args.tokenized_cache_file.strip()
if tokenized_cache_file:
    cache_dir = os.path.dirname(tokenized_cache_file)
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
    map_kwargs["cache_file_name"] = tokenized_cache_file

train_tokenized_datasets = dataset_train.map(preprocess_function, **map_kwargs)
train_tokenized_datasets = train_tokenized_datasets.filter(
    lambda x: (len(x["labels"]) > 10) and (min_length < len(x["input_ids"]) < max_length)
)
train_tokenized_datasets.set_format("torch")

# Release raw string-heavy objects early to lower host-memory peak.
del dataset_train
gc.collect()

print(train_tokenized_datasets)
print(tokenizer.decode(train_tokenized_datasets[0]["input_ids"]))
print(tokenizer.decode(train_tokenized_datasets[0]["labels"]))

train_dataloader = DataLoader(train_tokenized_datasets, batch_size=batch_size, shuffle=True)

# Load large model objects after dataset tokenization/filtering to reduce host-memory peak.
model = LlamaForCausalLM.from_pretrained(model_checkpoint, torch_dtype=torch_dtype)

config = AutoConfig.from_pretrained(model_checkpoint)
draft_decoder = Effective_Draft_Decoder(
    config.hidden_size,
    config.hidden_size * 2,
    config.num_attention_heads,
    num_layers,
    config,
)
draft_decoder.lm_head.load_state_dict(model.lm_head.state_dict())
draft_decoder.embedding_layer.load_state_dict(model.model.embed_tokens.state_dict())
draft_decoder = draft_decoder.to(dtype=torch_dtype)

total_iters = len(train_dataloader) * epoch
optimizer = torch.optim.AdamW(
    filter(lambda p: p.requires_grad, draft_decoder.parameters()),
    lr=lr_max,
)
cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, total_iters - warm_up_iter)


class WarmUpLR(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer_obj, warmup_iters, base_lr, max_lr, last_epoch=-1):
        self.warmup_iters = warmup_iters
        self.base_lr = base_lr
        self.max_lr = max_lr
        super().__init__(optimizer_obj, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_iters:
            return [
                self.base_lr + (self.max_lr - self.base_lr) * self.last_epoch / self.warmup_iters
                for _ in self.optimizer.param_groups
            ]
        return [self.max_lr for _ in self.optimizer.param_groups]


warmup_scheduler = WarmUpLR(optimizer, warm_up_iter, base_lr=0.0, max_lr=lr_max)

accelerator = Accelerator()
model, train_dataloader, draft_decoder, optimizer, cosine_scheduler, warmup_scheduler = accelerator.prepare(
    model, train_dataloader, draft_decoder, optimizer, cosine_scheduler, warmup_scheduler
)

progress_bar = tqdm(range(total_iters))
opt_steps = 0
model.eval()

for _ in range(epoch):
    for step, batch in enumerate(train_dataloader):
        input_ids = batch["input_ids"].to(accelerator.device)
        labels = input_ids.clone()
        labels[:, :-batch["labels"].shape[1]] = -100

        with torch.no_grad():
            outputs = model(input_ids, output_hidden_states=True, labels=labels)

        if layer_indices is None:
            enc_out = outputs.hidden_states[hidden_layer]
        else:
            enc_out = torch.stack([outputs.hidden_states[i] for i in layer_indices], dim=1)

        _, kl_loss = draft_decoder(
            enc_out,
            input_ids,
            llm_logits=outputs.logits[:, -batch["labels"].shape[1] :, :],
        )
        progress_bar.update(1)

        loss = kl_loss / gradient_accumulation_steps
        accelerator.backward(loss)

        if (step + 1) % gradient_accumulation_steps == 0:
            opt_steps += 1
            optimizer.step()
            if opt_steps < warm_up_iter:
                warmup_scheduler.step()
            else:
                cosine_scheduler.step()
            optimizer.zero_grad()

            if accelerator.is_main_process and opt_steps % 20 == 0:
                progress_bar.set_postfix(kl=f"{kl_loss.item():.4f}", llm=f"{outputs.loss.item():.4f}")

            if opt_steps % save_step == 0:
                print(
                    f"loss: {loss}, kl_loss: {kl_loss}, llm_loss: {outputs.loss}, "
                    f"p: {(opt_steps * gradient_accumulation_steps / len(train_dataloader) * epoch)}"
                )
                accelerator.wait_for_everyone()
                unwrapped_model = accelerator.unwrap_model(draft_decoder)
                torch.save(unwrapped_model.state_dict(), f"./draft_model_{opt_steps}.pt")

accelerator.wait_for_everyone()
if accelerator.is_main_process:
    os.makedirs("./checkpoints", exist_ok=True)
    unwrapped = accelerator.unwrap_model(draft_decoder)
    last_path = f"./checkpoints/draft_model_last_step{opt_steps}.pt"
    accelerator.save(unwrapped.state_dict(), last_path)
    print(f"[Saved] final draft model -> {last_path}")
