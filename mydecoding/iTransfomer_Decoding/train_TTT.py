import argparse
import gc
import os
import random
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from accelerate import Accelerator
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import AutoConfig, AutoTokenizer

from model import Effective_Draft_Decoder, chunk_kl_batchmean
from modeling_llama import LlamaForCausalLM

os.environ["NCCL_P2P_DISABLE"] = "1"
os.environ["NCCL_IB_DISABLE"] = "1"
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

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


def choose_pred_len(pred_len: int) -> int:
    if pred_len > 0:
        return pred_len
    return random.randint(5, 10)


def build_self_attn_mask(
    seq_len: int,
    encoder_len: int,
    inp_len: int,
    pred_len: int,
    dtype: torch.dtype,
    device: torch.device,
):
    mask_min = torch.finfo(dtype).min
    causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1).bool()
    attention_mask = torch.zeros((seq_len, seq_len), dtype=dtype, device=device)
    attention_mask[causal_mask] = mask_min

    for start in range(encoder_len, seq_len, pred_len):
        left = inp_len + start - encoder_len
        attention_mask[start : start + pred_len, left:start] = mask_min
    return attention_mask


def forward_draft_logits_with_static_memory(
    draft_decoder: Effective_Draft_Decoder,
    encoder_self: torch.Tensor,
    memory: torch.Tensor,
    suffix_input_ids: torch.Tensor,
    inp_len: int,
    pred_len: int,
):
    if inp_len <= 0:
        raise ValueError("inp_len must be > 0 to align next-token logits for suffix[0].")
    label_len = suffix_input_ids.shape[1]
    input_embeds = draft_decoder.embedding_layer(suffix_input_ids)
    hidden_states = torch.cat([encoder_self, input_embeds], dim=1)

    position_ids = torch.arange(encoder_self.shape[1], dtype=torch.long, device=hidden_states.device)
    position_ids = torch.cat(
        [
            position_ids,
            torch.arange(inp_len, inp_len + label_len, dtype=torch.long, device=hidden_states.device),
        ],
        dim=0,
    )[None, :]

    attn_mask = build_self_attn_mask(
        seq_len=hidden_states.shape[1],
        encoder_len=encoder_self.shape[1],
        inp_len=inp_len,
        pred_len=pred_len,
        dtype=hidden_states.dtype,
        device=hidden_states.device,
    )

    hidden_states = draft_decoder.decoder(
        hidden_states,
        attention_mask=attn_mask[None, None, :, :],
        position_ids=position_ids,
    )[0]

    # Align with slot_level_llama2_7B:
    # use the appended pred embeddings as query states.
    token_states = hidden_states[:, -label_len:, :]
    if memory.shape[1] > 0:
        mem_len = memory.shape[1]
        i = torch.arange(label_len, device=hidden_states.device).unsqueeze(1)
        j = torch.arange(mem_len, device=hidden_states.device).unsqueeze(0)
        visible_slots = (inp_len + i) // draft_decoder.slot_size
        mem_mask = j >= visible_slots

        cross_out, _ = draft_decoder.cross_attn(
            draft_decoder.cross_norm(token_states),
            memory,
            memory,
            attn_mask=mem_mask,
            need_weights=False,
        )
        token_states = draft_decoder.norm(token_states + cross_out)
    else:
        token_states = draft_decoder.norm(token_states)

    return draft_decoder.lm_head(token_states)


def kl_from_logits(
    draft_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    token_mask: Optional[torch.Tensor] = None,
    chunk_size: int = 4096,
):
    draft_step = draft_logits
    teacher_step = teacher_logits.detach()
    if token_mask is not None:
        token_mask = token_mask.to(draft_step.device, dtype=torch.bool)
        draft_step = draft_step[:, token_mask, :]
        teacher_step = teacher_step[:, token_mask, :]
    if draft_step.numel() == 0:
        return draft_logits.new_zeros(())
    draft_2d = draft_step.reshape(-1, draft_step.shape[-1]).contiguous()
    teacher_2d = teacher_step.reshape(-1, teacher_step.shape[-1]).contiguous()
    current_chunk = max(int(chunk_size), 1)
    while True:
        try:
            return chunk_kl_batchmean(draft_2d, teacher_2d, chunk_size=current_chunk)
        except torch.OutOfMemoryError:
            if current_chunk == 1:
                raise
            next_chunk = max(current_chunk // 2, 1)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            print(f"[WARN] KL OOM at chunk_size={current_chunk}; retry with chunk_size={next_chunk}.")
            current_chunk = next_chunk


def ce_from_logits_masked(
    draft_logits: torch.Tensor,
    suffix_gold: torch.Tensor,
    token_mask: Optional[torch.Tensor] = None,
):
    if suffix_gold.shape[1] <= 0:
        return draft_logits.new_zeros(())
    logits = draft_logits
    labels = suffix_gold
    if token_mask is not None:
        token_mask = token_mask.to(logits.device, dtype=torch.bool)
        logits = logits[:, token_mask, :]
        labels = labels[:, token_mask]
    if logits.numel() == 0:
        return draft_logits.new_zeros(())
    return F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]).float(),
        labels.reshape(-1).contiguous(),
    )


def kl_from_next_token_logits(
    draft_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    token_mask: Optional[torch.Tensor] = None,
    chunk_size: int = 4096,
):
    if draft_logits.shape[1] <= 1 or teacher_logits.shape[1] <= 1:
        return draft_logits.new_zeros(())
    return kl_from_logits(
        draft_logits[:, :-1, :],
        teacher_logits[:, :-1, :],
        token_mask=token_mask,
        chunk_size=chunk_size,
    )


def ce_from_next_token_logits(
    draft_logits: torch.Tensor,
    suffix_gold: torch.Tensor,
    token_mask: Optional[torch.Tensor] = None,
):
    if suffix_gold.shape[1] <= 1:
        return draft_logits.new_zeros(())
    return ce_from_logits_masked(
        draft_logits[:, :-1, :],
        suffix_gold[:, 1:],
        token_mask=token_mask,
    )


def build_step0_pred_suffix(suffix_gold: torch.Tensor, step0_logits: torch.Tensor):
    pred_suffix = suffix_gold.clone()
    if suffix_gold.shape[1] <= 1:
        return pred_suffix
    # step0_logits[:, t] predicts suffix token at position t+1.
    pred_next = step0_logits.argmax(dim=-1)
    pred_suffix[:, 1:] = pred_next[:, :-1]
    return pred_suffix


def replace_block_second_span_with_pred(
    suffix_gold: torch.Tensor,
    pred_suffix: torch.Tensor,
    block_len: int,
    replace_width: int,
):
    polluted = suffix_gold.clone()
    label_len = suffix_gold.shape[1]
    if label_len <= 0:
        return polluted

    block_len = max(int(block_len), 1)
    replace_width = int(replace_width)
    if replace_width <= 0:
        return polluted

    # Replace positions starting from the 2nd token in each block.
    start = 1
    end = start + replace_width
    token_pos = torch.arange(label_len, device=suffix_gold.device)
    in_block = token_pos % block_len
    replace_mask = (in_block >= start) & (in_block < end)
    if replace_mask.any():
        polluted[:, replace_mask] = pred_suffix[:, replace_mask]
    return polluted


def build_blockwise_tail_mask(label_len: int, block_len: int, tail_start: int, device: torch.device):
    if label_len <= 0:
        return torch.zeros((0,), device=device, dtype=torch.bool)
    block_len = max(int(block_len), 1)
    token_pos = torch.arange(label_len, device=device)
    # token_pos indexes next-token logits (k predicts suffix token k+1),
    # so block position must be computed on (k+1).
    in_block = (token_pos + 1) % block_len
    # Keep only tail targets in every block: [tail_start, block_len-1].
    return in_block >= int(tail_start)


def ttt_kl_loss(
    draft_decoder: Effective_Draft_Decoder,
    encoder_out: torch.Tensor,
    suffix_gold: torch.Tensor,
    teacher_logits: torch.Tensor,
    inp_len: int,
    pred_len: int,
    ttt_steps: int,
    ttt_lambda1: float,
    ttt_lambda2: float,
    kl_chunk_size: int,
):
    # Fast path: build encoder/memory once and backprop once on summed KL.
    encoder_self, memory = draft_decoder._split_encoder_out(
        encoder_out,
        slot_size=draft_decoder.slot_size,
        apply_dropout=draft_decoder.training,
    )
    step0_logits = forward_draft_logits_with_static_memory(
        draft_decoder=draft_decoder,
        encoder_self=encoder_self,
        memory=memory,
        suffix_input_ids=suffix_gold,
        inp_len=inp_len,
        pred_len=pred_len,
    )

    kl0 = kl_from_next_token_logits(step0_logits, teacher_logits, token_mask=None, chunk_size=kl_chunk_size)
    with torch.no_grad():
        ce0 = ce_from_next_token_logits(step0_logits, suffix_gold, token_mask=None)
    total_kl = kl0

    zero = kl0.new_zeros(())
    kl1 = zero
    kl2 = zero

    if ttt_steps <= 0:
        return total_kl, kl0, kl1, kl2, ce0

    label_len = suffix_gold.shape[1]
    with torch.no_grad():
        pred_suffix = build_step0_pred_suffix(suffix_gold, step0_logits)
    del step0_logits

    if ttt_steps >= 1 and ttt_lambda1 > 0:
        polluted1 = replace_block_second_span_with_pred(
            suffix_gold=suffix_gold,
            pred_suffix=pred_suffix,
            block_len=pred_len,
            replace_width=max(pred_len - 1, 0),
        )
        step1_logits = forward_draft_logits_with_static_memory(
            draft_decoder=draft_decoder,
            encoder_self=encoder_self,
            memory=memory,
            suffix_input_ids=polluted1,
            inp_len=inp_len,
            pred_len=pred_len,
        )
        mask1 = build_blockwise_tail_mask(
            label_len=max(label_len - 1, 0),
            block_len=pred_len,
            tail_start=1,
            device=step1_logits.device,
        )
        kl1 = kl_from_next_token_logits(step1_logits, teacher_logits, token_mask=mask1, chunk_size=kl_chunk_size)
        total_kl = total_kl + ttt_lambda1 * kl1

    if ttt_steps >= 2 and ttt_lambda2 > 0:
        polluted2 = replace_block_second_span_with_pred(
            suffix_gold=suffix_gold,
            pred_suffix=pred_suffix,
            block_len=pred_len,
            replace_width=2,
        )
        step2_logits = forward_draft_logits_with_static_memory(
            draft_decoder=draft_decoder,
            encoder_self=encoder_self,
            memory=memory,
            suffix_input_ids=polluted2,
            inp_len=inp_len,
            pred_len=pred_len,
        )
        mask2 = build_blockwise_tail_mask(
            label_len=max(label_len - 1, 0),
            block_len=pred_len,
            tail_start=2,
            device=step2_logits.device,
        )
        kl2 = kl_from_next_token_logits(step2_logits, teacher_logits, token_mask=mask2, chunk_size=kl_chunk_size)
        total_kl = total_kl + ttt_lambda2 * kl2

    return total_kl, kl0, kl1, kl2, ce0


parser = argparse.ArgumentParser(description="Train EDD draft model with token-level TTT")
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
parser.add_argument("--dataloader_num_workers", type=int, default=4, help="Num workers for DataLoader")
parser.add_argument(
    "--pin_memory",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="Use pinned host memory for faster H2D transfer",
)

parser.add_argument("--slot_size", type=int, default=5, help="Slot size for memory compression")
parser.add_argument("--layer_dropout", type=float, default=0.0, help="Layer-axis dropout prob in fusion")
parser.add_argument("--slot_dropout", type=float, default=0.0, help="Slot-axis dropout prob in fusion")
parser.add_argument("--pred_len", type=int, default=0, help="Draft self-attn block length; <=0 uses random 5~10")
parser.add_argument("--kl_chunk_size", type=int, default=2048, help="Chunk size for token-level KL over vocab axis")

parser.add_argument("--ttt_steps", type=int, default=1, choices=[0, 1, 2], help="Number of token-level TTT simulated steps")
parser.add_argument(
    "--ttt_k",
    type=int,
    default=5,
    help="Deprecated (ignored in blockwise-TTT mode). Step-1/2 use replace_per_block=1/2.",
)
parser.add_argument("--ttt_lambda1", type=float, default=0.5, help="Weight for Step-1 KL")
parser.add_argument("--ttt_lambda2", type=float, default=0.0, help="Weight for Step-2 KL")
parser.add_argument(
    "--ttt_step1_warmup_steps",
    type=int,
    default=0,
    help="Enable Step-1/2 only after this many optimizer steps (Step-0 always on).",
)
parser.add_argument(
    "--ttt_align_to_slot",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="Deprecated (ignored in blockwise-TTT mode).",
)
args = parser.parse_args()
if args.ttt_k != 5:
    print("[WARN] --ttt_k is deprecated and ignored in blockwise-TTT mode.")
if args.ttt_align_to_slot is not True:
    print("[WARN] --ttt_align_to_slot is deprecated and ignored in blockwise-TTT mode.")

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

del dataset_train
gc.collect()

print(train_tokenized_datasets)
print(tokenizer.decode(train_tokenized_datasets[0]["input_ids"]))
print(tokenizer.decode(train_tokenized_datasets[0]["labels"]))

train_dataloader = DataLoader(
    train_tokenized_datasets,
    batch_size=batch_size,
    shuffle=True,
    num_workers=max(int(args.dataloader_num_workers), 0),
    pin_memory=bool(args.pin_memory),
    persistent_workers=bool(args.dataloader_num_workers and args.dataloader_num_workers > 0),
)

model = LlamaForCausalLM.from_pretrained(model_checkpoint, torch_dtype=torch_dtype)
for p in model.parameters():
    p.requires_grad_(False)

config = AutoConfig.from_pretrained(model_checkpoint)
draft_decoder = Effective_Draft_Decoder(
    config.hidden_size,
    config.hidden_size * 2,
    config.num_attention_heads,
    num_layers,
    config,
    layer_dropout=args.layer_dropout,
    slot_dropout=args.slot_dropout,
    slot_size=args.slot_size,
)
draft_decoder.lm_head.load_state_dict(model.lm_head.state_dict())
draft_decoder.embedding_layer.load_state_dict(model.model.embed_tokens.state_dict())
draft_decoder = draft_decoder.to(dtype=torch_dtype)

total_iters = max(len(train_dataloader) * epoch, 1)
optimizer = torch.optim.AdamW(
    filter(lambda p: p.requires_grad, draft_decoder.parameters()),
    lr=lr_max,
)
cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    max(total_iters - warm_up_iter, 1),
)


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
        label_len = int(batch["labels"].shape[1])
        labels[:, :-label_len] = -100

        with torch.no_grad():
            outputs = model(input_ids, output_hidden_states=True, labels=labels)

        if layer_indices is None:
            enc_out = outputs.hidden_states[hidden_layer].detach()
        else:
            enc_out = torch.stack([outputs.hidden_states[i] for i in layer_indices], dim=1).detach()

        inp_len = input_ids.shape[1] - label_len
        if inp_len <= 0:
            del outputs
            continue
        # Align with slot_level_llama2_7B: teacher logits use pred segment states.
        teacher_logits = outputs.logits[:, inp_len : inp_len + label_len, :].detach().contiguous()
        llm_loss_value = float(outputs.loss.detach())
        del outputs
        suffix_gold = input_ids[:, inp_len:]
        pred_len = choose_pred_len(args.pred_len)
        effective_ttt_steps = args.ttt_steps if opt_steps >= args.ttt_step1_warmup_steps else 0

        total_kl, kl0, kl1, kl2, ce0 = ttt_kl_loss(
            draft_decoder=draft_decoder,
            encoder_out=enc_out,
            suffix_gold=suffix_gold,
            teacher_logits=teacher_logits,
            inp_len=inp_len,
            pred_len=pred_len,
            ttt_steps=effective_ttt_steps,
            ttt_lambda1=args.ttt_lambda1,
            ttt_lambda2=args.ttt_lambda2,
            kl_chunk_size=args.kl_chunk_size,
        )

        progress_bar.update(1)
        loss = total_kl / gradient_accumulation_steps
        accelerator.backward(loss)

        if (step + 1) % gradient_accumulation_steps == 0:
            opt_steps += 1
            optimizer.step()
            if opt_steps < warm_up_iter:
                warmup_scheduler.step()
            else:
                cosine_scheduler.step()
            optimizer.zero_grad(set_to_none=True)

            if accelerator.is_main_process and opt_steps % 20 == 0:
                progress_bar.set_postfix(
                    kl=f"{total_kl.item():.4f}",
                    kl0=f"{kl0.item():.4f}",
                    kl1=f"{kl1.item():.4f}",
                    kl2=f"{kl2.item():.4f}",
                    ttt=effective_ttt_steps,
                    ce=f"{ce0.item():.4f}",
                    llm=f"{llm_loss_value:.4f}",
                )

            if opt_steps % save_step == 0:
                print(
                    f"loss: {loss}, total_kl: {total_kl}, kl0: {kl0}, kl1: {kl1}, kl2: {kl2}, "
                    f"ce0: {ce0}, llm_loss: {llm_loss_value}, "
                    f"p: {(opt_steps * gradient_accumulation_steps / len(train_dataloader) * epoch)}"
                )
                accelerator.wait_for_everyone()
                unwrapped_model = accelerator.unwrap_model(draft_decoder)
                torch.save(unwrapped_model.state_dict(), f"./draft_model_ttt_{opt_steps}.pt")

accelerator.wait_for_everyone()
if accelerator.is_main_process:
    os.makedirs("./checkpoints", exist_ok=True)
    unwrapped = accelerator.unwrap_model(draft_decoder)
    last_path = f"./checkpoints/draft_model_ttt_last_step{opt_steps}.pt"
    accelerator.save(unwrapped.state_dict(), last_path)
    print(f"[Saved] final draft model -> {last_path}")
