import argparse
import os
import random

import numpy as np
import torch
from transformers import AutoConfig, AutoTokenizer

from modeling_llama import LlamaForCausalLM
from model import Effective_Draft_Decoder

os.environ["NCCL_P2P_DISABLE"] = "1"
os.environ["NCCL_IB_DISABLE"] = "1"

seed_val = 888
np.random.seed(seed_val)
torch.manual_seed(seed_val)
torch.cuda.manual_seed_all(seed_val)
random.seed(seed_val)


def _parse_layers(s: str):
    if s is None:
        return None
    s = s.strip()
    if not s:
        return None
    s = s.replace("[", "").replace("]", "")
    layers = [int(x) for x in s.split(",") if x.strip()]
    return layers if layers else None


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


def _select_encoder_out(outputs, hidden_layer: int, layer_indices):
    if layer_indices is None:
        return outputs.hidden_states[hidden_layer]
    return torch.stack([outputs.hidden_states[i] for i in layer_indices], dim=1)


@torch.no_grad()
def cuda_bench_ms(fn, iters=200, warmup=20):
    """Measure average runtime (ms) with CUDA events."""
    starter = torch.cuda.Event(enable_timing=True)
    ender = torch.cuda.Event(enable_timing=True)

    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    times = []
    for _ in range(iters):
        starter.record()
        fn()
        ender.record()
        torch.cuda.synchronize()
        times.append(starter.elapsed_time(ender))
    return sum(times) / len(times)


@torch.no_grad()
def build_single_token_context(
    prompt: str,
    llm_model,
    draft_model,
    tokenizer,
    hidden_layer=-4,
    layer_indices=None,
):
    tok = tokenizer(prompt, return_tensors="pt")
    input_ids = tok["input_ids"].cuda()
    attention_mask = tok["attention_mask"].cuda()

    out = llm_model(input_ids, attention_mask=attention_mask, output_hidden_states=True, use_cache=True)
    past_key_values = out.past_key_values
    new_token_1 = out.logits[:, -1:, :].argmax(dim=-1)

    encoder_out = _select_encoder_out(out, hidden_layer, layer_indices)
    slot_cache = draft_model.init_slot_cache(
        encoder_out,
        slot_size=draft_model.slot_size,
        apply_dropout=False,
    )

    encoder_self, memory = draft_model.split_from_slot_cache(slot_cache)

    return {
        "attention_mask": attention_mask,
        "new_token_1": new_token_1,
        "past_key_values": past_key_values,
        "encoder_self": encoder_self,
        "memory": memory,
    }


@torch.no_grad()
def calibrate_sd_st_threshold(
    prompt: str,
    llm_model,
    draft_model,
    tokenizer,
    hidden_layer=-4,
    layer_indices=None,
    iters=200,
    warmup=20,
):
    ctx = build_single_token_context(
        prompt,
        llm_model,
        draft_model,
        tokenizer,
        hidden_layer=hidden_layer,
        layer_indices=layer_indices,
    )

    attention_mask = ctx["attention_mask"]
    new_token_1 = ctx["new_token_1"]
    past_key_values = ctx["past_key_values"]
    encoder_self = ctx["encoder_self"]
    memory = ctx["memory"]

    one_mask = torch.ones(1, 1, device="cuda", dtype=attention_mask.dtype)
    attn_mask_one = torch.cat([attention_mask, one_mask], dim=1)

    def run_target_verify():
        llm_model(
            new_token_1,
            attention_mask=attn_mask_one,
            past_key_values=past_key_values,
            output_hidden_states=False,
            use_cache=False,
        )

    def run_draft_one_step():
        dec_inp = torch.cat([encoder_self, draft_model.embedding_layer(new_token_1)], dim=1)
        hidden_states = draft_model.decoder(dec_inp, use_cache=False)[0]
        token_state = hidden_states[:, -1:, :]
        if memory.shape[1] > 0:
            cross_out, _ = draft_model.cross_attn(
                draft_model.cross_norm(token_state),
                memory,
                memory,
                attn_mask=None,
                need_weights=False,
            )
            token_state = draft_model.norm(token_state + cross_out)
        else:
            token_state = draft_model.norm(token_state)
        _ = draft_model.lm_head(token_state)

    st = cuda_bench_ms(run_target_verify, iters=iters, warmup=warmup)
    sd = cuda_bench_ms(run_draft_one_step, iters=iters, warmup=warmup)
    return sd, st, sd / st


def main():
    parser = argparse.ArgumentParser("Calibrate SD verify threshold by latency ratio")
    parser.add_argument("--model_checkpoint", type=str, required=True)
    parser.add_argument("--draft_model_checkpoint", type=str, required=True)
    parser.add_argument("--prompt", type=str, default="Write a Python function that returns the factorial of n.")
    parser.add_argument("--hidden_layer", type=int, default=-4)
    parser.add_argument("--fusion_layers", type=str, default="")
    parser.add_argument("--top_k", type=int, default=5, help="Unused in strict single-token mode.")
    parser.add_argument("--tree_threshold", type=float, default=0.036, help="Unused in strict single-token mode.")
    parser.add_argument("--iters", type=int, default=200)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--dtype", type=str, default="bf16", choices=["bf16", "fp16", "fp32"])
    parser.add_argument("--tokenizer_dir", type=str, default="")
    parser.add_argument("--slot_size", type=int, default=5)
    args = parser.parse_args()

    tokenizer = _load_tokenizer(args.model_checkpoint, args.tokenizer_dir.strip())
    torch_dtype = _resolve_dtype(args.dtype)
    layer_indices = _parse_layers(args.fusion_layers)

    llm = LlamaForCausalLM.from_pretrained(
        args.model_checkpoint,
        torch_dtype=torch_dtype,
        device_map="auto",
    ).cuda().eval()

    cfg = AutoConfig.from_pretrained(args.model_checkpoint)
    draft = Effective_Draft_Decoder(
        cfg.hidden_size,
        cfg.hidden_size * 2,
        cfg.num_attention_heads,
        1,
        cfg,
        slot_size=args.slot_size,
    )
    draft.lm_head.load_state_dict(llm.lm_head.state_dict())
    draft.embedding_layer.load_state_dict(llm.model.embed_tokens.state_dict())
    draft = draft.to(dtype=torch_dtype)

    sd_state = torch.load(args.draft_model_checkpoint, map_location="cpu")
    draft.load_state_dict(sd_state, strict=True)
    draft = draft.cuda().eval()

    sd, st, thr = calibrate_sd_st_threshold(
        prompt=args.prompt,
        llm_model=llm,
        draft_model=draft,
        tokenizer=tokenizer,
        hidden_layer=args.hidden_layer,
        layer_indices=layer_indices,
        iters=args.iters,
        warmup=args.warmup,
    )

    print(f"sd (draft strict-1-token) = {sd:.4f} ms")
    print(f"st (target strict-1-token)= {st:.4f} ms")
    print(f"recommended threshold (sd/st) = {thr:.6f}")
    print("note: --top_k/--tree_threshold are ignored in this strict single-token benchmark.")


if __name__ == "__main__":
    main()
