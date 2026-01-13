# infer_head1_ar3.py
"""
示例：用 head1 自回归生成 3 步 token（greedy）
- 每一步：先跑 base_model 得到 base_hidden（因为 head1 的输入是 base_hidden）
- 然后用 head1 输出 logits，选 head1 的 argmax 作为 next token
- 将该 token 拼回 input_ids，继续下一步
"""

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from mydecoding.config.dual_decoder import DualDecoderConfig
from mydecoding.models.dual_decoder import DualDecoderModel


def load_model_and_tokenizer(
    ckpt_path: str,
    device: str | None = None,
):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # ⚠️ 配置要和训练时一致（至少 base_model_name 一致）
    config = DualDecoderConfig(
        base_model_name_or_path="Qwen/Qwen2.5-3B",
        num_draft_candidates=3,
        max_speculative_steps=1,

        # 这些字段不重要：dual_decoder.py 里会用 base_model.config.hidden_size 对齐 head1
        draft_hidden_size=1024,
        draft_num_layers=4,

        fusion_hidden_size=1024,
        fusion_num_heads=4,
        fusion_dropout=0.0,

        decoder_num_layers=2,
        decoder_num_heads=8,
        decoder_dropout=0.0,
    )

    tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = DualDecoderModel(config)

    # 你的 ckpt 如果是只保存 head1（推荐），就用 strict=False 或专门 load head1
    state = torch.load(ckpt_path, map_location="cpu")

    if "head1_state_dict" in state:
        # ✅只加载 head1
        missing, unexpected = model.head1.load_state_dict(state["head1_state_dict"], strict=True)
        print("[load head1] done.")
    else:
        # 兼容旧 ckpt：可能直接保存整个 model.state_dict()
        missing, unexpected = model.load_state_dict(state, strict=False)
        print("[load_state_dict] missing keys:", len(missing))
        print("[load_state_dict] unexpected keys:", len(unexpected))

    model.to(device)

    # dtype 对齐到 base_model
    base_dtype = model.base_model.model.embed_tokens.weight.dtype
    model.to(dtype=base_dtype)

    model.eval()
    return model, tokenizer, device


@torch.no_grad()
def generate_head1_autoregressive_3steps(
    model: DualDecoderModel,
    tokenizer: AutoTokenizer,
    prompt: str,
    steps: int = 3,
    temperature: float = 1.0,
    device: str = "cuda",
    show_topk: int = 3,
):
    """
    用 head1 自回归生成 steps 个 token（greedy）
    """
    enc = tokenizer(prompt, return_tensors="pt", add_special_tokens=True)
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)

    # autocast dtype
    use_bf16 = torch.cuda.is_bf16_supported()
    autocast_dtype = torch.bfloat16 if use_bf16 else torch.float16

    print("\n=== Prompt ===")
    print(prompt)

    for step in range(steps):
        # 1) base forward -> base_hidden (head1 的输入依赖 base hidden)
        with torch.autocast(device_type="cuda", dtype=autocast_dtype, enabled=(device == "cuda")):
            base_out = model.base_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                use_cache=False,  # 为了简单展示，不用 cache
                return_dict=True,
            )

        base_hidden = base_out.hidden_states[-1]      # (B, T, H)
        base_logits_last = base_out.logits[:, -1, :]  # (B, V)
        base_probs_last = F.softmax(base_logits_last, dim=-1)  # (B, V)

        # 2) head1 forward -> head1_logits
        with torch.autocast(device_type="cuda", dtype=autocast_dtype, enabled=(device == "cuda")):
            head1_logits, cand_ids, cand_beliefs, _ = model.head1(base_hidden, attention_mask)

        # head1_logits 是 (B, V)，是针对 “下一个 token” 的分布（对应最后一个位置）
        head1_probs = F.softmax(head1_logits / max(float(temperature), 1e-6), dim=-1)  # (B,V)

        # head1 greedy 选 token
        next_token_id = torch.argmax(head1_probs, dim=-1)  # (B,)
        next_id = next_token_id.item()
        next_text = tokenizer.decode([next_id], skip_special_tokens=False)

        # 3) 打印对比信息
        print(f"\n[head1 step {step}] decided token id = {next_id}")
        print(f"  decided token text: {repr(next_text)}")

        # head1 top-k
        top_probs, top_ids = torch.topk(head1_probs[0], k=show_topk)
        print("  head1 top-k (id / text / prob / base_prob):")
        for i in range(show_topk):
            tid = top_ids[i].item()
            txt = tokenizer.decode([tid], skip_special_tokens=False)
            p_head1 = top_probs[i].item()
            p_base = base_probs_last[0, tid].item()
            print(
                f"    head1_top{i+1}: id={tid:<6} text={repr(txt):<10} "
                f"head1={p_head1:.6f} ({p_head1:.2e})  base={p_base:.6f} ({p_base:.2e})"
            )

        # 4) autoregressive append
        next_token = next_token_id.unsqueeze(0)  # (1,1)
        input_ids = torch.cat([input_ids, next_token], dim=1)
        attention_mask = torch.cat([attention_mask, torch.ones_like(next_token)], dim=1)

        if tokenizer.eos_token_id is not None and next_id == tokenizer.eos_token_id:
            print("[stop] EOS generated.")
            break

    # 输出结果
    full_text = tokenizer.decode(input_ids[0], skip_special_tokens=True, clean_up_tokenization_spaces=True)
    new_text = tokenizer.decode(input_ids[0, enc["input_ids"].shape[1]:], skip_special_tokens=True, clean_up_tokenization_spaces=True)

    print("\n=== New text (head1 generated) ===")
    print(new_text)

    print("\n=== Full text ===")
    print(full_text)

    return full_text, new_text


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ⚠️ 改成你的 ckpt
    ckpt_path = "training/checkpoints/stageA_head1_step5000_last.pt"

    model, tokenizer, device = load_model_and_tokenizer(ckpt_path=ckpt_path, device=device)

    while True:
        prompt = input("\n输入 prompt（回车退出）：\n> ").strip()
        if not prompt:
            break

        generate_head1_autoregressive_3steps(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            steps=20,           # ✅生成三步
            temperature=1.0,
            device=device,
            show_topk=3,
        )


if __name__ == "__main__":
    main()
