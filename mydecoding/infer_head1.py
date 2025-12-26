# mydecoding/mydecoding/infer_head1.py
"""
推理逻辑：
- 生成完全由 basemodel(Qwen) 决定（greedy 自回归）。
- head1 只做“旁观者”：每一步展示自己的 top-k 候选 + 置信度，
  以及 basemodel 在同一组 token 上的概率，方便对比。
"""

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from mydecoding.config.dual_decoder import DualDecoderConfig
from mydecoding.models.dual_decoder import DualDecoderModel


def load_model_and_tokenizer(
    ckpt_path: str = "training/checkpoints/stageA_head1_step5000.pt",
    device: str | None = None,
):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # ⚠️ 这里的配置要和你训练时一致
    config = DualDecoderConfig(
        base_model_name_or_path="Qwen/Qwen2.5-3B",
        num_draft_candidates=3,
        max_speculative_steps=1,
        draft_hidden_size=1024,
        draft_num_layers=2,
        fusion_hidden_size=1024,
        fusion_num_heads=4,
        fusion_dropout=0.1,
        decoder_num_layers=2,
        decoder_num_heads=8,
        decoder_dropout=0.1,
        draft_loss_weight=1.0,
        fusion_loss_weight=1.0,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        config.base_model_name_or_path,
        use_fast=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = DualDecoderModel(config)

    state = torch.load(ckpt_path, map_location="cpu")
    missing, unexpected = model.load_state_dict(state, strict=False)
    print("[load_state_dict] missing keys:", len(missing))
    print("[load_state_dict] unexpected keys:", len(unexpected))

    model.to(device)
    base_dtype = model.base_model.model.embed_tokens.weight.dtype
    print("base_model dtype:", base_dtype)
    model.to(dtype=base_dtype)
    model.eval()

    return model, tokenizer, config, device


@torch.no_grad()
def generate_base_with_head1_monitor(
    model: DualDecoderModel,
    tokenizer: AutoTokenizer,
    prompt: str,
    max_new_tokens: int = 64,
    temperature: float = 1.0,
    device: str = "cuda",
    topk_to_show: int = 3,
):
    """
    basemodel greedy 生成；head1 只展示候选 + 置信度。
    """

    enc = tokenizer(
        prompt,
        return_tensors="pt",
        padding=False,
        add_special_tokens=True,
    )
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)
    prompt_len = input_ids.shape[1]

    use_bf16 = torch.cuda.is_bf16_supported()
    autocast_dtype = torch.bfloat16 if use_bf16 else torch.float16

    decided_tokens = []

    for step in range(max_new_tokens):
        # ========== 1. basemodel 前向 ==========
        with torch.autocast(device_type="cuda", dtype=autocast_dtype):
            base_out = model.base_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                use_cache=False,
            )
        base_hidden = base_out.hidden_states[-1]          # (B,T,H)
        base_logits = base_out.logits[:, -1, :]           # (B,V)
        base_probs = F.softmax(
            base_logits / max(float(temperature), 1e-6), dim=-1
        )                                                 # (B,V)

        # basemodel greedy 选 token
        next_token_id = torch.argmax(base_probs, dim=-1)  # (B,)
        decided_tokens.append(next_token_id.item())

        # basemodel 自己的 top-k
        base_top_probs, base_top_ids = torch.topk(base_probs[0], k=topk_to_show, dim=-1)

        # ========== 2. head1 监控 ==========
        with torch.autocast(device_type="cuda", dtype=autocast_dtype):
            head1_logits, cand_ids, cand_beliefs, s1 = model.head1(
                base_hidden, attention_mask
            )

        cand_ids = cand_ids[0]           # (K,)
        cand_beliefs = cand_beliefs[0]   # (K,) head1 自己的概率
        head1_topk = min(topk_to_show, cand_ids.numel())

        # basemodel 在 head1 这些候选上的概率
        base_probs_for_cands = base_probs[0].gather(-1, cand_ids)

        # ========== 3. 打印信息 ==========
        decided_text = tokenizer.decode([next_token_id.item()], skip_special_tokens=False)
        print(f"\n[step {step}] decided token id = {next_token_id.item()}")
        print(f"  decided token text: {repr(decided_text)}")

        # basemodel top-k
        print("  base top-k (id / text / prob):")
        for i in range(topk_to_show):
            tid = base_top_ids[i].item()
            txt = tokenizer.decode([tid], skip_special_tokens=False)
            p = base_top_probs[i].item()
            print(f"    base_top{i+1}: id={tid:<6} text={repr(txt):<8} prob={p:.6f}")

        # head1 候选 + 置信度
        print("  head1 candidates (id / text / head1_prob / base_prob):")
        for rank in range(head1_topk):
            tid = cand_ids[rank].item()
            txt = tokenizer.decode([tid], skip_special_tokens=False)
            p_head1 = cand_beliefs[rank].item()
            p_base = base_probs_for_cands[rank].item()
            # 用 6 位小数，同时附一份科学计数，避免看起来都是 0
            print(
                f"    top{rank+1}: id={tid:<6} "
                f"text={repr(txt):<10} "
                f"head1={p_head1:.6f} ({p_head1:.2e})  "
                f"base={p_base:.6f} ({p_base:.2e})"
            )

        # ========== 4. basemodel 自回归前进 ==========
        next_token = next_token_id.unsqueeze(0)  # (1,1)
        input_ids = torch.cat([input_ids, next_token], dim=1)
        extra_mask = torch.ones_like(next_token, device=device)
        attention_mask = torch.cat([attention_mask, extra_mask], dim=1)

        if next_token_id.item() == tokenizer.eos_token_id:
            break

    # 新生成部分
    new_token_ids = input_ids[0, prompt_len:]
    new_text = tokenizer.decode(
        new_token_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=True,
    )

    full_text = tokenizer.decode(
        input_ids[0],
        skip_special_tokens=True,
        clean_up_tokenization_spaces=True,
    )

    return full_text, new_text


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt_path = "training/checkpoints/dual_decoder_stageA_head1_student.pt"

    print(f"loading model from {ckpt_path} ...")
    model, tokenizer, config, device = load_model_and_tokenizer(
        ckpt_path=ckpt_path,
        device=device,
    )

    while True:
        try:
            prompt = input("\n输入一个 prompt（或直接回车退出）：\n> ").strip()
        except EOFError:
            break
        if not prompt:
            break

        full, new = generate_base_with_head1_monitor(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            max_new_tokens=64,
            temperature=1.0,
            device=device,
            topk_to_show=3,
        )

        print("\n=== 生成结果(新增部分) ===")
        print(new)
        print("\n=== 全部文本 ===")
        print(full)


if __name__ == "__main__":
    main()
