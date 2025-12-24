# mydecoding/mydecoding/infer_dual_decoder.py
"""
使用已经训练好的 dual-decoder 做推理（step-level 生成示例）：

- 加载 tokenizer + DualDecoderModel
- 加载 StageB 的 checkpoint
- 给一个 prompt，迭代地生成 max_new_tokens 个 token
- 每一步都通过 head1 + fusion + head2，取最后一 phase 的 top-1 作为下一个 token
  （这里为了简单，没有做和 base 的 accept/reject，只是用学生模型自己滚）
"""

import torch
from transformers import AutoTokenizer

from mydecoding.config.dual_decoder import DualDecoderConfig
from mydecoding.models.dual_decoder import DualDecoderModel


def load_model_and_tokenizer(
    ckpt_path: str,
    device: str = "cuda",
):
    # 配置要和训练时保持一致
    config = DualDecoderConfig(
        base_model_name_or_path="Qwen/Qwen2.5-3B",
        num_draft_candidates=3,
        max_speculative_steps=2,   # 你 StageB 训练用多少就填多少
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
    model.load_state_dict(state, strict=False)
    model.to(device)
    model.eval()

    return model, tokenizer, config


@torch.no_grad()
def speculative_generate(
    model: DualDecoderModel,
    tokenizer,
    config: DualDecoderConfig,
    prompt: str,
    max_new_tokens: int = 32,
    temperature: float = 1.0,
    device: str = "cuda",
):
    # 编码 prompt
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        padding=False,
        add_special_tokens=True,
    )
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)

    num_phases = 1 + int(config.max_speculative_steps)

    use_bf16 = torch.cuda.is_bf16_supported()
    autocast_dtype = torch.bfloat16 if use_bf16 else torch.float16

    for step in range(max_new_tokens):
        with torch.autocast(device_type="cuda", dtype=autocast_dtype):
            out = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                num_phases=num_phases,
                temperature=temperature,
            )

        # out.candidate_ids 形状大致是 (B, num_phases, K)
        cand_ids = out.candidate_ids  # (1, P, K)
        # 取最后一个 phase 的 top-1 作为下一个 token
        next_token_id = cand_ids[0, -1, 0].item()

        # 把新 token 接到序列后面
        next_token = torch.tensor([[next_token_id]], device=device, dtype=input_ids.dtype)
        input_ids = torch.cat([input_ids, next_token], dim=1)
        # attention_mask 同步扩展
        extra_mask = torch.ones_like(next_token, device=device)
        attention_mask = torch.cat([attention_mask, extra_mask], dim=1)

    # 解码整个序列
    output_text = tokenizer.decode(
        input_ids[0],
        skip_special_tokens=True,
        clean_up_tokenization_spaces=True,
    )
    return output_text


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt_path = "training/checkpoints/dual_decoder_stageB_fusion_head2.pt"

    print(f"loading model from {ckpt_path} ...")
    model, tokenizer, config = load_model_and_tokenizer(ckpt_path, device=device)

    while True:
        try:
            prompt = input("\n输入一个 prompt（或直接回车退出）：\n> ").strip()
        except EOFError:
            break

        if not prompt:
            break

        text = speculative_generate(
            model=model,
            tokenizer=tokenizer,
            config=config,
            prompt=prompt,
            max_new_tokens=64,
            temperature=1.0,
            device=device,
        )
        print("\n=== 生成结果 ===")
        print(text)


if __name__ == "__main__":
    main()
