import torch
from transformers import AutoTokenizer

from mydecoding.config.dual_decoder import DualDecoderConfig
from mydecoding.models.dual_decoder import DualDecoderModel


def load_model_and_tokenizer(
    ckpt_path: str = "training/checkpoints/dual_decoder_stageB_student.pt",
):
    """
    加载 tokenizer + DualDecoderModel，并把 StageB 的 checkpoint 灌进去
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ===== 配置：要和训练时保持一致 =====
    config = DualDecoderConfig(
        base_model_name_or_path="Qwen/Qwen2.5-3B",
        num_draft_candidates=3,
        max_speculative_steps=2,        # => 3 个 phase（1 个 head1 + 2 步 head2）
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
        # acceptance_threshold 用默认的就行（config 里有默认值）
    )

    tokenizer = AutoTokenizer.from_pretrained(
        config.base_model_name_or_path,
        use_fast=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = DualDecoderModel(config)
    state = torch.load(ckpt_path, map_location="cpu")
    # 严格匹配参数名
    model.load_state_dict(state, strict=False)
    model.to(device)
    
    # ★ 关键：统一 dtype，避免 BF16 / Float 混用
    base_dtype = model.base_model.model.embed_tokens.weight.dtype
    print("base_model dtype:", base_dtype)  # 调试用，可以删掉
    model.to(dtype=base_dtype)
    
    
    model.eval()

    return model, tokenizer, device, config


def speculative_generate(
    model: DualDecoderModel,
    tokenizer: AutoTokenizer,
    device: str,
    config: DualDecoderConfig,
    prompt: str,
    max_new_tokens: int = 8,
):
    """
    使用 DualDecoderModel.generate 做一步一步的推理，并打印每一步：
      - 最终决定的 token
      - 每个 phase 的 top-k 候选（这里 k = 3）
    """
    # ===== 编码 prompt =====
    enc = tokenizer(
        prompt,
        return_tensors="pt",
        padding=False,
        add_special_tokens=True,
    )
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)

    # 和训练时一样：phase 总数 = 1（head1） + max_speculative_steps（head2 步数）
    num_phases = 1 + int(config.max_speculative_steps)

    gen_iter = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=max_new_tokens,
        num_phases=num_phases,
        temperature=1.0,
        acceptance_threshold=getattr(config, "acceptance_threshold", 0.05),
    )

    # 记录新生成的 token，最后一起 decode 出来
    decided_token_ids = []

    for step, (decided_token, candidate_groups) in enumerate(gen_iter):
        # decided_token: (B, 1)，我们只看 batch 0
        decided_id = decided_token[0, 0].item()
        decided_token_ids.append(decided_id)
        decided_text = tokenizer.decode(
            [decided_id],
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )

        print(f"\n[step {step}] decided token id = {decided_id}")
        print(f"  decided token text: {repr(decided_text)}")

        # candidate_groups: List[Tensor]，长度 = phase 数
        #   - candidate_groups[0] 是 head1 的 top-k
        #   - candidate_groups[1], [2] 是 head2 不同 phase 的 top-k
        for phase_idx, cand_tensor in enumerate(candidate_groups, start=1):
            # cand_tensor 形状 (B, K)，取 batch 0 -> (K,)
            cand_ids = cand_tensor[0]
            cand_ids_list = cand_ids.tolist()
            cand_texts = [
                tokenizer.decode(
                    [tid],
                    skip_special_tokens=False,
                    clean_up_tokenization_spaces=False,
                )
                for tid in cand_ids_list
            ]

            phase_name = "head1" if phase_idx == 1 else f"head2_phase{phase_idx-1}"
            print(f"  {phase_name} candidates (phase {phase_idx}):")
            for rank, (tid, txt) in enumerate(
                zip(cand_ids_list, cand_texts), start=1
            ):
                print(f"    top{rank}: id={tid:<6} text={repr(txt)}")

        # 如果生成到 eos，可以提前结束
        if decided_id == tokenizer.eos_token_id:
            break

    # ===== 打印最终生成的完整文本 =====
    if decided_token_ids:
        new_text = tokenizer.decode(
            decided_token_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
    else:
        new_text = ""

    full_text = tokenizer.decode(
        input_ids[0],
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    ) + new_text

    print("\n=== 最终生成文本 ===")
    print(full_text)


def main():
    ckpt_path = "training/checkpoints/dual_decoder_stageB_student.pt"
    model, tokenizer, device, config = load_model_and_tokenizer(ckpt_path)

    print("loaded model, device:", device)

    while True:
        try:
            prompt = input("\n输入一个 prompt（或直接回车退出）：\n> ")
        except EOFError:
            break

        if not prompt.strip():
            break

        speculative_generate(
            model,
            tokenizer,
            device,
            config,
            prompt,
            max_new_tokens=8,
        )


if __name__ == "__main__":
    main()
