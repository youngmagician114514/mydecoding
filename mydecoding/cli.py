import argparse
import torch

from .config import DualDecoderConfig
from .model import DualDecoderModel


def build_model(model_name: str) -> DualDecoderModel:
    config = DualDecoderConfig(base_model_name_or_path=model_name)
    return DualDecoderModel(config)


def main():
    parser = argparse.ArgumentParser(
        description="Belief-fused speculative decoding on top of Llama (Hydra-inspired)."
    )
    parser.add_argument("--model", type=str, required=True, help="HF repo or local path for the Llama base model (e.g., meta-llama/Meta-Llama-3-8B).")
    parser.add_argument("--prompt", type=str, required=True, help="Text prompt.")
    parser.add_argument("--max-new-tokens", type=int, default=64, help="Number of new tokens to generate.")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature.")
    args = parser.parse_args()

    model = build_model(args.model)
    tokenizer = model.tokenizer

    inputs = tokenizer(args.prompt, return_tensors="pt").to(model.base_model.device)
    generated = []
    for token in model.generate(
        input_ids=inputs["input_ids"],
        attention_mask=inputs.get("attention_mask"),
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
    ):
        generated.append(token)

    new_tokens = torch.cat(generated, dim=-1)
    decoded = tokenizer.decode(new_tokens[0], skip_special_tokens=True)
    print(decoded)


if __name__ == "__main__":
    main()
