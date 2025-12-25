from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

model_name = "Qwen/Qwen2.5-3B"
tok = AutoTokenizer.from_pretrained(model_name, use_fast=True)
model = AutoModelForCausalLM.from_pretrained(model_name).cuda().eval()

prompt = "Q: what is 1+1?A:"
inputs = tok(prompt, return_tensors="pt").to("cuda")
with torch.no_grad():
    out = model.generate(
        **inputs,
        max_new_tokens=20,
        do_sample=False,
    )
print(tok.decode(out[0], skip_special_tokens=True))
