# save_wiki.py
from datasets import load_dataset

# 下 wikitext（在“有网 + 有权限”的那台机器上）
ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")

out_path = "wikitext_train.txt"

with open(out_path, "w", encoding="utf-8") as f:
    for ex in ds:
        text = ex["text"]
        # 有些样本是空行，可以直接跳过
        if not text.strip():
            continue
        # 可选：把内部换行替换掉，保证一条样本一行
        text = text.replace("\n", " ")
        f.write(text + "\n")

print("saved to", out_path)
