import argparse
import os
import random


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="path to train.jsonl")
    ap.add_argument("--output", required=True, help="path to output jsonl (sampled 50%)")
    ap.add_argument("--ratio", type=float, default=0.5, help="keep ratio (default 0.5)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed)

    kept = 0
    total = 0

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    with open(args.input, "r", encoding="utf-8") as fin, open(args.output, "w", encoding="utf-8") as fout:
        for line in fin:
            total += 1
            if random.random() < args.ratio:
                fout.write(line)
                kept += 1

    print(f"Done. kept={kept} total={total} ratio={kept/total:.4f} -> {args.output}")


if __name__ == "__main__":
    main()
