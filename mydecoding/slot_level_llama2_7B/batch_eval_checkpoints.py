import argparse
import os
import subprocess
import sys
import time
from pathlib import Path


def run_cmd(cmd, log_file):
    start = time.time()
    with log_file.open("a", encoding="utf-8") as f:
        f.write("\n" + "=" * 120 + "\n")
        f.write(f"[CMD] {' '.join(cmd)}\n")
        f.flush()
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            print(line, end="")
            f.write(line)
        code = proc.wait()
    return code, time.time() - start


def main():
    parser = argparse.ArgumentParser("Batch evaluate draft_model_xxxx.pt checkpoints")
    parser.add_argument("--model_checkpoint", type=str, required=True)
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--ckpt_dir", type=str, required=True, help="Directory containing draft_model_XXXX.pt")
    parser.add_argument("--start", type=int, default=5000)
    parser.add_argument("--end", type=int, default=40000)
    parser.add_argument("--interval", type=int, default=5000)
    parser.add_argument("--num_layers", type=int, default=1)
    parser.add_argument("--fusion_layers", type=str, default="[-2,-1,-3]")
    parser.add_argument("--draft_len", type=int, default=5)
    parser.add_argument("--slot_size", type=int, default=5)
    parser.add_argument("--threshold", type=float, default=0.036)
    parser.add_argument("--dtype", type=str, default="bf16", choices=["bf16", "fp16", "fp32"])
    parser.add_argument("--tokenizer_dir", type=str, default="", help="Optional tokenizer cache directory")
    parser.add_argument("--python_bin", type=str, default=sys.executable)
    parser.add_argument("--log_dir", type=str, default="./batch_eval_logs")
    parser.add_argument("--stop_on_error", action="store_true")
    args = parser.parse_args()

    os.makedirs(args.log_dir, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    summary_path = Path(args.log_dir) / f"summary_{ts}.txt"

    steps = list(range(args.start, args.end + 1, args.interval))
    print(f"[INFO] total checkpoints to scan: {len(steps)}")
    print(f"[INFO] summary: {summary_path}")

    with summary_path.open("w", encoding="utf-8") as summary:
        summary.write("step\tcheckpoint\teval_humaneval_edd\teval_humaneval\tseconds_edd\tseconds_tree\n")

        for step in steps:
            ckpt = Path(args.ckpt_dir) / f"draft_model_{step}.pt"
            if not ckpt.exists():
                print(f"[SKIP] missing: {ckpt}")
                summary.write(f"{step}\t{ckpt}\tMISSING\tMISSING\t0\t0\n")
                summary.flush()
                continue

            print(f"\n[RUN] checkpoint: {ckpt}")
            log_file = Path(args.log_dir) / f"eval_step_{step}_{ts}.log"

            cmd_edd = [
                args.python_bin,
                "eval_humaneval_edd.py",
                "--model_checkpoint",
                args.model_checkpoint,
                "--draft_model_checkpoint",
                str(ckpt),
                "--data_dir",
                args.data_dir,
                "--num_layers",
                str(args.num_layers),
                "--fusion_layers",
                args.fusion_layers,
                "--draft_len",
                str(args.draft_len),
                "--slot_size",
                str(args.slot_size),
                "--dtype",
                args.dtype,
            ]
            if args.tokenizer_dir:
                cmd_edd.extend(["--tokenizer_dir", args.tokenizer_dir])
            code_edd, sec_edd = run_cmd(cmd_edd, log_file)

            cmd_tree = [
                args.python_bin,
                "eval_humaneval.py",
                "--model_checkpoint",
                args.model_checkpoint,
                "--draft_model_checkpoint",
                str(ckpt),
                "--data_dir",
                args.data_dir,
                "--num_layers",
                str(args.num_layers),
                "--fusion_layers",
                args.fusion_layers,
                "--threshold",
                str(args.threshold),
                "--slot_size",
                str(args.slot_size),
                "--dtype",
                args.dtype,
            ]
            if args.tokenizer_dir:
                cmd_tree.extend(["--tokenizer_dir", args.tokenizer_dir])
            code_tree, sec_tree = run_cmd(cmd_tree, log_file)

            summary.write(
                f"{step}\t{ckpt}\t{code_edd}\t{code_tree}\t{sec_edd:.1f}\t{sec_tree:.1f}\n"
            )
            summary.flush()

            if args.stop_on_error and (code_edd != 0 or code_tree != 0):
                print("[STOP] stop_on_error triggered.")
                break

    print(f"\n[DONE] finished. summary saved to: {summary_path}")


if __name__ == "__main__":
    main()
