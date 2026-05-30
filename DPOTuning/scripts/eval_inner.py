"""DPO-4: Inner-loop generation diagnostic.

Runs on any saved checkpoint. Reports avg gen length, p90 gen length, and
refusal rate on the 50 fixed prompts in prompts/fixed_50.json.

Run after every training checkpoint (Tier 1). For headline runs (DPO-6 and
DPO-17), also run on intermediate checkpoints to build the length-drift curve.

Usage:
    python scripts/eval_inner.py --checkpoint checkpoints/dpo-vanilla/checkpoint-XXXX
    python scripts/eval_inner.py --checkpoint checkpoints/sft-zephyr-lora/checkpoint-17205 --tag sft_baseline
    python scripts/eval_inner.py --checkpoint checkpoints/dpo-vanilla --all_checkpoints
"""

import argparse
import json
import csv
import os
import numpy as np
from pathlib import Path

import torch
from scripts.generation import load_model, generate
from scripts.refusal_classifier import RefusalClassifier

PROMPTS_PATH = Path(__file__).parent.parent / "prompts" / "fixed_50.json"
RESULTS_CSV = Path(__file__).parent.parent / "results" / "runs.csv"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, help="Path to checkpoint dir or HF model ID")
    parser.add_argument("--base_model", default="mistralai/Mistral-7B-v0.1")
    parser.add_argument("--tag", default=None, help="Human-readable run tag for runs.csv")
    parser.add_argument("--all_checkpoints", action="store_true", help="Iterate all checkpoints under --checkpoint")
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--run_id", default=None, help="Explicit run ID (overrides default ckpt-name derivation)")
    parser.add_argument("--stage", default="", choices=["", "base", "sft", "dpo", "simpo"])
    parser.add_argument("--beta", default="")
    parser.add_argument("--epochs", default="")
    parser.add_argument("--lr", default="")
    parser.add_argument("--lora_r", default="")
    parser.add_argument("--simpo_gamma", default="")
    parser.add_argument("--notes", default="")
    return parser.parse_args()


def run_diagnostic(model, tokenizer, prompts, max_new_tokens):
    clf = RefusalClassifier()
    results = []
    for p in prompts:
        out = generate(model, tokenizer, [{"role": "user", "content": p["prompt"]}], max_new_tokens)
        results.append({
            "n_tokens": len(tokenizer.encode(out)),
            "is_refusal": clf.is_refusal(p["prompt"], out),
            "should_refuse": p.get("should_refuse", False),
        })

    lengths = [r["n_tokens"] for r in results]
    harmful = [r for r in results if r["should_refuse"]]
    benign  = [r for r in results if not r["should_refuse"]]

    return {
        "avg_gen_length":       float(np.mean(lengths)),
        "p90_gen_length":       float(np.percentile(lengths, 90)),
        "harmful_refusal_rate": float(np.mean([r["is_refusal"] for r in harmful])) if harmful else 0.0,
        "over_refusal_rate":    float(np.mean([r["is_refusal"] for r in benign]))  if benign  else 0.0,
        "n_prompts": len(results),
    }


def eval_checkpoint(checkpoint_path, args, prompts):
    print(f"\n=== Evaluating: {checkpoint_path} ===")
    model, tokenizer = load_model(args.base_model, checkpoint_path)
    stats = run_diagnostic(model, tokenizer, prompts, args.max_new_tokens)

    print(f"avg_gen_length       : {stats['avg_gen_length']:.1f} tokens")
    print(f"p90_gen_length       : {stats['p90_gen_length']:.1f} tokens")
    print(f"harmful_refusal_rate : {stats['harmful_refusal_rate'] * 100:.1f}%  (want ~100%)")
    print(f"over_refusal_rate    : {stats['over_refusal_rate'] * 100:.1f}%  (want ~0%)")

    del model
    torch.cuda.empty_cache()
    return stats


CSV_FIELDNAMES = [
    "run_id", "checkpoint", "tag", "stage",
    "beta", "epochs", "lr", "lora_r", "simpo_gamma",
    "max_new_tokens",
    "avg_gen_length", "p90_gen_length",
    "harmful_refusal_rate", "over_refusal_rate", "pref_acc",
    "mt_bench", "alpacaeval2_lc", "notes",
]


def append_to_csv(row):
    """Append one row to runs.csv with column alignment matching the canonical 18-column schema.

    Defensive: if the file is missing a trailing newline, csv.writer would glue the new row onto
    the last existing row. Add the newline first. (Same guard as eval_outer.py / notebook helpers.)
    """
    write_header = not RESULTS_CSV.exists() or RESULTS_CSV.stat().st_size == 0
    if RESULTS_CSV.exists() and RESULTS_CSV.stat().st_size > 0:
        with open(RESULTS_CSV, "rb") as f:
            f.seek(-1, 2)
            last_byte = f.read(1)
        if last_byte not in (b"\n", b"\r"):
            with open(RESULTS_CSV, "ab") as f:
                f.write(b"\n")
    with open(RESULTS_CSV, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDNAMES, extrasaction="ignore")
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def main():
    args = parse_args()
    prompts = json.loads(PROMPTS_PATH.read_text(encoding="utf-8"))

    checkpoints = []
    if args.all_checkpoints:
        base = Path(args.checkpoint)
        checkpoints = sorted(base.glob("checkpoint-*"), key=lambda p: int(p.name.split("-")[-1]))
    else:
        checkpoints = [Path(args.checkpoint)]

    for ckpt in checkpoints:
        stats = eval_checkpoint(str(ckpt), args, prompts)
        run_id = args.run_id or (
            f"{ckpt.parent.name}/{ckpt.name}" if ckpt.name.startswith("checkpoint-") else ckpt.name
        )
        append_to_csv({
            "run_id": run_id,
            "checkpoint": str(ckpt),
            "tag": args.tag or "",
            "stage": args.stage or "",
            "beta": args.beta,
            "epochs": args.epochs,
            "lr": args.lr,
            "lora_r": args.lora_r,
            "simpo_gamma": args.simpo_gamma,
            "max_new_tokens": args.max_new_tokens,
            **stats,
            "notes": args.notes or "",
        })
        print(f"Written to {RESULTS_CSV}")


if __name__ == "__main__":
    main()
