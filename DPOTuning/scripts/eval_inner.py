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
import re
import numpy as np
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

PROMPTS_PATH = Path(__file__).parent.parent / "prompts" / "fixed_50.json"
RESULTS_CSV = Path(__file__).parent.parent / "results" / "runs.csv"

REFUSAL_PATTERNS = [
    r"\bi('m| am) (sorry|unable|not able)\b",
    r"\bi (cannot|can't|won't|will not)\b",
    r"\bI don't feel comfortable\b",
    r"\bAs an AI\b",
]
_REFUSAL_RE = re.compile("|".join(REFUSAL_PATTERNS), re.IGNORECASE)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, help="Path to checkpoint dir or HF model ID")
    parser.add_argument("--base_model", default="mistralai/Mistral-7B-v0.1")
    parser.add_argument("--tag", default=None, help="Human-readable run tag for runs.csv")
    parser.add_argument("--all_checkpoints", action="store_true", help="Iterate all checkpoints under --checkpoint")
    parser.add_argument("--max_new_tokens", type=int, default=512)
    return parser.parse_args()


def load_model(base_model_id, checkpoint_path):
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)

    model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model = PeftModel.from_pretrained(model, checkpoint_path)
    model.eval()
    return model, tokenizer


@torch.inference_mode()
def generate(model, tokenizer, messages, max_new_tokens):
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    prompt_len = inputs["input_ids"].shape[1]
    output_ids = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
    )
    return tokenizer.decode(output_ids[0][prompt_len:], skip_special_tokens=True)


def is_refusal(text):
    return bool(_REFUSAL_RE.search(text))


def run_diagnostic(model, tokenizer, prompts, max_new_tokens):
    lengths, refusals = [], []
    for p in prompts:
        out = generate(model, tokenizer, [{"role": "user", "content": p["prompt"]}], max_new_tokens)
        tokens = len(tokenizer.encode(out))
        lengths.append(tokens)
        refusals.append(is_refusal(out))

    return {
        "avg_gen_length": float(np.mean(lengths)),
        "p90_gen_length": float(np.percentile(lengths, 90)),
        "refusal_rate": float(np.mean(refusals)),
        "n_prompts": len(prompts),
    }


def eval_checkpoint(checkpoint_path, args, prompts):
    print(f"\n=== Evaluating: {checkpoint_path} ===")
    model, tokenizer = load_model(args.base_model, checkpoint_path)
    stats = run_diagnostic(model, tokenizer, prompts, args.max_new_tokens)

    print(f"avg_gen_length : {stats['avg_gen_length']:.1f} tokens")
    print(f"p90_gen_length : {stats['p90_gen_length']:.1f} tokens")
    print(f"refusal_rate   : {stats['refusal_rate'] * 100:.1f}%")

    del model
    torch.cuda.empty_cache()
    return stats


def append_to_csv(row):
    fieldnames = [
        "run_id", "checkpoint", "tag", "avg_gen_length", "p90_gen_length",
        "refusal_rate", "pref_acc", "mt_bench", "alpacaeval2_lc", "notes",
    ]
    exists = RESULTS_CSV.exists()
    with open(RESULTS_CSV, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def main():
    args = parse_args()
    prompts = json.loads(PROMPTS_PATH.read_text())

    checkpoints = []
    if args.all_checkpoints:
        base = Path(args.checkpoint)
        checkpoints = sorted(base.glob("checkpoint-*"), key=lambda p: int(p.name.split("-")[-1]))
    else:
        checkpoints = [Path(args.checkpoint)]

    for ckpt in checkpoints:
        stats = eval_checkpoint(str(ckpt), args, prompts)
        run_id = f"{ckpt.parent.name}/{ckpt.name}" if ckpt.name.startswith("checkpoint-") else ckpt.name
        append_to_csv({
            "run_id": run_id,
            "checkpoint": str(ckpt),
            "tag": args.tag or "",
            **stats,
            "pref_acc": "",
            "mt_bench": "",
            "alpacaeval2_lc": "",
            "notes": "",
        })
        print(f"Written to {RESULTS_CSV}")


if __name__ == "__main__":
    main()
