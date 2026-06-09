"""AlpacaEval 2 generation (our client, correct SFT-merge load) — matches DPO-15/DPO-17.

Generates 805 outputs with scripts.generation.generate (greedy, max_new_tokens=1024) —
the SAME generate fn the prior AE runs used — but loads SimPO correctly via sft_adapter.
Output JSON ({instruction, output, generator, dataset}) is then judged by the alpaca_eval
package (weighted_alpaca_eval_gpt4_turbo_new) in the isolated .ae_venv.

Run:
  HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1 PYTHONPATH=. \
    python scripts/gen_alpaca.py \
      --checkpoint checkpoints/simpo-gbr05-fixed/checkpoint-3723 \
      --sft_adapter checkpoints/sft-zephyr-lora/checkpoint-17205 \
      --model_name zephyr-simpo-ep1
"""
import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from datasets import load_dataset

from scripts.generation import load_model, generate

OUT_DIR = Path("results/alpaca_eval")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--sft_adapter", default=None)
    ap.add_argument("--base_model", default="mistralai/Mistral-7B-v0.1")
    ap.add_argument("--model_name", required=True, help="generator label, e.g. zephyr-simpo-ep1")
    ap.add_argument("--max_new_tokens", type=int, default=1024)
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / f"{args.model_name}_outputs.json"
    if out_path.exists():
        n = len(json.loads(out_path.read_text()))
        if n >= 805:
            print(f"[skip] {out_path} already has {n} outputs")
            return

    print(f"[data] loading tatsu-lab/alpaca_eval (805 prompts)", flush=True)
    ds = load_dataset("tatsu-lab/alpaca_eval", "alpaca_eval", trust_remote_code=True)["eval"]
    print(f"[data] {len(ds)} prompts", flush=True)

    print(f"[load] {args.checkpoint} (sft_adapter={args.sft_adapter})", flush=True)
    model, tok = load_model(args.base_model, args.checkpoint, sft_adapter_path=args.sft_adapter)

    results = []
    for i, ex in enumerate(ds):
        out = generate(model, tok, [{"role": "user", "content": ex["instruction"]}],
                       max_new_tokens=args.max_new_tokens)
        results.append({"instruction": ex["instruction"], "output": out,
                        "generator": args.model_name, "dataset": ex.get("dataset", "alpaca_eval")})
        if (i + 1) % 25 == 0:
            print(f"  [{i+1}/{len(ds)}]", flush=True)
            out_path.write_text(json.dumps(results, ensure_ascii=False, indent=2))  # checkpoint progress
    out_path.write_text(json.dumps(results, ensure_ascii=False, indent=2))
    print(f"[done] wrote {len(results)} outputs -> {out_path}", flush=True)


if __name__ == "__main__":
    main()
