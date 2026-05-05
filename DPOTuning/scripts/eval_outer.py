"""Outer-loop eval wrapper — MT-Bench and AlpacaEval 2 LC.

Costs ~$5–10/run (MT-Bench) and ~$1–10/run (AlpacaEval 2 LC).
Only run when inner-loop (eval_inner.py) passes:
  - avg gen length stable (within ±10% of SFT baseline)
  - refusal rate not spiking

Usage:
    # MT-Bench only
    python scripts/eval_outer.py --checkpoint checkpoints/dpo-vanilla/checkpoint-XXXX --mt_bench

    # AlpacaEval 2 LC only
    python scripts/eval_outer.py --checkpoint checkpoints/dpo-vanilla/checkpoint-XXXX --alpacaeval

    # Both
    python scripts/eval_outer.py --checkpoint checkpoints/dpo-vanilla/checkpoint-XXXX --mt_bench --alpacaeval

Prerequisites:
    MT-Bench:   git clone https://github.com/lm-sys/FastChat && pip install fschat[model_worker]
    AlpacaEval: pip install alpaca_eval && export OPENAI_API_KEY=...
"""

import argparse
import subprocess
import json
import csv
from pathlib import Path

RESULTS_CSV = Path(__file__).parent.parent / "results" / "runs.csv"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--mt_bench", action="store_true")
    parser.add_argument("--alpacaeval", action="store_true")
    parser.add_argument("--tag", default=None)
    return parser.parse_args()


def run_mt_bench(checkpoint):
    # TODO: integrate FastChat's llm_judge pipeline
    # Expected output: dict with {"mt_bench_score": float, "per_category": {...}}
    raise NotImplementedError(
        "MT-Bench integration pending. "
        "Run FastChat's llm_judge manually and record result in results/runs.csv.\n"
        "  cd FastChat && python -m fastchat.llm_judge.gen_model_answer "
        f"--model-path {checkpoint} --model-id dpo-eval"
    )


def run_alpacaeval(checkpoint):
    # TODO: integrate alpaca_eval pipeline
    # Expected output: dict with {"lc_win_rate": float, "win_rate": float}
    raise NotImplementedError(
        "AlpacaEval integration pending. "
        "Run alpaca_eval manually and record result in results/runs.csv.\n"
        "  alpaca_eval --model_outputs <generated_outputs.json>"
    )


def main():
    args = parse_args()
    results = {"checkpoint": args.checkpoint, "tag": args.tag or ""}

    if args.mt_bench:
        mt = run_mt_bench(args.checkpoint)
        results["mt_bench"] = mt["mt_bench_score"]

    if args.alpacaeval:
        ae = run_alpacaeval(args.checkpoint)
        results["alpacaeval2_lc"] = ae["lc_win_rate"]

    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
