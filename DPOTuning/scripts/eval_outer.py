"""Outer-loop eval wrapper — MT-Bench and AlpacaEval 2 LC.

Costs ~$5–10/run (MT-Bench) and ~$1–10/run (AlpacaEval 2 LC).
Only run when inner-loop (eval_inner.py) passes:
  - avg gen length stable (within ±10% of SFT baseline)
  - refusal rate not spiking

Usage:
    # Mistral base
    python scripts/eval_outer.py \\
        --checkpoint mistralai/Mistral-7B-v0.1 --model_id mistral-base \\
        --mt_bench --tag base --stage base --run_id dpo5_base

    # SFT LoRA checkpoint
    python scripts/eval_outer.py \\
        --checkpoint checkpoints/sft-zephyr-lora/checkpoint-17205 \\
        --base_model mistralai/Mistral-7B-v0.1 --model_id sft-qlora \\
        --mt_bench --tag sft --stage sft --run_id dpo5_sft

    # DPO LoRA checkpoint + both evals
    python scripts/eval_outer.py \\
        --checkpoint checkpoints/dpo-vanilla/checkpoint-XXXX \\
        --base_model mistralai/Mistral-7B-v0.1 --model_id dpo-vanilla \\
        --mt_bench --alpacaeval --tag dpo_default --stage dpo

Prerequisites:
    MT-Bench:   pip install fschat[model_worker,llm_judge]
    AlpacaEval: pip install alpaca_eval
    Both:       export OPENAI_API_KEY=...
"""

import argparse
import csv
import json
import os
import shutil
import statistics
import subprocess
import sys
import tempfile
import uuid
from pathlib import Path

RESULTS_CSV = Path(__file__).parent.parent / "results" / "runs.csv"
CSV_FIELDNAMES = [
    "run_id", "checkpoint", "tag", "stage",
    "beta", "epochs", "lr", "lora_r", "simpo_gamma",
    "avg_gen_length", "p90_gen_length",
    "harmful_refusal_rate", "over_refusal_rate", "pref_acc",
    "mt_bench", "alpacaeval2_lc", "notes",
]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True,
                   help="HF model ID or path to LoRA adapter checkpoint")
    p.add_argument("--base_model", default=None,
                   help="Base model HF ID — required when checkpoint is a LoRA adapter")
    p.add_argument("--model_id", default=None,
                   help="Unique label for FastChat output files (derived from --tag if omitted)")
    p.add_argument("--mt_bench", action="store_true")
    p.add_argument("--alpacaeval", action="store_true")
    p.add_argument("--tag", default=None,
                   help="Short label, e.g. base, sft, dpo_b01_e3")
    p.add_argument("--stage", default=None, choices=["base", "sft", "dpo", "simpo"])
    p.add_argument("--run_id", default=None,
                   help="Explicit run ID (auto-generated if omitted)")
    return p.parse_args()


def _merge_lora(base_model_id: str, lora_path: str, output_dir: Path):
    """Merge LoRA adapter into base model weights and save to output_dir."""
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"  Loading base model {base_model_id} (float16, CPU)...")
    base = AutoModelForCausalLM.from_pretrained(
        base_model_id, torch_dtype=torch.float16, device_map="cpu"
    )
    print(f"  Attaching LoRA adapter from {lora_path}...")
    model = PeftModel.from_pretrained(base, lora_path)
    print("  Merging and unloading LoRA weights...")
    model = model.merge_and_unload()
    model.save_pretrained(str(output_dir))
    AutoTokenizer.from_pretrained(lora_path).save_pretrained(str(output_dir))
    print(f"  Merged model saved → {output_dir}")


def _fastchat_llm_judge_dir() -> Path:
    """Return path to fastchat/llm_judge in the installed package."""
    try:
        import fastchat
        d = Path(fastchat.__file__).parent / "llm_judge"
        if d.is_dir():
            return d
    except ImportError:
        pass
    raise ImportError(
        "fschat not installed or llm_judge directory missing.\n"
        "Install with: pip install fschat[model_worker,llm_judge]"
    )


def _parse_mt_bench_score(model_id: str, judge_dir: Path) -> tuple[float, int]:
    """Extract average MT-Bench score from gpt-4_single.jsonl."""
    judgment_file = (
        judge_dir / "data" / "mt_bench" / "model_judgment" / "gpt-4_single.jsonl"
    )
    if not judgment_file.exists():
        raise FileNotFoundError(
            f"Judgment file not found: {judgment_file}\n"
            "Did gen_judgment complete without errors?"
        )

    scores = []
    with open(judgment_file) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            if entry.get("model") != model_id:
                continue
            score = entry.get("score", -1)
            if isinstance(score, (int, float)) and score != -1:
                scores.append(float(score))

    if not scores:
        raise ValueError(
            f"No valid scores found for model_id={model_id!r} in {judgment_file}"
        )

    avg = round(statistics.mean(scores), 2)
    print(f"  MT-Bench {model_id}: {avg:.2f}  ({len(scores)} turns scored)")
    return avg, len(scores)


def run_mt_bench(checkpoint: str, base_model: str | None, model_id: str) -> dict:
    """Run FastChat MT-Bench: gen_model_answer → gen_judgment → parse scores.

    When base_model is given (LoRA checkpoint), merges weights into a temp
    full model, passes that to FastChat, then cleans up.
    """
    if not os.environ.get("OPENAI_API_KEY"):
        raise EnvironmentError(
            "OPENAI_API_KEY not set — GPT-4 judge requires OpenAI API access."
        )

    judge_dir = _fastchat_llm_judge_dir()
    merged_dir = None
    model_path = checkpoint

    if base_model:
        merged_dir = Path(tempfile.mkdtemp(prefix="mt_bench_merged_"))
        print(f"\nMerging LoRA → {merged_dir}")
        _merge_lora(base_model, checkpoint, merged_dir)
        model_path = str(merged_dir)

    try:
        # Step 1: generate model answers (GPU, free)
        print(f"\n[1/2] Generating MT-Bench answers: {model_id}")
        subprocess.run(
            [
                sys.executable, "-m", "fastchat.llm_judge.gen_model_answer",
                "--model-path", model_path,
                "--model-id", model_id,
                "--bench-name", "mt_bench",
                "--num-gpus-per-model", "1",
            ],
            check=True,
            cwd=str(judge_dir),
        )

        # Step 2: judge with GPT-4 (~$5–10)
        print(f"\n[2/2] Running GPT-4 judgment: {model_id}  (~$5–10)")
        subprocess.run(
            [
                sys.executable, "-m", "fastchat.llm_judge.gen_judgment",
                "--model-list", model_id,
                "--judge-model", "gpt-4",
                "--bench-name", "mt_bench",
                "--mode", "single",
            ],
            check=True,
            cwd=str(judge_dir),
        )

        score, n_turns = _parse_mt_bench_score(model_id, judge_dir)
        return {"mt_bench_score": score, "n_turns": n_turns}

    finally:
        if merged_dir and merged_dir.exists():
            print(f"\nCleaning up merged model at {merged_dir}...")
            shutil.rmtree(merged_dir, ignore_errors=True)


def run_alpacaeval(checkpoint: str, base_model: str | None, model_id: str) -> dict:
    # DPO-15
    raise NotImplementedError(
        "AlpacaEval integration pending (DPO-15).\n"
        "Run manually: alpaca_eval --model_outputs <outputs.json>"
    )


def _append_csv(row: dict):
    """Append one result row to results/runs.csv."""
    RESULTS_CSV.parent.mkdir(parents=True, exist_ok=True)
    write_header = not RESULTS_CSV.exists() or RESULTS_CSV.stat().st_size == 0
    with open(RESULTS_CSV, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDNAMES, extrasaction="ignore")
        if write_header:
            writer.writeheader()
        writer.writerow(row)
    print(f"\nAppended to {RESULTS_CSV}")


def main():
    args = parse_args()
    model_id = args.model_id or (args.tag or Path(args.checkpoint).name)
    run_id = args.run_id or f"run_{uuid.uuid4().hex[:8]}"

    row = {
        "run_id": run_id,
        "checkpoint": args.checkpoint,
        "tag": args.tag or "",
        "stage": args.stage or "",
    }

    if args.mt_bench:
        mt = run_mt_bench(args.checkpoint, args.base_model, model_id)
        row["mt_bench"] = mt["mt_bench_score"]

    if args.alpacaeval:
        ae = run_alpacaeval(args.checkpoint, args.base_model, model_id)
        row["alpacaeval2_lc"] = ae["lc_win_rate"]

    _append_csv(row)
    print(json.dumps(row, indent=2))


if __name__ == "__main__":
    main()
