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

from scripts.generation import load_model, _stop_token_ids, _strip_role_markers

OUT_DIR = Path("results/alpaca_eval")
AE_PROMPTS = "data/alpaca_eval_prompts.json"


@torch.inference_mode()
def gen_batch(model, tok, instructions, max_new_tokens, batch_size):
    """Batched greedy generation (left-padded) — same outputs as single, much faster."""
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    stop_ids = _stop_token_ids(tok)
    outs = []
    for i in range(0, len(instructions), batch_size):
        chunk = instructions[i:i + batch_size]
        prompts = [tok.apply_chat_template([{"role": "user", "content": x}],
                                           tokenize=False, add_generation_prompt=True) for x in chunk]
        enc = tok(prompts, return_tensors="pt", padding=True).to(model.device)
        gen = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False,
                             eos_token_id=stop_ids, pad_token_id=tok.eos_token_id)
        plen = enc["input_ids"].shape[1]
        for j in range(len(chunk)):
            txt = tok.decode(gen[j][plen:], skip_special_tokens=True)
            outs.append(_strip_role_markers(txt))
        print(f"  [{min(i+batch_size, len(instructions))}/{len(instructions)}]", flush=True)
    return outs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--sft_adapter", default=None)
    ap.add_argument("--base_model", default="mistralai/Mistral-7B-v0.1")
    ap.add_argument("--model_name", required=True, help="generator label, e.g. zephyr-simpo-ep1")
    ap.add_argument("--max_new_tokens", type=int, default=1024)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--limit", type=int, default=0, help="generate only first N (smoke test)")
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / f"{args.model_name}_outputs.json"
    if out_path.exists():
        n = len(json.loads(out_path.read_text()))
        if n >= 805:
            print(f"[skip] {out_path} already has {n} outputs")
            return

    # datasets 5.x dropped loading-script support for tatsu-lab/alpaca_eval, so we read
    # the identical 805-prompt eval set extracted from the prior AE output files.
    print(f"[data] loading AE2 prompts from {AE_PROMPTS}", flush=True)
    ds = json.loads(Path(AE_PROMPTS).read_text())
    if not args.limit:
        assert len(ds) == 805, f"expected 805 prompts, got {len(ds)}"
    if args.limit:
        ds = ds[: args.limit]
    print(f"[data] {len(ds)} prompts", flush=True)

    print(f"[load] {args.checkpoint} (sft_adapter={args.sft_adapter})", flush=True)
    model, tok = load_model(args.base_model, args.checkpoint, sft_adapter_path=args.sft_adapter)

    instructions = [ex["instruction"] for ex in ds]
    outputs = gen_batch(model, tok, instructions, args.max_new_tokens, args.batch_size)
    results = [{"instruction": ex["instruction"], "output": o,
                "generator": args.model_name, "dataset": ex.get("dataset", "alpaca_eval")}
               for ex, o in zip(ds, outputs)]
    out_path.write_text(json.dumps(results, ensure_ascii=False, indent=2))
    print(f"[done] wrote {len(results)} outputs -> {out_path}", flush=True)


if __name__ == "__main__":
    main()
