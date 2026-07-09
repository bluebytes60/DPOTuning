"""Smoke test: does a SimPO checkpoint load correctly and generate coherent text?

Uses the PROVEN loader build_model_and_tokenizer (base + SFT merged, same path that
worked in diag_nan_ckpt.py) then attaches the SimPO adapter — instead of
generation.load_model, which stalled. Reuses generation.generate for decoding.

Run (unbuffered so progress is visible):
  HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1 python scripts/smoke_gen.py \
    --config configs/simpo_qlora.yaml \
    --checkpoint checkpoints/simpo-gbr05-fixed/checkpoint-3723
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from omegaconf import OmegaConf
from peft import PeftModel

from scripts.train_simpo import build_model_and_tokenizer
from scripts.generation import generate

PROMPTS = [
    "What is the capital of France?",
    "Explain photosynthesis in two sentences.",
    "Write a haiku about the ocean.",
    "Give me three tips for staying focused while working from home.",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="config with model_name_or_path = SFT checkpoint")
    ap.add_argument("--checkpoint", required=True, help="SimPO adapter checkpoint to attach")
    ap.add_argument("--max_new_tokens", type=int, default=256)
    args = ap.parse_args()
    cfg = OmegaConf.load(args.config)

    t0 = time.time()
    print(f"[load] base+SFT via build_model_and_tokenizer (sft={cfg.model_name_or_path})", flush=True)
    model, tok = build_model_and_tokenizer(cfg)
    print(f"[load] base+SFT merged in {time.time()-t0:.0f}s; attaching SimPO adapter {args.checkpoint}", flush=True)
    model = PeftModel.from_pretrained(model, args.checkpoint)
    model.config.use_cache = True
    model.eval()
    print(f"[load] OK in {time.time()-t0:.0f}s total — generating...\n", flush=True)

    for p in PROMPTS:
        out = generate(model, tok, [{"role": "user", "content": p}], args.max_new_tokens)
        print(f"### PROMPT: {p}\n{out}\n{'-'*70}", flush=True)


if __name__ == "__main__":
    main()
