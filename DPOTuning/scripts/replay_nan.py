"""Replay the frozen DPO-17 step-3613 NaN in SECONDS — no training loop.

Loads <capture>/nan_weights (weights @ failure) + nan_batch.pt (exact offending
micro-batch), runs ONE forward through TRL's real concatenated_forward -> cpo_loss
path, reproduces the non-finite loss, and prints a per-layer autopsy (first layer
that emits inf/nan). Use this to MEASURE the cause and to test forward-level fixes
instantly (e.g. --fp32-logits) instead of waiting ~8h per attempt.

Usage:
  HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 \
    python scripts/replay_nan.py --config configs/simpo_qlora_capture.yaml \
      --capture checkpoints/simpo-gbr05-capture/nan_capture [--fp32-logits]
"""
import argparse
import os
import torch
from omegaconf import OmegaConf
from peft import PeftModel
from datasets import Dataset
from trl.experimental.cpo import CPOTrainer, CPOConfig

from train_simpo import build_model_and_tokenizer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--capture", required=True, help="path to .../nan_capture dir")
    ap.add_argument("--fp32-logits", action="store_true",
                    help="patch selective_log_softmax to upcast logits to fp32 (test a candidate fix)")
    args = ap.parse_args()
    cfg = OmegaConf.load(args.config)

    model, tok = build_model_and_tokenizer(cfg)
    model = PeftModel.from_pretrained(model, os.path.join(args.capture, "nan_weights"))
    model.eval()

    batch = torch.load(os.path.join(args.capture, "nan_batch.pt"), weights_only=False)
    batch = {k: (v.to(model.device) if torch.is_tensor(v) else v) for k, v in batch.items()}

    if args.fp32_logits:
        import trl.trainer.utils as U
        import trl.experimental.cpo.cpo_trainer as M
        _orig = U.selective_log_softmax
        def patched(logits, index):
            return _orig(logits.float(), index)
        U.selective_log_softmax = patched
        if hasattr(M, "selective_log_softmax"):
            M.selective_log_softmax = patched
        print("[replay] fp32-logits fix ACTIVE")

    cpo = CPOConfig(output_dir="/tmp/replay", loss_type=cfg.loss_type, cpo_alpha=0.0,
                    beta=cfg.beta, simpo_gamma=cfg.simpo_gamma, max_length=cfg.max_length,
                    bf16=cfg.bf16, report_to="none", per_device_train_batch_size=1)
    dummy = Dataset.from_list([{"prompt": "x", "chosen": "y", "rejected": "z"}])
    trainer = CPOTrainer(model=model, args=cpo, train_dataset=dummy,
                         eval_dataset=dummy, processing_class=tok)

    records, handles = [], []
    def mk(nm):
        def h(_m, _i, o):
            t = o[0] if isinstance(o, (tuple, list)) else o
            if torch.is_tensor(t):
                records.append((nm, float(t.float().abs().max()),
                                int(torch.isinf(t).sum()), int(torch.isnan(t).sum())))
        return h
    for nm, mod in model.named_modules():
        if mod.__class__.__name__.endswith("DecoderLayer"):
            handles.append(mod.register_forward_hook(mk(nm)))

    with torch.no_grad():
        out = trainer.concatenated_forward(model, batch)
    ch_lp, rej_lp, ch_lg, rej_lg = out[0], out[1], out[2], out[3]
    loss = trainer.cpo_loss(ch_lp, rej_lp)[0]
    for h in handles:
        h.remove()

    finite = bool(torch.isfinite(loss).all() and torch.isfinite(ch_lp).all()
                  and torch.isfinite(rej_lp).all())
    first_bad = next(((nm, inf, nan) for nm, mx, inf, nan in records if inf or nan), None)
    print(f"\n=== REPLAY {args.capture} (fp32_logits={args.fp32_logits}) ===")
    print(f"loss = {loss.item()}   finite = {finite}")
    print(f"chosen_logps   = {ch_lp.float().tolist()}")
    print(f"rejected_logps = {rej_lp.float().tolist()}")
    print(f"chosen_logits   absmax/inf/nan = {float(ch_lg.float().abs().max())}/"
          f"{int(torch.isinf(ch_lg).sum())}/{int(torch.isnan(ch_lg).sum())}")
    print(f"rejected_logits absmax/inf/nan = {float(rej_lg.float().abs().max())}/"
          f"{int(torch.isinf(rej_lg).sum())}/{int(torch.isnan(rej_lg).sum())}")
    print(f"FIRST non-finite layer = {first_bad}")
    print(f"layer absmax tail = {records[-8:]}")


if __name__ == "__main__":
    main()
