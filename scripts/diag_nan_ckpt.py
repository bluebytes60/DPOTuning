"""Diagnostic: replay the step-3613 NaN batch on the PRE-CLIFF checkpoint-3500.

DPO-17 H_state is proven (same 4 rows are finite at step-0, nan at step-3613). The
failure is a non-finite forward LOSS (not grad explosion). Hypothesis: at the drifted
step-~3600 weights a completion token's prob underflows to 0 in bf16 -> log(0)=-inf
-> nan SimPO loss. This script tests that on checkpoint-3500 (113 steps before the cliff):

For each of the 4 rows, through TRL's exact concatenated_forward -> cpo_loss path, in
BOTH bf16 (the training dtype, should reproduce non-finite/extreme logps) and fp32
(should be finite), we print:gergergergrgeergv22222222f23wf22erfwferwerff2222edfwfe2w3qwefqwefefwfe
  - chosen/rejected avg logps, the SimPO loss, and finite? for each dtype
  - whether bf16 logps are -inf/nan while fp32 are finite  <- confirms underflow + fp32 fix

Run:
  HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 \
    python scripts/diag_nan_ckpt.py --config configs/simpo_qlora.yaml \
      --ckpt checkpoints/simpo-gbr05/checkpoint-3500
"""
import argparse
import torch
from omegaconf import OmegaConf
from datasets import load_dataset, Dataset
from peft import PeftModel
from trl.experimental.cpo import CPOTrainer, CPOConfig

from train_simpo import build_model_and_tokenizer, format_dataset

SUBSTRINGS = [
    "extract the title from a JSON formatted news article using C++",
    "What's the length of the air?",
    "123 : 36 : : 221",
    "Translate it from the English language to the Oriya language",
]


def run_pass(trainer, model, label):
    """Run all 4 rows (batch=1 each) through the exact loss path; return per-row stats."""
    pad = -100  # DPODataCollatorWithPadding default label_pad_token_id (TRL 0.29 has no attr)
    collator = trainer.data_collator
    model.eval()
    print(f"\n===== PASS: {label} (model dtype = {next(model.parameters()).dtype}) =====")
    print(f"{'#':>2} {'prompt':40} {'logp_ch':>11} {'logp_rej':>11} {'loss':>11}  finite?")
    rows = []
    for i in range(len(trainer.train_dataset)):
        ex = trainer.train_dataset[i]
        n_ch = int((torch.tensor(ex["chosen_labels"]) != pad).sum())
        n_rej = int((torch.tensor(ex["rejected_labels"]) != pad).sum())
        batch = collator([ex])
        batch = {k: (v.to(model.device) if torch.is_tensor(v) else v) for k, v in batch.items()}
        with torch.no_grad():
            ch_logps, rej_logps = trainer.concatenated_forward(model, batch)[:2]
            loss = trainer.cpo_loss(ch_logps, rej_logps)[0]
        finite = bool(torch.isfinite(loss).all() and torch.isfinite(ch_logps).all()
                      and torch.isfinite(rej_logps).all())
        tag = SUBSTRINGS[i][:38] if i < len(SUBSTRINGS) else "?"
        print(f"{i:>2} {tag:40} {ch_logps.item():>11.4f} {rej_logps.item():>11.4f} "
              f"{loss.item():>11.4f}  {finite}   (n_ch={n_ch}, n_rej={n_rej})")
        rows.append((finite, ch_logps.item(), rej_logps.item(), loss.item()))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", required=True, help="path to checkpoint-NNNN (LoRA adapter)")
    args = ap.parse_args()
    cfg = OmegaConf.load(args.config)

    # Merged SFT base (bf16), then stack the trained checkpoint adapter on top.
    model, tokenizer = build_model_and_tokenizer(cfg)
    print(f"\n[load] applying trained adapter: {args.ckpt}")
    model = PeftModel.from_pretrained(model, args.ckpt)

    # Pull the same 4 dumped rows by prompt match.
    ds = load_dataset(list(cfg.dataset_mixer.keys())[0], split=list(cfg.dataset_splits)[0])
    ds = format_dataset(ds, tokenizer)
    picked = []
    for sub in SUBSTRINGS:
        hit = ds.filter(lambda e, s=sub: s in e["prompt"])
        print(f"[find] {sub[:40]!r:45} -> {len(hit)} match(es)")
        if len(hit):
            picked.append(hit[0])
    four = Dataset.from_list(picked)

    cpo_config = CPOConfig(
        output_dir="/tmp/diag_ckpt", loss_type=cfg.loss_type, cpo_alpha=0.0,
        beta=cfg.beta, simpo_gamma=cfg.simpo_gamma,
        per_device_train_batch_size=1, max_length=cfg.max_length,
        bf16=cfg.bf16, report_to="none",
    )
    # model is already a PeftModel -> no peft_config (avoid double-wrap)
    trainer = CPOTrainer(model=model, args=cpo_config, train_dataset=four,
                         eval_dataset=four, processing_class=tokenizer)

    # Pass A: bf16 (training dtype) — expect this to reproduce non-finite / extreme logps.
    bf16_rows = run_pass(trainer, model, "bf16 (as trained)")

    # Pass B: fp32 — upcast the whole model; expect finite (validates the fp32-loss fix).
    model.float()
    fp32_rows = run_pass(trainer, model, "fp32 (upcast)")

    # Verdict.
    print("\n===== VERDICT =====")
    any_bf16_bad = any(not r[0] for r in bf16_rows)
    all_fp32_ok = all(r[0] for r in fp32_rows)
    for i in range(len(bf16_rows)):
        b, f = bf16_rows[i], fp32_rows[i]
        flag = ""
        if not b[0] and f[0]:
            flag = "  <-- bf16 NON-FINITE, fp32 FINITE (underflow confirmed)"
        elif not b[0] and not f[0]:
            flag = "  <-- non-finite in BOTH (not just precision)"
        print(f"row {i} ({SUBSTRINGS[i][:30]!r:32}): bf16 finite={b[0]}  fp32 finite={f[0]}{flag}")
    print(f"\nbf16 reproduces non-finite at ckpt-3500: {any_bf16_bad}")
    print(f"fp32 makes all 4 finite:                 {all_fp32_ok}")
    if any_bf16_bad and all_fp32_ok:
        print(">>> MECHANISM CONFIRMED: bf16 underflow -> -inf logp -> nan loss; fp32 upcast fixes it.")
    elif not any_bf16_bad:
        print(">>> ckpt-3500 (113 steps pre-cliff) is still finite — drift not yet severe; "
              "the underflow crosses the threshold between step 3500 and 3613.")


if __name__ == "__main__":
    main()
