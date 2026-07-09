"""Diagnostic: replay the step-3613 NaN batch's 4 rows on a FRESH model.

Decisive test for H_data (a row is intrinsically poison -> 0/0 nan on any model)
vs H_state (rows innocent; nan only at the drifted step-3600 weights).

For each of the 4 rows, on the untrained SFT model, through TRL's EXACT code path
(tokenize_row -> concatenated_forward -> cpo_loss), we print:
  - chosen/rejected scored-token counts (labels != -100)  <- 0 here == 0/0 == H_data
  - per-example avg logps and the SimPO loss, flagging non-finite.

Run:  HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python scripts/diag_nan_batch.py --config configs/simpo_qlora.yaml
"""
import argparse
import torch
from trl.experimental.cpo import CPOTrainer, CPOConfig
from peft import LoraConfig
from datasets import load_dataset, Dataset
from omegaconf import OmegaConf

from train_simpo import build_model_and_tokenizer, format_dataset

# Distinctive substrings of the 4 prompts the NaN-guard dumped at step 3613.
SUBSTRINGS = [
    "extract the title from a JSON formatted news article using C++",
    "What's the length of the air?",
    "123 : 36 : : 221",
    "Translate it from the English language to the Oriya language",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = OmegaConf.load(args.config)

    model, tokenizer = build_model_and_tokenizer(cfg)

    # Load + format the same training split, then pull the 4 dumped rows by prompt match.
    ds = load_dataset(list(cfg.dataset_mixer.keys())[0], split=list(cfg.dataset_splits)[0])
    ds = format_dataset(ds, tokenizer)
    rows = []
    for sub in SUBSTRINGS:
        hit = ds.filter(lambda e, s=sub: s in e["prompt"])
        print(f"[find] {sub[:40]!r:45} -> {len(hit)} match(es)")
        if len(hit):
            rows.append(hit[0])
    four = Dataset.from_list(rows)

    lora = LoraConfig(r=cfg.lora_r, lora_alpha=cfg.lora_alpha, lora_dropout=cfg.lora_dropout,
                      target_modules=list(cfg.lora_target_modules), bias="none", task_type="CAUSAL_LM")
    cpo_config = CPOConfig(
        output_dir="/tmp/diag", loss_type=cfg.loss_type, cpo_alpha=0.0,
        beta=cfg.beta, simpo_gamma=cfg.simpo_gamma,
        per_device_train_batch_size=1, max_length=cfg.max_length,
        bf16=cfg.bf16, report_to="none",
    )
    trainer = CPOTrainer(model=model, args=cpo_config, train_dataset=four,
                         eval_dataset=four, processing_class=tokenizer, peft_config=lora)

    pad = -100  # DPODataCollatorWithPadding default label_pad_token_id (TRL 0.29 exposes no attr)
    print(f"\nlabel_pad_token_id={pad}  pad_token_id={tokenizer.pad_token_id} "
          f"(eos={tokenizer.eos_token_id})\n")

    collator = trainer.data_collator
    model.eval()
    print(f"{'#':>2} {'prompt':45} {'chosen_tok':>10} {'rej_tok':>8} "
          f"{'logp_ch':>9} {'logp_rej':>9} {'loss':>9}  finite?")
    for i in range(len(trainer.train_dataset)):
        ex = trainer.train_dataset[i]
        n_ch  = int((torch.tensor(ex["chosen_labels"])   != pad).sum())
        n_rej = int((torch.tensor(ex["rejected_labels"]) != pad).sum())
        batch = collator([ex])
        batch = {k: (v.to(model.device) if torch.is_tensor(v) else v) for k, v in batch.items()}
        with torch.no_grad():
            out = trainer.concatenated_forward(model, batch)
            ch_logps, rej_logps = out[0], out[1]
            loss = trainer.cpo_loss(ch_logps, rej_logps)[0]
        finite = bool(torch.isfinite(loss).all() and torch.isfinite(ch_logps).all()
                      and torch.isfinite(rej_logps).all())
        tag = SUBSTRINGS[i][:43] if i < len(SUBSTRINGS) else "?"
        print(f"{i:>2} {tag:45} {n_ch:>10} {n_rej:>8} "
              f"{ch_logps.item():>9.3f} {rej_logps.item():>9.3f} {loss.item():>9.3f}  {finite}")


if __name__ == "__main__":
    main()
