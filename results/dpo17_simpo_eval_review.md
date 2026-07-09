# DPO-17 SimPO — Full Evaluation (3 epochs × 3 benchmarks = 9 points)

Evaluation of the clean 3-epoch SimPO QLoRA run (`checkpoints/simpo-gbr05-fixed`,
the run that completed after the 0/0-NaN fix). All checkpoints loaded as
**base + SFT-merge + SimPO adapter** (the SimPO LoRA is a delta on base+SFT; loading
without the merge silently degrades output — see `scripts/generation.load_model`).

## Results — all 9 data points

| epoch | ckpt | eval-inner avg_len | eval-acc | MT-Bench | AE2 LC | AE raw | AE len(ch) |
|---|---|---|---|---|---|---|---|
| **ep1** | checkpoint-3723  | 438.6 | 0.775 | **7.33** | **31.58** | 30.90 | 1941 |
| ep2 | checkpoint-7446  | 457.4 | 0.813 | 6.86 | 24.71 | 23.24 | 1771 |
| ep3 | checkpoint-11169 | 430.0 | 0.821 | 6.37 | 20.03 | 18.64 | 1717 |

## Anchors (same SFT base / data / eval methodology; from DPO-8 & DPO-15)

| model | MT-Bench | AE2 LC | AE raw | AE len(ch) |
|---|---|---|---|---|
| SFT (zephyr template) | 6.29 | 5.83 | 3.60 | 893 |
| DPO-6 ep1 / ep2 / ep3 | 6.03 / 6.67 / 6.82 | 5.35 / 8.24 / 10.82 | 6.03 / 9.60 / 13.50 | 2306 / 2706 / 2678 |
| **SimPO best (ep1)** | **7.33** | **31.58** | 30.90 | 1941 |
| Zephyr-7B-β (published full-FT) | 7.34 | 13.20 | — | — |

## Findings

**1. SimPO ≫ DPO on AlpacaEval 2 LC.** Best SimPO (ep1) = **31.58% LC** vs best DPO (ep3)
= 10.82% and published Zephyr-7B-β = 13.20%. On MT-Bench, SimPO ep1 (7.33) edges DPO's
best (6.82) and matches Zephyr-7B-β (7.34) — from a QLoRA r=128 adapter, not full FT.

**2. Over-optimization: epoch 1 is the best model.** Both *judged* benchmarks decline
monotonically with epochs — MT-Bench 7.33 → 6.86 → 6.37, AE2 LC 31.58 → 24.71 → 20.03 —
**even as the trainer's eval reward-accuracy rises** (0.775 → 0.813 → 0.821). More SimPO
epochs sharpen the preference ranking but degrade real generation quality. Pick epoch 1.
(SimPO's paper recommends ~1 epoch; here that holds even though LoRA underfits per pass.)

**3. Length control holds (the SimPO point).** LC ≈ raw at every epoch
(gaps +0.69 / +1.48 / +1.39) vs DPO ep3's **−2.68** (raw 13.50 > LC 10.82, inflated by
2678-char outputs). SimPO wins without verbosity-gaming; lengths stay ~1700–1940 ch and
eval-inner gen-length stays flat (438→457→430) where DPO's climbed (417→656→614).

## Methodology (for reproducibility / comparability)

- **eval-inner** (`scripts/eval_inner.py --sft_adapter ...`): 50 fixed prompts, greedy,
  `max_new_tokens=1024` (matches DPO-15/DPO-8). Reports avg/p90 gen length + refusal rates.
- **MT-Bench** (`scripts/run_mtbench.py`): our-client generation (correct Zephyr template
  via the tokenizer's baked-in `chat_template` + SFT-merge load) with fschat's exact
  per-category `temperature_config`; judged by the **identical fschat GPT-4 single-grade
  prompts** (verified line-by-line vs `fastchat/llm_judge/common.py`: NEED_REF_CATS,
  template selection, `[[rating]]` regex, temp=0/max_tokens=2048). Judge model `gpt-4`,
  same as DPO-8. Chosen over installing fschat because the env now has transformers 5.x
  (incompatible with fschat); generating via our client also sidesteps the zephyr-tag
  template-routing bug from DPO-5.
- **AlpacaEval 2** (`scripts/gen_alpaca.py` + `scripts/judge_alpaca.py`): our-client
  generation (greedy, 1024, batched) of the 805 AE2 prompts; judged by the `alpaca_eval`
  package with `weighted_alpaca_eval_gpt4_turbo_new` (gpt-4-turbo, length-controlled),
  same annotator as DPO-15. Judge runs in an isolated `.ae_venv` (datasets<4 for the
  reference loader) so it can't perturb the training env.

Raw artifacts: `results/mt_bench/model_answer|model_judgment/zephyr-simpo-ep*`,
`results/alpaca_eval/zephyr-simpo-ep*_outputs.json`, and `results/runs.csv`.
