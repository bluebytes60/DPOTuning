# Money Table

Best checkpoint per method. Same SFT init, same cleaned-UltraFeedback data, same eval
methodology across all rows (see per-story review docs for provenance).

| Model | MT-Bench | AE2 LC | AE2 Raw | LC − Raw | Avg gen len (chars) |
|---|---|---|---|---|---|
| Mistral-7B-v0.1 (base) | 3.17 | — | — | — | — |
| SFT (UltraChat, QLoRA) | 6.29 | 5.83% | 3.60% | +2.23 | 893 |
| DPO vanilla (UltraFeedback, QLoRA, β=0.1, best = ep3) | 6.82 | 10.82% | 13.50% | **−2.68** | 2,678 |
| **SimPO (UltraFeedback, QLoRA, β=2.0 γ=1.0, best = ep1)** | **7.33** | **31.58%** | 30.90% | **+0.69** | 1,941 |
| Zephyr-7B-β (published, full-FT) | **7.34** | 13.20% | — | — | — |

## Epoch trajectory (over-optimization / length-bias evidence)

| Method | Metric | ep1 | ep2 | ep3 |
|---|---|---|---|---|
| DPO | MT-Bench | 6.03 | 6.67 | **6.82** |
| DPO | AE2 LC | 5.35% | 8.24% | **10.82%** |
| DPO | AE2 gen len (chars) | 2,306 | 2,706 | 2,678 |
| SimPO | MT-Bench | **7.33** | 6.86 | 6.37 |
| SimPO | AE2 LC | **31.58%** | 24.71% | 20.03% |
| SimPO | trainer eval acc | 0.775 | 0.813 | 0.821 |
| SimPO | AE2 gen len (chars) | 1,941 | 1,771 | 1,717 |

## How to read

- **SFT → DPO gap** = alignment effect. DPO ep3 clears SFT by +0.53 MT-Bench / +4.99 pp AE2 LC.
- **DPO → SimPO gap on AE2 LC** = length-bias correction, quantified. DPO's best checkpoint
  is length-inflated (raw 13.50 > LC 10.82, i.e. **−2.68** from length control on 2,678-char
  outputs). SimPO wins *without* verbosity: LC ≈ raw (**+0.69**) at ~700 fewer chars.
- **DPO/SimPO vs Zephyr-7B-β** = QLoRA-vs-full-FT cost. DPO lands at ~80% of Zephyr's published
  AE2 LC (10.82 vs 13.20) and 0.52 below on MT-Bench — the honest QLoRA gap. SimPO ep1 **matches
  Zephyr on MT-Bench (7.33 vs 7.34) and exceeds its published AE2 LC 2.4×** — from a QLoRA r=128
  adapter, not full fine-tuning.
- **Over-optimization:** for SimPO, epoch 1 is the best model — both *judged* benchmarks decline
  ep1→ep3 (MT 7.33→6.37, LC 31.58→20.03) **even as trainer eval accuracy rises** (0.775→0.821).
  More SimPO passes sharpen the preference ranking but degrade real generation quality.

## Not run (scope notes)

- **β sweep {0.01, 0.1, 0.3}** (DPO-9) and **LR sweep** (DPO-11): deferred in favor of the
  SimPO comparison, which isolates the loss function as the only variable and carries more
  narrative weight (see `dpo15_alpaca_review.md`). LR sensitivity was instead explored via the
  SimPO collapse (lr=5e-6 vs paper-exact 5e-7 — see `dpo17_simpo_nan_rootcause.md`).
- **SimPO γ=0.3 sensitivity row**: config exists (`configs/simpo_qlora_gbr03.yaml`) but was not
  evaluated; only the γ=1.0 (gbr=0.5) run has results.

Sources: `results/runs.csv`, `dpo8_mt_bench_review.md`, `dpo15_alpaca_review.md`,
`dpo17_simpo_eval_review.md`.
