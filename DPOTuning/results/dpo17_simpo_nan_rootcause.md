# DPO-17 SimPO NaN — Root Cause (MEASURED, resolved 2026-06-07)

> **This supersedes the cause sections of `dpo17_simpo_collapse.md`.** That doc's
> "lr=5e-6 / drift / overflow" explanations were unvalidated guesses and are WRONG.
> The root cause below was captured from the exact failing step, not inferred.

## TL;DR
The SimPO QLoRA run deterministically NaNs at **step 3613** (epoch ~0.97). The cause is
**a single poison row** (an Oriya-translation example) whose **completions truncate to
zero tokens**, so SimPO's length-normalized reward computes **0 / 0 = NaN**. It is
`H_data` (bad data), not `H_state` (training dynamics). It was *not* learning rate, *not*
rsLoRA, *not* gradient explosion, *not* bf16 overflow.

## How we proved it (the method that finally worked)
Instead of guessing-and-guarding (which we did for two days), we **captured the exact
failure and replayed it**:
- `NaNGuardCPOTrainer._capture_failure` (in `scripts/train_simpo.py`) dumps, the instant
  the loss is non-finite: the exact offending micro-batch (`nan_batch.pt`), the adapter
  weights at that step (`nan_weights/`), and a per-layer autopsy (`nan_autopsy.json`).
- A full resumable checkpoint is saved at step 3600 (13 steps before the cliff) so any
  future experiment resumes in minutes, not ~8 h.
- `scripts/replay_nan.py` reloads weights+batch and reproduces the NaN in seconds.

**Autopsy result (step 3613):**
- logits are **finite** — abs-max 47.5 / 48.5, **0 inf, 0 nan**. (Kills the overflow theory.)
- **no layer** produced a non-finite value (`first_nonfinite_layer: null`).
- the NaN is isolated to **row 3**: `chosen_logps[3]=NaN`, `rejected_logps[3]=NaN`.
- captured scored-token counts: `chosen=[184,142,364,0]`, `rejected=[391,44,100,0]`
  → **row 3 has 0 scored tokens on BOTH sides** → `sum(logps)/0 = 0/0 = NaN`.

## The mechanism, with the real numbers for row 3
Row 3 is a `flan_v2_niv2` English→Oriya translation task.

| part | characters | tokens | tokens/char |
|---|---|---|---|
| prompt | 1079 | 535 | 0.50 |
| chosen answer | 563 | 608 | 1.08 |
| rejected answer | 792 | **1024** | 1.29 |
| (English, for contrast) | 83 | 16 | 0.19 |

Three things line up:
1. **Tokenizer inefficiency on Oriya.** The Mistral/Zephyr BPE vocab is English-centric;
   Oriya falls back to byte-level pieces → ~**1.3 tokens/char** (~6× English's 0.19). A
   paragraph-sized answer (792 chars) becomes **1024 tokens** = the entire `max_length`.
   *(The tokenizer does not fail — it is just very inefficient on this script.)*
2. **TRL truncation rule.** `CPOTrainer.tokenize_row` cuts each answer to
   `max_length − longer_response_length`. Here longer = rejected = 1024, so the budget is
   `1024 − 1024 = 0` → **both** answers are emptied — even the chosen (608), which alone
   would have fit beside the 535-token prompt. One oversized answer poisons both sides.
3. **SimPO length normalization.** reward = `sum(token_logps) / num_tokens` = `0/0 = NaN`.
   DPO is immune because it *sums* (empty → 0, finite); SimPO *averages*.

Toy version (`max_length=10`): prompt=3, chosen=4, rejected=12 → budget `10−12=−2→0` →
both answers cut to 0 → `0/0` NaN, even though prompt+chosen=7 would have fit.

## Why the existing filter missed it
`_completion_survives_truncation` (train_simpo.py) **models** TRL's truncation arithmetic
(`k = max_length − longer`) but the model has gaps vs TRL 0.29's actual `tokenize_row`
(e.g. it does not correctly account for the case here). It predicted "survives" for row 3.
**An approximation of the library's truncation will always risk a gap.**

## The fix
Stop modeling truncation. Filter on **ground truth**: after the trainer tokenizes the
dataset, drop any row whose `chosen_labels` or `rejected_labels` has **0 non-`-100`
tokens**. This uses the exact tensors training will consume, so it cannot have a gap.
The cheap approximate pre-filter is kept as a first pass; the exact post-tokenization
filter is the guarantee. Provable instantly: after filtering, 0 zero-token rows remain.

## Hypotheses we tried and DISPROVED (do not revisit)
- **lr=5e-6 was the cause** — retracted; NaN'd at paper-exact 5e-7 too.
- **Empty-completion approx filter fixes it** — partial; it dropped 144 rows but MISSED
  the truncation-empties-both case (this row).
- **rsLoRA (α/√r) helps** — at fixed r=128 it is just an 11× scalar bump; made grad_norm
  worse; wrong lever. (bf16 has fp32's exponent range, so fp32/precision was never it.)
- **H_state / gradient explosion** — grad_norm was normal (≈18–34) right up to the NaN;
  the failure is a non-finite *forward loss*, so `max_grad_norm` was structurally irrelevant.
- **bf16 forward overflow → inf logits** — autopsy shows logits finite, 0 inf/nan. Wrong.

## Lessons
1. **Measure, don't guess.** Two days of assumption-driven guards failed; one capture of the
   exact (weights, batch) gave the answer immediately. Keep the capture/replay tooling.
2. **SimPO ≠ DPO on degenerate rows.** Averaging makes 0-token completions fatal (0/0).
3. **Non-Latin scripts inflate token counts massively** — they blow past `max_length` and
   trigger truncation edge cases. Length filtering must use the real tokenized labels.
4. **Operational:** never strip optimizer/rng from the *newest* checkpoint (a janitor sort
   bug did, destroying our cheap resume path and forcing a full re-run). Prefer
   `save_only_model` or a correct keep-latest rule.

## Artifacts
- Capture instrumentation: `scripts/train_simpo.py` (`NaNGuardCPOTrainer._capture_failure`)
- Capture config: `configs/simpo_qlora_capture.yaml`
- Replay harness: `scripts/replay_nan.py`
- Pre-cliff diagnostics: `scripts/diag_nan_batch.py`, `scripts/diag_nan_ckpt.py`
- Captured failure: `checkpoints/simpo-gbr05-capture/nan_capture/` (gitignored; local only)
- Resumable pre-cliff weights: `checkpoints/simpo-gbr05-capture/checkpoint-3600/` (local)
