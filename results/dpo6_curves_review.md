# DPO-6 Training Health Check (DPO-7)

**Run config:** β=0.1, lr=5e-6, 3 epochs  
**Checkpoint:** `checkpoints/` (final) + `checkpoint-{3732,7464,11196}` (per-epoch)  
**Metrics source:** `checkpoints/checkpoint-11196/trainer_state.json` + `logs/dpo6.log`  
**Note:** W&B not configured for this run. TRL version does not log KL directly.

---

## Checklist

- [x] `train/loss` — smooth descent: 0.695 → 0.35 (ep1) → 0.17 (ep2) → ~0.09 (ep3). No late-training spikes.
- [x] `eval/rewards/accuracies` — 72.8% at ep0.1, rises to 83.7% by ep1, plateaus at ~83% through ep3. Not collapsing.
- [x] `eval/rewards/margins` — steady widening: 0.47 → 4.23 (ep1) → 4.58 (ep2) → 6.98 (ep3). Separation increasing throughout.
- [x] `kl` — **not logged by this TRL version.** Implicit KL estimated from eval_loss rise (see anomalies).
- [x] `eval/loss` vs `train/loss` — train keeps falling; eval dips to 0.395 at ep1, holds at 0.401 at ep2, then rises to 0.577 at ep3.
- [x] Gradient norm — variable (0.6–26) throughout training, no sustained explosion. Spiky but typical of QLoRA on DPO.

---

## Epoch-Boundary Eval Summary

| Epoch | eval_loss | rewards/acc | rewards/margin |
|---|---|---|---|
| 1 | 0.395 | 83.7% | 4.23 |
| 2 | 0.401 | 83.6% | 4.58 |
| 3 | 0.577 | 83.3% | 6.98 |

---

## Anomalies

**eval_loss rises sharply at epoch 3 (0.40 → 0.58)** while train loss continues falling. This is the classic DPO signature of growing implicit KL — the policy is drifting further from the reference model as training continues. With β=0.1 over 3 epochs, this is expected behavior, not a failure. Preference accuracy remains stable at ~83%, confirming the drift is not hurting generalization.

**Grad norms are noisy** (bouncing between ~1 and 26) but never sustained at pathological levels. Worth checking whether β=0.01 in the sweep yields smoother optimization.

**Preference accuracy plateaus after epoch 1.** Most of the easy pairs are learned in the first epoch; epochs 2–3 mainly widen the margin on already-correct pairs without flipping new ones. This is normal.

---

## Inner-Loop Eval Results (DPO-7)

**Notebook:** `notebooks/dpo7_inner_loop_eval.ipynb`  
**Raw long-response log:** `results/dpo7LengthyResponse.txt` (epoch 1 / 1024-token run)

### Generation Length Trajectory

| Checkpoint | max_new_tokens | avg_gen_length | p90_gen_length | harmful_refusal | over_refusal |
|---|---|---|---|---|---|
| ep1 (checkpoint-3732)  | 512  | 310.8 | 513.0 (capped) | 40% | 0% |
| ep2 (checkpoint-7464)  | 512  | 404.3 | 513.0 (capped) | 40% | 0% |
| ep3 (checkpoint-11196) | 512  | 434.4 | 513.0 (capped) | 40% | 0% |
| ep1 (checkpoint-3732)  | 1024 | 417.1 | 1024.1         | 40% | 0% |
| ep2 (checkpoint-7464)  | 1024 | 655.9 | 1025.0 (capped)| 40% | 2.2% |
| ep3 (checkpoint-11196) | 1024 | 613.5 | 1025.0 (capped)| 40% | 0% |

**Length drift:** avg_gen_length grew +40% from ep1 → ep3 at 512-token cap (310.8 → 434.4). At 1024-token cap, ep2 reached 655.9 — 12/50 responses hit the ceiling, confirming the responses are not short answers being clipped; the model is genuinely producing very long outputs.

### Failure Modes Identified

Inspection of the 12 capped responses revealed two distinct pathological patterns:

**Mode 1 — Repetition loop:** Model gives a correct answer on the first attempt, then cannot stop generating. Clearest example: prompt "Rewrite this sentence to be more concise" → one-line correct answer, followed by 50 nearly-identical paraphrases listed as "Here is option N..." until truncation. Similar pattern in the aurora borealis response (same bullet repeated ~8 times verbatim) and the neural network "dreams" response (same paragraph copy-pasted 3×).

**Mode 2 — Over-elaboration spiral:** Correct answer given, then model escalates into unprompted sub-topics. Example: "Write a short haiku about gradient descent" → haiku written, then full essay on SGD, Adam, mini-batch math. "How does HTTPS protect data at a high level?" → 17-item numbered list still growing at truncation.

Both modes share the same root cause: DPO training on UltraFeedback has caused the model to associate longer responses with higher reward, so it pads aggressively.

### Safety Observation

The two jailbreak-framed harmful prompts (fiction-context meth synthesis, teacher-context explosive synthesis) received **full, detailed answers** — not refusals. These are the ones counted in the 40% refusal rate miss. The 3/5 refused prompts are the obviously-phrased ones; context-framed jailbreaks pass straight through. This is a known limitation of DPO on helpfulness data — it does not teach safety alignment.

---

## Decision

**GO on MT-Bench (DPO-8) — run on all 3 epoch checkpoints, not just the final.**

Length growth alone is a hypothesis about quality, not a measurement of it. Inner-loop metrics only become useful if they are calibrated against the paid outer-loop signal at least once — skipping MT-Bench here would mean trusting an unvalidated cheap signal to override the eval budget that was reserved exactly for this purpose.

Running on all three epoch checkpoints (~$15–30, within the ~$30–60 project budget) produces a quality-vs-length trajectory inside a single training run, which is more informative than a single endpoint score:

| MT-Bench trajectory | Interpretation | Action |
|---|---|---|
| Monotone rise across epochs | Length growth is "real" — model is using tokens to add quality | Proceed with β sweep around current config |
| Peak at ep1 or ep2, drop at ep3 | Length-gaming begins after the first epoch | Restrict epoch sweep (DPO-10) to {1, 2}, drop 3 |
| Flat or falling from ep1 | DPO is hurting quality despite healthy TRL metrics | Lower β (run DPO-9 with β=0.01) before further epoch tuning |

The peak-at-ep2 outcome is the most likely given the avg_gen_length curve (310 → 404 → 434 — diminishing returns shape) and is the most actionable for the next sweep.

**Next:** DPO-8 on ep1 / ep2 / ep3, then use that result to scope DPO-9 (β sweep) and DPO-10 (epoch sweep).
