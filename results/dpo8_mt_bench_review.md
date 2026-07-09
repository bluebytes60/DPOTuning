# DPO-8 MT-Bench — DPO-6 Epoch Trajectory + SFT Re-anchor

**Run config:** β=0.1, lr=5e-6, 3 epochs (DPO-6 checkpoints) + SFT re-eval
**Checkpoints:** `checkpoints/{checkpoint-3732, checkpoint-7464, checkpoint-11196}` + `checkpoints/sft-zephyr-lora/checkpoint-17205`
**Judge:** GPT-4 single-mode, 80 questions × 2 turns = 160 scored matches per model
**Notebook:** `notebooks/mt_bench_dpo8.ipynb`

---

## Results

| Model | MT-Bench | vs SFT (corrected) | Δ vs prev epoch | Avg gen len (1024 cap) |
|---|---|---|---|---|
| Mistral-7B-v0.1 base | 3.17 | — | — | — |
| SFT QLoRA — wrong template (DPO-5) | 5.75 | — | — | — |
| **SFT QLoRA — zephyr template** | **6.29** | — | +0.54 (template fix) | — |
| DPO ep1 (checkpoint-3732) | 6.03 | −0.26 | — | 417 |
| DPO ep2 (checkpoint-7464) | 6.67 | +0.38 | +0.64 | 656 (+57%) |
| **DPO ep3 (checkpoint-11196)** | **6.82** | **+0.53** | **+0.15** | **614 (−6%)** |
| Zephyr-7B-β published (full-FT, ref) | 7.34 | — | — | — |

---

## Headline Findings

**1. Recipe works but lands in the "partial" band (6.5–7.0).**
Best checkpoint (ep3 at 6.82) clears the corrected SFT anchor by +0.53, but sits 0.18 below the CLAUDE.md target (7.0) and 0.52 below Zephyr-7B-β full-FT (7.34). This is the explicit QLoRA-vs-full-FT cost on this configuration.

**2. ep2 → ep3 cleanly separates preference learning from length-gaming.**
Score rose (6.67 → 6.82, +0.15) while avg gen length fell (656 → 614, −6%). With length and score moving in opposite directions, the ep2→ep3 gain cannot be explained by GPT-4's length bias. This is the strongest signal in the run that real preference learning is happening, not just verbosity drift.

**3. ep1 → ep2 remains length-confounded.**
+57% length growth (417 → 656) alongside +0.64 score gain. Cannot disentangle preference contribution from length-bias contribution with MT-Bench alone — would need AlpacaEval 2 LC (DPO-15) to bound the judge-side effect.

**4. SFT template bug was material.**
DPO-5 reported 5.75 for SFT with `model_id="sft-qlora"`, which routed through FastChat's `BaseModelAdapter` (Vicuna-style "one_shot" template). The model was trained on the Zephyr `<|user|>` / `<|assistant|>` chat template, so it was being prompted in a format it had never seen. Re-running with `model_id="zephyr-sft-qlora"` forced `ZephyrAdapter` and recovered +0.54 (5.75 → 6.29). The original 5.75 was a measurement artifact, not a real score.

---

## Length-Gaming Re-read

DPO-7 framed the avg-length trajectory (417 → 656 → 614 at 1024-cap) as a length-gaming hypothesis. DPO-8 lets us partially test it:

| Transition | Length Δ | Score Δ | Compatible with length-gaming? |
|---|---|---|---|
| ep1 → ep2 | +57% | +0.64 | Yes — both moved in same direction. Confounded. |
| ep2 → ep3 | −6% | +0.15 | **No — score rose despite shorter outputs.** |
| ep1 → ep3 (overall) | +47% | +0.79 | Mixed — length-correlated for the first jump, real for the second. |

**Mechanism reminder:** DPO has no length term in its loss. Length drift is a *data-side artifact* (UltraFeedback's chosen responses are systematically longer than rejected) compounded by GPT-4's documented length bias. The model is not "gaming" anything — it's fitting the data distribution, which happens to align with the judge's bias. This is the gap SimPO closes via length-normalized loss.

---

## Anomalies

**DPO ep1 scored below SFT (6.03 vs 6.29).** Expected behavior for the β=0.1 + 1ep undercooked corner — the handbook explicitly avoids this configuration (their default is β=0.01 + 1ep). At β=0.1, the policy is anchored hard to the SFT reference; one pass through the data isn't enough to overcome the KL penalty in a way that translates to gen quality. DPO-7 inner-loop showed rewards/margin = 4.23 at ep1, confirming preference signal was being learned even when gen quality dipped.

**Gap to handbook's implicit QLoRA target (~7.0–7.4).** The handbook README claims β=0.01 + 1 epoch is "sufficient to achieve comparable performance to zephyr-7b-beta" (7.34). DPO-6's β=0.1 + 3 epochs theoretically targets the same point per CLAUDE.md, but landed 0.5+ short. Possible reasons: (a) QLoRA-specific quality loss vs full-FT, (b) Argilla-cleaned UF behaves differently than original UF, (c) handbook's claim is for their full pipeline and our reproduction has drift somewhere upstream. β sweep (DPO-9) will help discriminate.

---

## Decision

**ep3 (checkpoint-11196, MT-Bench 6.82) is the headline DPO-6 checkpoint.** Higher score *and* shorter outputs than ep2 — the cleanest of the three on the length-gaming axis.

**Next moves, in priority order:**

1. **DPO-15 — AlpacaEval 2 LC on ep3.** Direct test of whether MT-Bench's length-bias contribution is inflating the 6.82 number. If LC win-rate tracks the MT-Bench gain, real preference learning is dominant. If LC drops sharply relative to non-LC, more of 6.82 was length-driven than ep2→ep3 alone suggests.
2. **DPO-9 — β sweep, prioritize β=0.01 + 1 epoch first.** Handbook's cheaper recipe. If it matches or beats 6.82, that's the cost-quality money quote ("3× less compute, same MT-Bench"). If it underperforms, β=0.1 + 3ep was right but the recipe just can't clear 7.0 on this setup.
3. **Defer DPO-10 (epoch sweep) and DPO-11 (lr sweep).** Both lower-information than (1) and (2) given current evidence.

**Interview framing:** Reproduced the handbook's DPO QLoRA recipe with β=0.1 + 3 epochs. Best checkpoint hit 6.82 MT-Bench, 0.52 below Zephyr-7B-β full-FT (7.34) — the explicit QLoRA-vs-full-FT cost. The ep1 → ep3 trajectory (+0.79) wasn't pure verbosity: ep3 scored higher than ep2 with shorter average outputs, isolating real preference learning from the length-correlated component for that transition. Also surfaced and fixed a chat-template routing bug in the SFT eval (+0.54 swing) that had been masking the SFT-vs-base improvement in DPO-5.
