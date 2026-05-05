# Money Table

| Model | `eval/rewards/accuracies` | MT-Bench | AlpacaEval 2 LC | Avg Gen Length | Length Δ vs SFT |
|---|---|---|---|---|---|
| Mistral-7B-v0.1 (base) | ~50% (random) | ~3–4 | <5% | — | — |
| SFT (UltraChat, QLoRA) | ~60% | ~6–6.5 | ~7% | TBD | 0% |
| DPO vanilla (UltraFeedback, QLoRA, default config) | TBD | TBD | TBD | TBD | TBD |
| SimPO (UltraFeedback, QLoRA, gbr=0.5) | TBD | TBD | TBD | TBD | TBD |
| SimPO sensitivity (UltraFeedback, QLoRA, gbr=0.3) | TBD | TBD | TBD | TBD | TBD |
| Zephyr-7B-β (published, full-FT) | ~78% | **7.34** | **13.2%** | — | — |

**How to read:**
- SFT→DPO gap = alignment effect
- DPO→SimPO gap on AlpacaEval 2 LC (not MT-Bench) = length-bias correction, quantified
- DPO/SimPO vs Zephyr-7B-β = QLoRA-vs-full-FT cost
