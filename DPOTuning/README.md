# DPO Tuning — Reproducing Zephyr-7B-β with QLoRA

Reproducing the [Zephyr-7B-β](https://huggingface.co/HuggingFaceH4/zephyr-7b-beta) recipe
(Mistral base → SFT on UltraChat → DPO on UltraFeedback) using QLoRA on a single consumer GPU.
Goal: quantify the QLoRA-vs-full-FT cost-quality tradeoff explicitly.

## Experimental Design

| Stage | Model | Data | Loss | Status |
|---|---|---|---|---|
| 1 — Baseline | Mistral-7B-v0.1 base | — | — | Done |
| 2 — SFT | LoRA adapter A | UltraChat 200K | Cross-entropy | **Done** (1 epoch, A100) |
| 3 — DPO | LoRA adapter B (init from A) | UltraFeedback Binarized Cleaned | DPO | Not started |

## SFT Results (qualitative)

Checkpoint: `sft-zephyr-lora/checkpoint-17205` (1 epoch, 207K examples, A100 80GB)

The base model (Mistral-7B-v0.1 with no instruction tuning) shows characteristic raw-completion failures
when prompted with the Zephyr chat template:

- **Haiku prompt** → infinite loop of `"I'm sorry, I don't understand."`
- **Math prompt** → gives the answer but then repeats the question indefinitely (no EOS)
- **RL explanation** → runs off into more and more questions without stopping

Post-SFT, all three prompts produce clean, well-formed responses that stop at the right place.
The improvement is obvious and consistent across prompt types.

Full before/after comparison: [`notebooks/sft_qlora_A100.ipynb`](notebooks/sft_qlora_A100.ipynb)
Local inference eval (4090, bf16, no quantization): [`notebooks/eval_sft_adapter.ipynb`](notebooks/eval_sft_adapter.ipynb)

## Stack

- Base model: `mistralai/Mistral-7B-v0.1`
- LoRA: r=16, alpha=16, 7 target modules (q/k/v/o + gate/up/down proj)
- Quantization: 4-bit NF4 (training) / bf16 (inference)
- Libraries: transformers, peft, trl, bitsandbytes, datasets
- Optimizer: AdamW, cosine LR, warmup_ratio=0.1
- Config source: `alignment-handbook/recipes/zephyr-7b-beta/sft/config_qlora.yaml`

## Key Datasets

| Resource | HF ID |
|---|---|
| SFT data | `HuggingFaceH4/ultrachat_200k` |
| DPO data | `argilla/ultrafeedback-binarized-preferences-cleaned` |
| Base model | `mistralai/Mistral-7B-v0.1` |

Using Argilla's cleaned UltraFeedback — the original HuggingFaceH4 version contains thousands of
incorrect GPT-4 preference labels.

## Evaluation Plan

**Inner loop (free):** held-out preference accuracy, reward margin, KL divergence, generation diagnostics  
**Outer loop (~$5–10/run):** MT-Bench (target: Zephyr published = 7.34), AlpacaEval 2 LC

## Hardware

RTX 4090 24GB (DPO training + inference eval) / A100 80GB (SFT training)
