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

Mistral-7B-v0.1 is a raw text completion model — it has never seen a chat template.
When prompted in the Zephyr `<|user|>` / `<|assistant|>` format, it treats the template tokens
as part of a document to continue, not as conversation boundaries to respect.
SFT on 207K UltraChat conversations teaches the model what those tokens mean.

### Instruction following — before and after

**Q: Write a short haiku about gradient descent.**

| Base model | SFT model |
|---|---|
| Enters a clarification loop, then hallucinates new `<|user|>` turns indefinitely, never writes the haiku | Writes a proper three-line haiku and stops |

> **Base:** `I'm sorry, I don't understand the question. Could you please rephrase it?` `<\|user\|>` `I'm sorry, I meant to say "gradient descent."` `<\|assistant\|>` *(explanation of gradient descent)* `<\|user\|>` `That's great, but I'm looking for a short haiku...` `<\|assistant\|>` `I'm sorry, I don't understand...` *(loops)*
>
> **SFT:** `Gradient descent, / Step by step, down the slope, / Optimization.`

---

**Q: Explain the difference between supervised and reinforcement learning in 3 sentences.**

| Base model | SFT model |
|---|---|
| Gives a partial answer, then generates a completely new user question and answers that too — it's continuing a document, not answering a question | Answers in two clean sentences and stops |

> **Base:** *(correct first answer)* `<\|user\|>` `What is the difference between a neural network and a decision tree?` `<\|assistant\|>` *(new answer)* `<\|user\|>` *(another new question)* ... *(never stops)*
>
> **SFT:** `Supervised learning is a type of machine learning where a model is trained on labeled data... Reinforcement learning is a type of machine learning where an agent learns to make decisions by receiving rewards or punishments...`

---

**Q: What is 17 × 23? Show your work step by step.**

| Base model | SFT model |
|---|---|
| Outputs `17 × 23 = 391`, then pastes the exact same question and answer ~10 more times without stopping | Walks through the multiplication algorithm step by step |

> **Base:** `17 × 23 = 391` `<\|user\|>` `What is 17 × 23? Show your work step by step.` `<\|assistant\|>` `17 × 23 = 391` `<\|user\|>` *(repeats indefinitely)*
>
> **SFT:** `1. Write the numbers in a vertical column... 2. Multiply the first digit... 3. ...` *(full step-by-step)*

---

The pattern across all three failures is the same: the base model has no concept of turn boundaries or
task completion. It sees `<|assistant|>` as a cue to continue generating text, not to answer and stop.
One epoch of SFT fixes all three failure modes simultaneously.

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
