"""Stage 1 — Supervised Fine-Tuning on HH-RLHF chosen responses.

This is the SFT baseline that DPO builds on top of.
Without this step, DPO applied to a base model performs poorly because
the base model doesn't know how to follow instructions at all.

Pipeline: base model → [this script] → mistral-sft-lora → [dpo_train.py] → mistral-dpo-lora
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig

from data_utils import load_hh_rlhf_sft

MODEL_ID = "unsloth/Ministral-3-8B-Base-2512-unsloth-bnb-4bit"
OUTPUT_DIR = "./mistral-sft-lora"

# ── Hyperparameters ────────────────────────────────────────────────────────────
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
LORA_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj"]

LEARNING_RATE = 2e-4
NUM_EPOCHS = 1
BATCH_SIZE = 4
GRAD_ACCUM = 4      # effective batch = 16
MAX_SEQ_LENGTH = 512

NUM_TRAIN = 10_000
NUM_EVAL = 1_000
# ──────────────────────────────────────────────────────────────────────────────


def main():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=LORA_TARGET_MODULES,
        bias="none",
        task_type="CAUSAL_LM",
    )

    dataset = load_hh_rlhf_sft(num_train=NUM_TRAIN, num_eval=NUM_EVAL)
    print(f"Train: {len(dataset['train'])} samples | Eval: {len(dataset['eval'])} samples")

    training_args = SFTConfig(
        output_dir=OUTPUT_DIR,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        learning_rate=LEARNING_RATE,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        bf16=True,
        optim="paged_adamw_8bit",
        logging_steps=25,
        eval_strategy="steps",
        eval_steps=200,
        save_strategy="steps",
        save_steps=200,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        max_seq_length=MAX_SEQ_LENGTH,
        packing=False,
        dataset_text_field="text",
        remove_unused_columns=True,
        report_to="none",
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["eval"],
        peft_config=lora_config,
        processing_class=tokenizer,
    )

    trainer.train()
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"SFT adapter saved to {OUTPUT_DIR}")
    print("Next step: python dpo_train.py --sft-adapter ./mistral-sft-lora --beta 0.1")


if __name__ == "__main__":
    main()
