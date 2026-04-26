"""Stage 2 — DPO fine-tuning on HH-RLHF preference pairs.

Loads SFT adapter (from sft_train.py), merges it into the base model,
then trains a fresh DPO LoRA on preference pairs. Run with different
--beta values to study the KL-alignment tradeoff.

Usage:
    # Single run
    python dpo_train.py --sft-adapter ./mistral-sft-lora --beta 0.1

    # Sweep (run 3 times)
    for beta in 0.1 0.3 0.5; do
        python dpo_train.py --sft-adapter ./mistral-sft-lora --beta $beta
    done
"""

import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, PeftModel, prepare_model_for_kbit_training
from trl import DPOTrainer, DPOConfig

from data_utils import load_hh_rlhf

MODEL_ID = "unsloth/Ministral-3-8B-Base-2512-unsloth-bnb-4bit"

# ── Hyperparameters ────────────────────────────────────────────────────────────
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
LORA_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj"]

LEARNING_RATE = 5e-5
NUM_EPOCHS = 1
BATCH_SIZE = 2
GRAD_ACCUM = 8      # effective batch = 16
MAX_LENGTH = 512
MAX_PROMPT_LENGTH = 256

NUM_TRAIN = 10_000
NUM_EVAL = 1_000
# ──────────────────────────────────────────────────────────────────────────────


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--beta", type=float, default=0.1,
        help="KL penalty coefficient. Higher = stay closer to reference. Try 0.1, 0.3, 0.5."
    )
    parser.add_argument(
        "--sft-adapter", type=str, default=None,
        help="Path to SFT LoRA adapter to initialize from. If omitted, starts from base model."
    )
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = f"./mistral-dpo-beta{args.beta}"
    print(f"Beta={args.beta} | SFT adapter={args.sft_adapter} | Output={output_dir}")

    tokenizer = AutoTokenizer.from_pretrained(
        args.sft_adapter if args.sft_adapter else MODEL_ID
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model)

    if args.sft_adapter:
        # Merge SFT LoRA into base weights before applying DPO LoRA.
        # This mirrors the Zephyr recipe: base → SFT → DPO as separate LoRA stages.
        print(f"Merging SFT adapter from {args.sft_adapter}...")
        model = PeftModel.from_pretrained(model, args.sft_adapter)
        model = model.merge_and_unload()
        print("SFT adapter merged.")

    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=LORA_TARGET_MODULES,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    dataset = load_hh_rlhf(num_train=NUM_TRAIN, num_eval=NUM_EVAL)
    print(f"Train: {len(dataset['train'])} samples | Eval: {len(dataset['eval'])} samples")

    training_args = DPOConfig(
        output_dir=output_dir,
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
        beta=args.beta,
        max_length=MAX_LENGTH,
        max_prompt_length=MAX_PROMPT_LENGTH,
        remove_unused_columns=False,
        report_to="none",
    )

    trainer = DPOTrainer(
        model=model,
        ref_model=None,  # PEFT: reference = base with adapters disabled
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["eval"],
        processing_class=tokenizer,
    )

    trainer.train()
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"DPO adapter (beta={args.beta}) saved to {output_dir}")


if __name__ == "__main__":
    main()
