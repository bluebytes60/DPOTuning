"""Stage 2 — SFT QLoRA training.

Usage (A100, RunPod):
    python scripts/train_sft.py --config configs/sft_qlora.yaml
    python scripts/train_sft.py --config configs/sft_qlora.yaml --max_steps 500  # smoke test
"""

import argparse
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig
from datasets import load_dataset
from omegaconf import OmegaConf


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--max_steps", type=int, default=-1)
    return parser.parse_args()


def load_config(path):
    cfg = OmegaConf.load(path)
    return cfg


def build_model_and_tokenizer(cfg):
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=cfg.load_in_4bit,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name_or_path)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    if hasattr(cfg, "chat_template"):
        tokenizer.chat_template = cfg.chat_template

    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_name_or_path,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        attn_implementation=cfg.attn_implementation,
    )
    model.config.use_cache = False
    return model, tokenizer


def build_lora(model, cfg):
    model = prepare_model_for_kbit_training(model)
    lora_config = LoraConfig(
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        target_modules=list(cfg.lora_target_modules),
        bias="none",
        task_type="CAUSAL_LM",
    )
    return get_peft_model(model, lora_config)


def load_data(cfg, tokenizer):
    # TODO: support dataset_mixture config key (multi-dataset)
    raw_train = load_dataset("HuggingFaceH4/ultrachat_200k", split="train_sft")
    raw_eval = load_dataset("HuggingFaceH4/ultrachat_200k", split="test_sft")
    raw_eval = raw_eval.select(range(cfg.get("test_split_size", 1000)))

    def fmt(batch):
        return {
            "text": [
                tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
                for msgs in batch["messages"]
            ]
        }

    ds_train = raw_train.map(fmt, batched=True, num_proc=cfg.dataset_num_proc, remove_columns=raw_train.column_names)
    ds_eval = raw_eval.map(fmt, batched=True, num_proc=cfg.dataset_num_proc, remove_columns=raw_eval.column_names)
    return ds_train, ds_eval


def main():
    args = parse_args()
    cfg = load_config(args.config)

    model, tokenizer = build_model_and_tokenizer(cfg)
    model = build_lora(model, cfg)
    model.print_trainable_parameters()

    ds_train, ds_eval = load_data(cfg, tokenizer)

    sft_config = SFTConfig(
        output_dir=cfg.output_dir,
        num_train_epochs=cfg.num_train_epochs,
        max_steps=args.max_steps if args.max_steps > 0 else cfg.get("max_steps", -1),
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        per_device_eval_batch_size=cfg.per_device_eval_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        gradient_checkpointing=cfg.gradient_checkpointing,
        gradient_checkpointing_kwargs=dict(cfg.gradient_checkpointing_kwargs),
        learning_rate=cfg.learning_rate,
        lr_scheduler_type=cfg.lr_scheduler_type,
        warmup_ratio=cfg.warmup_ratio,
        bf16=cfg.bf16,
        do_eval=cfg.do_eval,
        eval_strategy=cfg.eval_strategy,
        logging_steps=cfg.logging_steps,
        save_strategy=cfg.save_strategy,
        save_steps=cfg.save_steps,
        save_total_limit=cfg.save_total_limit,
        report_to=cfg.report_to,
        max_length=cfg.max_seq_length,
        dataset_text_field="text",
        packing=True,
        dataloader_num_workers=4,
        use_liger_kernel=cfg.get("use_liger_kernel", False),
        seed=cfg.seed,
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=ds_train,
        eval_dataset=ds_eval,
        processing_class=tokenizer,
    )
    trainer.train()
    trainer.save_model()


if __name__ == "__main__":
    main()
