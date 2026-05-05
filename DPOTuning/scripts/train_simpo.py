"""Stage 4 — SimPO QLoRA training (reference-free, length-normalized).

Uses TRL CPOTrainer with loss_type="simpo" (requires TRL >= 0.9.0).
Reference-free: no reference model loaded → ~5GB less VRAM vs vanilla DPO.

Usage (A100, RunPod):
    # Paper defaults (gbr=0.5)
    python scripts/train_simpo.py --config configs/simpo_qlora.yaml

    # Sensitivity check (gbr=0.3)
    python scripts/train_simpo.py --config configs/simpo_qlora_gbr03.yaml
"""

import argparse
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig
from trl import CPOTrainer, CPOConfig
from datasets import load_dataset
from omegaconf import OmegaConf


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output_dir", type=str, default=None)
    return parser.parse_args()


def load_config(path, args):
    cfg = OmegaConf.load(path)
    if args.output_dir is not None:
        cfg.output_dir = args.output_dir
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

    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_name_or_path,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        attn_implementation=cfg.attn_implementation,
    )
    model.config.use_cache = False
    return model, tokenizer


def load_data(cfg):
    dataset_id = list(cfg.dataset_mixer.keys())[0]
    splits = list(cfg.dataset_splits)
    ds_train = load_dataset(dataset_id, split=splits[0])
    ds_eval = load_dataset(dataset_id, split=splits[1])
    return ds_train, ds_eval


def main():
    args = parse_args()
    cfg = load_config(args.config, args)

    gbr = cfg.simpo_gamma / cfg.beta
    print(f"SimPO: beta={cfg.beta}, simpo_gamma={cfg.simpo_gamma}, gbr={gbr:.2f}")
    print(f"Output: {cfg.output_dir}")

    model, tokenizer = build_model_and_tokenizer(cfg)
    ds_train, ds_eval = load_data(cfg)

    lora_config = LoraConfig(
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        target_modules=list(cfg.lora_target_modules),
        bias="none",
        task_type="CAUSAL_LM",
    )

    cpo_config = CPOConfig(
        output_dir=cfg.output_dir,
        loss_type=cfg.loss_type,
        beta=cfg.beta,
        simpo_gamma=cfg.simpo_gamma,
        num_train_epochs=cfg.num_train_epochs,
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
        eval_steps=cfg.eval_steps,
        logging_steps=cfg.logging_steps,
        max_length=cfg.max_length,
        max_prompt_length=cfg.max_prompt_length,
        optim=cfg.optim,
        save_strategy=cfg.save_strategy,
        save_steps=cfg.save_steps,
        save_total_limit=cfg.save_total_limit,
        report_to=cfg.report_to,
        seed=cfg.seed,
    )

    trainer = CPOTrainer(
        model=model,
        args=cpo_config,
        train_dataset=ds_train,
        eval_dataset=ds_eval,
        processing_class=tokenizer,
        peft_config=lora_config,
    )
    trainer.train()
    trainer.save_model()


if __name__ == "__main__":
    main()
