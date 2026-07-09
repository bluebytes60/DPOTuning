"""Stage 3 — DPO QLoRA training (vanilla DPO).

Usage (A100, RunPod):
    python scripts/train_dpo.py --config configs/dpo_vanilla_qlora.yaml
    python scripts/train_dpo.py --config configs/dpo_vanilla_qlora.yaml --beta 0.01
    python scripts/train_dpo.py --config configs/dpo_vanilla_qlora.yaml --beta 0.3 --num_train_epochs 1
"""

import argparse
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, PeftModel
from trl import DPOTrainer, DPOConfig
from datasets import load_dataset
from omegaconf import OmegaConf


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    # CLI overrides for sweeps — set in config as default, override here
    parser.add_argument("--beta", type=float, default=None)
    parser.add_argument("--learning_rate", type=float, default=None)
    parser.add_argument("--num_train_epochs", type=int, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--save_steps", type=int, default=None)
    parser.add_argument("--eval_steps", type=int, default=None)
    return parser.parse_args()


def load_config(path, args):
    cfg = OmegaConf.load(path)
    if args.beta is not None:
        cfg.beta = args.beta
    if args.learning_rate is not None:
        cfg.learning_rate = args.learning_rate
    if args.num_train_epochs is not None:
        cfg.num_train_epochs = args.num_train_epochs
    if args.output_dir is not None:
        cfg.output_dir = args.output_dir
    if args.save_steps is not None:
        cfg.save_steps = args.save_steps
    if args.eval_steps is not None:
        cfg.eval_steps = args.eval_steps
    return cfg


def build_model_and_tokenizer(cfg):
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name_or_path)
    tokenizer.pad_token = tokenizer.eos_token

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    ) if cfg.load_in_4bit else None

    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_name_or_path,
        quantization_config=bnb_config,
        device_map="auto",
        dtype=torch.bfloat16,
        attn_implementation=cfg.attn_implementation,
    )
    model.config.use_cache = False
    return model, tokenizer


def format_dataset(ds, tokenizer):
    """Convert chosen/rejected from message lists to chat-templated strings.

    The argilla dataset stores chosen/rejected as lists of message dicts.
    DPOTrainer 0.29 expects plain strings, so we apply the chat template here.
    prompt  -> formatted up to the last user turn (add_generation_prompt=True)
    chosen  -> full conversation including chosen assistant response
    rejected-> full conversation including rejected assistant response
    """
    def format_row(example):
        chosen_msgs   = example["chosen"]    # list of {role, content}
        rejected_msgs = example["rejected"]
        prompt_msgs   = chosen_msgs[:-1]     # everything except the final assistant turn

        example["prompt"]    = tokenizer.apply_chat_template(
            prompt_msgs, tokenize=False, add_generation_prompt=True
        )
        example["chosen"]    = tokenizer.apply_chat_template(chosen_msgs,   tokenize=False)
        example["rejected"]  = tokenizer.apply_chat_template(rejected_msgs, tokenize=False)
        return example

    return ds.map(format_row, num_proc=4)


def load_data(cfg, tokenizer):
    dataset_id = list(cfg.dataset_mixer.keys())[0]
    splits = list(cfg.dataset_splits)
    train_split, eval_split = splits[0], splits[1]
    ds_train = load_dataset(dataset_id, split=train_split)
    ds_eval  = load_dataset(dataset_id, split=eval_split)
    ds_train = format_dataset(ds_train, tokenizer)
    ds_eval  = format_dataset(ds_eval,  tokenizer)
    return ds_train, ds_eval


def main():
    args = parse_args()
    cfg = load_config(args.config, args)

    print(f"Config: beta={cfg.beta}, lr={cfg.learning_rate}, epochs={cfg.num_train_epochs}")
    print(f"Output: {cfg.output_dir}")

    model, tokenizer = build_model_and_tokenizer(cfg)
    ds_train, ds_eval = load_data(cfg, tokenizer)

    lora_config = LoraConfig(
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        target_modules=list(cfg.lora_target_modules),
        bias="none",
        task_type="CAUSAL_LM",
    )

    dpo_config = DPOConfig(
        output_dir=cfg.output_dir,
        beta=cfg.beta,
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
        optim=cfg.optim,
        save_strategy=cfg.save_strategy,
        **({} if OmegaConf.select(cfg, "save_steps") is None else {"save_steps": cfg.save_steps}),
        save_total_limit=cfg.save_total_limit,
        report_to=cfg.report_to,
        seed=cfg.seed,
    )

    trainer = DPOTrainer(
        model=model,
        args=dpo_config,
        train_dataset=ds_train,
        eval_dataset=ds_eval,
        processing_class=tokenizer,
        peft_config=lora_config,
    )
    trainer.train()
    trainer.save_model()


if __name__ == "__main__":
    main()
