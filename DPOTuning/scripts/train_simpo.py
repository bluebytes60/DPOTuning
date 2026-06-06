"""Stage 4 — SimPO QLoRA training (reference-free, length-normalized).

Uses TRL CPOTrainer with loss_type="simpo" (requires TRL >= 0.9.0).
Reference-free: no reference model loaded → ~5GB less VRAM vs vanilla DPO.

IMPORTANT: cpo_alpha=0.0 is REQUIRED for pure SimPO. Without it, TRL applies the
CPO objective term in addition to SimPO and you get a hybrid loss. The DPO-17
story warned about this exact bug — it's silent, the run completes, but the
comparison row is no longer "pure SimPO vs vanilla DPO."

Usage (A100, RunPod):
    # Paper defaults (gbr=0.5)
    python scripts/train_simpo.py --config configs/simpo_qlora.yaml

    # Sensitivity check (gbr=0.3)
    python scripts/train_simpo.py --config configs/simpo_qlora_gbr03.yaml

    # Override hparams for sweeps
    python scripts/train_simpo.py --config configs/simpo_qlora.yaml --beta 2.5 --simpo_gamma 1.25
"""

import argparse
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, PeftModel, PeftConfig
# TRL 0.29 moved CPO to trl.experimental.cpo (top-level export removed).
# requirements.txt pins trl==0.29.0, so import from the experimental path.
from trl.experimental.cpo import CPOTrainer, CPOConfig
from datasets import load_dataset
from omegaconf import OmegaConf


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    # CLI overrides for sweeps — set in config as default, override here
    parser.add_argument("--beta", type=float, default=None)
    parser.add_argument("--simpo_gamma", type=float, default=None)
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
    if args.simpo_gamma is not None:
        cfg.simpo_gamma = args.simpo_gamma
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

    # Load the BASE model, then apply + MERGE the SFT adapter so the SimPO LoRA
    # trains on a clean base+SFT model with a single trainable adapter.
    #
    # Do NOT pass the SFT checkpoint straight to from_pretrained: transformers 5.x
    # attaches the adapter via its native PeftAdapterMixin (a model attribute, not a
    # peft.PeftModel). That slips past CPOTrainer's "merge-and-unload first" guard
    # (trl.experimental.cpo.cpo_trainer raises only on isinstance(model, PeftModel))
    # and get_peft_model then STACKS a second adapter — an ambiguous double-adapter
    # state where the SFT start is no longer guaranteed in the forward pass.
    # merge_and_unload() bakes SFT into the base weights so the start is unambiguous.
    base_model_id = PeftConfig.from_pretrained(cfg.model_name_or_path).base_model_name_or_path

    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        quantization_config=bnb_config,
        device_map="auto",
        dtype=torch.bfloat16,
        attn_implementation=cfg.attn_implementation,
    )
    model = PeftModel.from_pretrained(base_model, cfg.model_name_or_path)
    model = model.merge_and_unload()
    model.config.use_cache = False
    return model, tokenizer


def format_dataset(ds, tokenizer):
    """Convert chosen/rejected from message lists to chat-templated strings.

    The argilla dataset stores chosen/rejected as lists of message dicts.
    CPOTrainer (like DPOTrainer) expects the *explicit prompt* format: plain
    strings where chosen/rejected are the COMPLETION only. CPOTrainer's
    build_tokenized_answer concatenates prompt+chosen itself, so putting the
    full conversation (prompt included) in chosen/rejected duplicates the
    prompt — fatal for SimPO, whose reward is normalized by completion length.
      prompt  -> formatted up to the last user turn (add_generation_prompt=True)
      chosen  -> chosen assistant response only (templated full minus prompt prefix)
      rejected-> rejected assistant response only
    """
    def format_row(example):
        chosen_msgs   = example["chosen"]
        rejected_msgs = example["rejected"]
        prompt_msgs   = chosen_msgs[:-1]

        prompt = tokenizer.apply_chat_template(
            prompt_msgs, tokenize=False, add_generation_prompt=True
        )
        chosen_full   = tokenizer.apply_chat_template(chosen_msgs,   tokenize=False)
        rejected_full = tokenizer.apply_chat_template(rejected_msgs, tokenize=False)

        # add_generation_prompt makes `prompt` an exact prefix of the full
        # templated conversation; slice it off to get the completion only.
        assert chosen_full.startswith(prompt) and rejected_full.startswith(prompt), (
            "templated prompt is not a prefix of the full conversation — chat "
            "template changed; completion slicing would be wrong"
        )
        example["prompt"]   = prompt
        example["chosen"]   = chosen_full[len(prompt):]
        example["rejected"] = rejected_full[len(prompt):]
        return example

    return ds.map(format_row, num_proc=4)


def _completion_survives_truncation(example, tokenizer, max_length, margin=2):
    """True if neither completion is emptied by TRL's prompt+response truncation.

    SimPO uses average_log_prob (length-normalized reward), so a ZERO-length
    completion is a 0/0 -> NaN that poisons grads and collapses the run to token
    soup. DPO is immune because it SUMS log-probs (empty -> 0, finite).

    The trigger lives in CPOTrainer's tokenize_row: it slices each response to
    `max_length - longer_response_length`. When the longer response exceeds
    max_length that bound goes negative and empties the *shorter* completion
    (and an exact == max_length empties both). We mirror that arithmetic here
    (+1 each for the BOS added to the prompt / EOS added to the answer, plus a
    small margin for tokenizer edge cases) and drop only the offenders (~0.1% of
    UltraFeedback) — keeping max_length=1024 for parity with the DPO run.
    """
    lp = len(tokenizer(example["prompt"],   add_special_tokens=False)["input_ids"]) + 1  # +BOS
    lc = len(tokenizer(example["chosen"],   add_special_tokens=False)["input_ids"]) + 1  # +EOS
    lr = len(tokenizer(example["rejected"], add_special_tokens=False)["input_ids"]) + 1  # +EOS
    longer = max(lc, lr)

    def kept(length):
        if lp + longer <= max_length:   # no truncation -> full response retained
            return length
        k = max_length - longer
        return min(length, k) if k >= 0 else max(0, length + k)

    return kept(lc) > margin and kept(lr) > margin


def load_data(cfg, tokenizer):
    dataset_id = list(cfg.dataset_mixer.keys())[0]
    splits = list(cfg.dataset_splits)
    train_split, eval_split = splits[0], splits[1]
    ds_train = load_dataset(dataset_id, split=train_split)
    ds_eval  = load_dataset(dataset_id, split=eval_split)
    ds_train = format_dataset(ds_train, tokenizer)
    ds_eval  = format_dataset(ds_eval,  tokenizer)

    # NaN-guard: drop rows whose truncation would empty a completion (SimPO 0/0).
    fn_kwargs = {"tokenizer": tokenizer, "max_length": cfg.max_length}
    for name, ds in (("train", ds_train), ("eval", ds_eval)):
        n0 = len(ds)
        kept = ds.filter(_completion_survives_truncation, num_proc=4, fn_kwargs=fn_kwargs)
        print(f"[NaN-guard] {name}: dropped {n0 - len(kept)}/{n0} empty-completion rows "
              f"-> {len(kept)} kept")
        if name == "train":
            ds_train = kept
        else:
            ds_eval = kept
    return ds_train, ds_eval


class NaNGuardCPOTrainer(CPOTrainer):
    """CPOTrainer that HALTS the moment the loss or any gradient goes non-finite,
    dumping the offending batch — instead of silently training on NaN for hours.

    The empty-completion filter in load_data removes ONE known NaN trigger. This
    catches any *other* trigger at the exact step it fires, so we can inspect the
    rows that caused it rather than guess. It checks both:
      - non-finite loss, and
      - non-finite gradients on a finite loss (the v2 run logged loss=7.537 with
        grad_norm=nan one step before collapse — a loss-only check would miss it).
    Detection happens AFTER super().training_step (forward+backward), so grads
    are populated. Cheap under LoRA: only adapter params carry grads.
    """

    def _dump_batch(self, inputs, reason):
        print(f"\n{'='*70}\n[NaN-guard] HALT at step {self.state.global_step}: {reason}\n{'='*70}")
        tok = self.processing_class
        for key in ("prompt_input_ids", "chosen_input_ids", "rejected_input_ids"):
            ids = inputs.get(key)
            if ids is None:
                continue
            print(f"\n--- {key} ({tuple(ids.shape)}) ---")
            for i, row in enumerate(ids):
                text = tok.decode(row[row != tok.pad_token_id], skip_special_tokens=False)
                print(f"  [row {i}] {text[:500]!r}")

    def training_step(self, model, inputs, num_items_in_batch=None):
        loss = super().training_step(model, inputs, num_items_in_batch=num_items_in_batch)

        if not torch.isfinite(loss).all():
            self._dump_batch(inputs, reason=f"non-finite loss = {loss.item()}")
            raise FloatingPointError("Non-finite SimPO loss; offending batch dumped above.")

        bad = [n for n, p in model.named_parameters()
               if p.requires_grad and p.grad is not None and not torch.isfinite(p.grad).all()]
        if bad:
            self._dump_batch(inputs, reason=f"finite loss={loss.item():.4f} but non-finite grad "
                                            f"in {len(bad)} params (first: {bad[0]})")
            raise FloatingPointError("Non-finite gradient on finite loss; offending batch dumped above.")

        return loss


def main():
    args = parse_args()
    cfg = load_config(args.config, args)

    gbr = cfg.simpo_gamma / cfg.beta
    print(f"SimPO: beta={cfg.beta}, simpo_gamma={cfg.simpo_gamma}, gbr={gbr:.2f}")
    print(f"        lr={cfg.learning_rate}, epochs={cfg.num_train_epochs}, cpo_alpha=0.0 (pure SimPO)")
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

    # cpo_alpha=0.0 disables the CPO term so the loss is pure SimPO.
    # Do NOT remove — see module docstring + DPO-17 Notion story.
    cpo_config = CPOConfig(
        output_dir=cfg.output_dir,
        loss_type=cfg.loss_type,
        cpo_alpha=0.0,
        beta=cfg.beta,
        simpo_gamma=cfg.simpo_gamma,
        num_train_epochs=cfg.num_train_epochs,
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        per_device_eval_batch_size=cfg.per_device_eval_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        gradient_checkpointing=cfg.gradient_checkpointing,
        gradient_checkpointing_kwargs=dict(cfg.gradient_checkpointing_kwargs),
        learning_rate=cfg.learning_rate,
        max_grad_norm=cfg.get("max_grad_norm", 1.0),
        lr_scheduler_type=cfg.lr_scheduler_type,
        warmup_ratio=cfg.warmup_ratio,
        bf16=cfg.bf16,
        do_eval=cfg.do_eval,
        eval_strategy=cfg.eval_strategy,
        eval_steps=cfg.eval_steps,
        logging_steps=cfg.logging_steps,
        max_length=cfg.max_length,
        # max_prompt_length removed: trl==0.29.0 CPOConfig dropped it (use max_length).
        optim=cfg.optim,
        save_strategy=cfg.save_strategy,
        save_steps=cfg.save_steps,
        save_total_limit=cfg.save_total_limit,
        report_to=cfg.report_to,
        seed=cfg.seed,
    )

    trainer = NaNGuardCPOTrainer(
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
