"""Shared generation utilities for Mistral-7B + Zephyr chat template checkpoints.

All checkpoints in this project (SFT and DPO) use mistralai/Mistral-7B-v0.1 as
the base with the Zephyr chat template (<|user|> / <|assistant|> / <|system|>).
Those role markers are NOT single tokens in Mistral's SentencePiece vocabulary,
so eos_token_id alone will not stop generation when the model emits them. We
post-process the decoded string to truncate at the first role marker.
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

_ROLE_MARKERS = ["<|assistant|>", "<|user|>", "<|system|>"]


def load_model(base_model_id: str, checkpoint_path: str):
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
    model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model = PeftModel.from_pretrained(model, checkpoint_path)
    model.eval()
    return model, tokenizer


def _stop_token_ids(tokenizer) -> list[int]:
    """Return EOS + role-marker token IDs to pass as eos_token_id.

    Only works when a marker is a single token (e.g. after the tokenizer has
    added it as a special token). The post-decode strip in `generate` handles
    the multi-token case common with Mistral's base tokenizer.
    """
    stop = [tokenizer.eos_token_id]
    for tok in [*_ROLE_MARKERS, "</s>"]:
        tid = tokenizer.convert_tokens_to_ids(tok)
        if tid != tokenizer.unk_token_id:
            stop.append(tid)
    return list(set(stop))


def _strip_role_markers(text: str) -> str:
    for marker in _ROLE_MARKERS:
        idx = text.find(marker)
        if idx != -1:
            text = text[:idx]
    return text.strip()


@torch.inference_mode()
def generate(model, tokenizer, messages: list[dict], max_new_tokens: int) -> str:
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    prompt_len = inputs["input_ids"].shape[1]
    output_ids = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        eos_token_id=_stop_token_ids(tokenizer),
        pad_token_id=tokenizer.eos_token_id,
    )
    raw = tokenizer.decode(output_ids[0][prompt_len:], skip_special_tokens=True)
    return _strip_role_markers(raw)
