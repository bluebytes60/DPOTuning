from datasets import load_dataset, DatasetDict


def _split_hh_sample(text: str) -> tuple[str, str]:
    """Split HH-RLHF conversation string into (prompt, final_response).

    HH-RLHF format: "\n\nHuman: ...\n\nAssistant: ...\n\nHuman: ...\n\nAssistant: <response>"
    We split at the last "\n\nAssistant:" so the prompt includes that token
    and the response is everything after it.
    """
    marker = "\n\nAssistant:"
    idx = text.rfind(marker)
    if idx == -1:
        return text, ""
    return text[: idx + len(marker)], text[idx + len(marker) :].strip()


def load_hh_rlhf(
    num_train: int = 10_000,
    num_eval: int = 1_000,
    seed: int = 42,
) -> DatasetDict:
    """Load HH-RLHF as (prompt, chosen, rejected) triplets for DPO training."""
    raw = load_dataset("Anthropic/hh-rlhf")

    def process(example):
        prompt, chosen = _split_hh_sample(example["chosen"])
        _, rejected = _split_hh_sample(example["rejected"])
        return {"prompt": prompt, "chosen": chosen, "rejected": rejected}

    processed = raw.map(process, remove_columns=["chosen", "rejected"])
    processed = processed.filter(
        lambda x: len(x["chosen"]) > 0 and len(x["rejected"]) > 0
    )

    train = processed["train"].shuffle(seed=seed).select(range(num_train))
    eval_ds = processed["test"].shuffle(seed=seed).select(
        range(min(num_eval, len(processed["test"])))
    )
    return DatasetDict({"train": train, "eval": eval_ds})


def load_hh_rlhf_sft(
    num_train: int = 10_000,
    num_eval: int = 1_000,
    seed: int = 42,
) -> DatasetDict:
    """Load HH-RLHF as full chosen conversations for SFT training.

    Returns a DatasetDict with a single 'text' column containing the entire
    chosen conversation (Human + Assistant turns). SFTTrainer learns to continue
    the assistant voice given the full conversation context.
    """
    raw = load_dataset("Anthropic/hh-rlhf")

    def process(example):
        # Keep the full chosen conversation as-is — SFT trains on the whole sequence
        return {"text": example["chosen"].strip()}

    processed = raw.map(process, remove_columns=["chosen", "rejected"])
    processed = processed.filter(lambda x: len(x["text"]) > 0)

    train = processed["train"].shuffle(seed=seed).select(range(num_train))
    eval_ds = processed["test"].shuffle(seed=seed).select(
        range(min(num_eval, len(processed["test"])))
    )
    return DatasetDict({"train": train, "eval": eval_ds})
