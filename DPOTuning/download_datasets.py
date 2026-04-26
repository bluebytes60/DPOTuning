"""DPO-2: Download and verify UltraChat 200K (SFT) and UltraFeedback cleaned (DPO).

Run with:
    conda run -n finetune python DPOTuning/download_datasets.py
"""

import os
from pathlib import Path
from datasets import load_dataset

HF_CACHE = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface"))


def _cache_size(name: str) -> str:
    cache_dir = HF_CACHE / "hub"
    total = sum(
        f.stat().st_size
        for d in cache_dir.glob(f"datasets--{name.replace('/', '--')}*")
        for f in d.rglob("*")
        if f.is_file()
    )
    return f"{total / 1e9:.2f} GB" if total >= 1e9 else f"{total / 1e6:.0f} MB"


def check_ultrachat():
    print("\n=== UltraChat 200K ===")
    ds = load_dataset("HuggingFaceH4/ultrachat_200k")
    print(f"Splits:  {list(ds.keys())}")
    print(f"Train:   {len(ds['train_sft']):,} examples")
    print(f"Test:    {len(ds['test_sft']):,} examples")
    print(f"Columns: {ds['train_sft'].column_names}")

    print("\n--- 3 spot-check examples (train_sft) ---")
    for i, ex in enumerate(ds["train_sft"].select(range(3))):
        turns = ex["messages"]
        print(f"[{i}] turns={len(turns)} | first_role={turns[0]['role']} | last_role={turns[-1]['role']}")
        print(f"     prompt[:80]: {turns[0]['content'][:80]!r}")

    size = _cache_size("HuggingFaceH4/ultrachat_200k")
    print(f"\nDisk usage: {size}")
    return ds


def check_ultrafeedback():
    print("\n=== UltraFeedback Binarized Cleaned ===")
    ds = load_dataset("argilla/ultrafeedback-binarized-preferences-cleaned")
    print(f"Splits:  {list(ds.keys())}")
    for split in ds:
        print(f"{split}: {len(ds[split]):,} examples")
    print(f"Columns: {ds['train'].column_names}")

    # Confirm chosen/rejected fields
    ex = ds["train"][0]
    assert "chosen" in ex, "missing 'chosen' field"
    assert "rejected" in ex, "missing 'rejected' field"
    print("\n--- 3 spot-check examples (train) ---")
    for i, ex in enumerate(ds["train"].select(range(3))):
        chosen_last = ex["chosen"][-1]["content"]
        rejected_last = ex["rejected"][-1]["content"]
        print(f"[{i}] chosen[:80]:   {chosen_last[:80]!r}")
        print(f"     rejected[:80]: {rejected_last[:80]!r}")

    size = _cache_size("argilla/ultrafeedback-binarized-preferences-cleaned")
    print(f"\nDisk usage: {size}")
    return ds


if __name__ == "__main__":
    print("=== DPO-2: Dataset Download & Verification ===")
    print(f"HF cache: {HF_CACHE}")

    check_ultrachat()
    check_ultrafeedback()

    print("\nAll checks passed. Ready for DPO-3 (SFT training).")
