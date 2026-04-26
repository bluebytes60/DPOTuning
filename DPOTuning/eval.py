"""Evaluate a trained adapter against the base and SFT baselines.

Produces the full comparison table from the experimental plan:

    Model                   | Pref Acc | Win Rate vs Base | Reward Score | KL from Ref
    Mistral-3-8B (base)     |  ~50%    |        —         |     X.X      |     0
    + SFT                   |   ??%    |       ??%        |     X.X      |    X.X
    + DPO β=0.1             |   ??%    |       ??%        |     X.X      |    X.X
    + DPO β=0.3             |   ??%    |       ??%        |     X.X      |    X.X
    + DPO β=0.5             |   ??%    |       ??%        |     X.X      |    X.X

Usage:
    # Evaluate a single adapter
    python eval.py --adapter ./mistral-dpo-beta0.1

    # Full comparison table (base + SFT + three DPO betas)
    python eval.py --compare \\
        base \\
        ./mistral-sft-lora \\
        ./mistral-dpo-beta0.1 \\
        ./mistral-dpo-beta0.3 \\
        ./mistral-dpo-beta0.5

    # Enable LLM-as-judge (requires ANTHROPIC_API_KEY env var, costs ~$1-2)
    python eval.py --adapter ./mistral-dpo-beta0.1 --llm-judge
"""

import argparse
import os
import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from peft import PeftModel

from data_utils import load_hh_rlhf

BASE_MODEL_ID = "unsloth/Ministral-3-8B-Base-2512-unsloth-bnb-4bit"
REWARD_MODEL_ID = "OpenAssistant/reward-model-deberta-v3-large-v2"
NUM_EVAL_SAMPLES = 500
NUM_GEN_EXAMPLES = 5


# ── Model loading ──────────────────────────────────────────────────────────────

def load_model(adapter_path: str | None):
    """Load base model, optionally with a LoRA adapter merged on top."""
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID, device_map="auto", torch_dtype=torch.bfloat16
    )
    tokenizer = AutoTokenizer.from_pretrained(
        adapter_path if adapter_path else BASE_MODEL_ID
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if adapter_path and adapter_path != "base":
        model = PeftModel.from_pretrained(model, adapter_path)
        model = model.merge_and_unload()

    model.eval()
    return model, tokenizer


# ── Core scoring ───────────────────────────────────────────────────────────────

def _avg_log_prob(model, tokenizer, prompt: str, response: str, device: str) -> float:
    """Average token log-probability of `response` conditioned on `prompt`."""
    full = prompt + response
    enc = tokenizer(full, return_tensors="pt", truncation=True, max_length=512).to(device)
    prompt_len = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)[
        "input_ids"
    ].shape[1]

    with torch.inference_mode():
        logits = model(**enc).logits

    log_probs = torch.log_softmax(logits[0, :-1], dim=-1)
    token_ids = enc["input_ids"][0, 1:]

    start = max(prompt_len - 1, 0)
    if start >= len(token_ids):
        return 0.0
    return log_probs[start:].gather(1, token_ids[start:].unsqueeze(1)).squeeze().mean().item()


def _kl_divergence(model, ref_model, tokenizer, prompt: str, response: str, device: str) -> float:
    """KL(model || ref_model) averaged over response tokens."""
    full = prompt + response
    enc = tokenizer(full, return_tensors="pt", truncation=True, max_length=512).to(device)
    prompt_len = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)[
        "input_ids"
    ].shape[1]

    with torch.inference_mode():
        model_logits = model(**enc).logits
        ref_logits = ref_model(**enc).logits

    start = max(prompt_len - 1, 0)
    seq_len = model_logits.shape[1] - 1
    if start >= seq_len:
        return 0.0

    model_lp = torch.log_softmax(model_logits[0, start:seq_len], dim=-1)
    ref_lp = torch.log_softmax(ref_logits[0, start:seq_len], dim=-1)
    # KL(p||q) = sum p * (log p - log q)
    model_p = model_lp.exp()
    kl = (model_p * (model_lp - ref_lp)).sum(dim=-1).mean().item()
    return max(kl, 0.0)  # numerical stability


# ── Metrics ────────────────────────────────────────────────────────────────────

def preference_accuracy(model, tokenizer, dataset, n: int, device: str) -> float:
    """% of pairs where the model assigns higher avg log-prob to chosen vs rejected.

    Random baseline = 50%. A well-trained DPO model typically reaches 65-80%.
    """
    correct, total = 0, min(n, len(dataset))
    for i in range(total):
        ex = dataset[i]
        lp_c = _avg_log_prob(model, tokenizer, ex["prompt"], ex["chosen"], device)
        lp_r = _avg_log_prob(model, tokenizer, ex["prompt"], ex["rejected"], device)
        correct += int(lp_c > lp_r)
        if (i + 1) % 100 == 0:
            print(f"  pref_acc {i+1}/{total}: {correct/(i+1):.3%}")
    return correct / total


def mean_kl_from_ref(model, ref_model, tokenizer, dataset, n: int, device: str) -> float:
    """Mean KL divergence from reference (base) model over chosen responses.

    Higher KL = model has drifted further from the base. DPO with low beta
    should show low KL; high beta should show even lower KL.
    """
    kls, total = [], min(n, len(dataset))
    for i in range(total):
        ex = dataset[i]
        kl = _kl_divergence(model, ref_model, tokenizer, ex["prompt"], ex["chosen"], device)
        kls.append(kl)
    return float(np.mean(kls))


def reward_model_score(dataset, model_outputs: list[str], n: int) -> float:
    """Score responses using a pretrained reward model.

    Uses OpenAssistant/reward-model-deberta-v3-large-v2 as an external judge.
    Higher score = more preferred by the reward model.
    Requires ~2GB additional VRAM.
    """
    print(f"  Loading reward model ({REWARD_MODEL_ID})...")
    rm_pipe = pipeline("text-classification", model=REWARD_MODEL_ID, device=0)
    scores = []
    for i in range(min(n, len(model_outputs))):
        text = dataset[i]["prompt"] + model_outputs[i]
        result = rm_pipe(text, truncation=True, max_length=512)
        scores.append(result[0]["score"])
    return float(np.mean(scores))


def generate_responses(model, tokenizer, dataset, n: int, device: str) -> list[str]:
    """Generate model responses for n prompts."""
    responses = []
    model.eval()
    for i in range(min(n, len(dataset))):
        inputs = tokenizer(
            dataset[i]["prompt"], return_tensors="pt", truncation=True, max_length=256
        ).to(device)
        with torch.inference_mode():
            out = model.generate(
                **inputs,
                max_new_tokens=128,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        response = tokenizer.decode(
            out[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True
        )
        responses.append(response)
    return responses


def llm_judge_win_rate(
    base_responses: list[str],
    model_responses: list[str],
    prompts: list[str],
) -> float:
    """Pairwise win rate vs base model using Claude as judge.

    Requires ANTHROPIC_API_KEY environment variable.
    Costs ~$1-2 for 200 comparisons with claude-haiku-4-5.
    """
    import anthropic
    client = anthropic.Anthropic()

    wins, ties, losses = 0, 0, 0
    for i, (prompt, base_r, model_r) in enumerate(
        zip(prompts, base_responses, model_responses)
    ):
        # Randomize A/B order to reduce position bias
        if i % 2 == 0:
            a, b, a_is_model = base_r, model_r, False
        else:
            a, b, a_is_model = model_r, base_r, True

        msg = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=10,
            messages=[{
                "role": "user",
                "content": (
                    f"Context: {prompt[-300:]}\n\n"
                    f"Response A: {a[:300]}\n\n"
                    f"Response B: {b[:300]}\n\n"
                    "Which response is more helpful and appropriate? "
                    "Reply with exactly one word: A, B, or Tie."
                ),
            }],
        )
        verdict = msg.content[0].text.strip().upper()
        if verdict == "TIE":
            ties += 1
        elif (verdict == "A" and a_is_model) or (verdict == "B" and not a_is_model):
            wins += 1
        else:
            losses += 1

        if (i + 1) % 50 == 0:
            total_so_far = wins + ties + losses
            print(f"  judge {i+1}: win={wins/total_so_far:.1%} tie={ties/total_so_far:.1%}")

    total = wins + ties + losses
    return wins / total


# ── Print helpers ──────────────────────────────────────────────────────────────

def print_examples(dataset, responses: list[str], label: str):
    for i in range(min(NUM_GEN_EXAMPLES, len(responses))):
        ex = dataset[i]
        print(f"\n{'='*60}  [{label}] Example {i+1}")
        print(f"PROMPT:\n{ex['prompt'][-300:]}")
        print(f"\nCHOSEN:   {ex['chosen'][:200]}")
        print(f"REJECTED: {ex['rejected'][:200]}")
        print(f"GENERATED: {responses[i][:200]}")


# ── Main ───────────────────────────────────────────────────────────────────────

def eval_single(adapter_path: str | None, use_llm_judge: bool, ref_model=None, ref_tokenizer=None):
    label = adapter_path if adapter_path else "base"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"\n{'─'*60}")
    print(f"Evaluating: {label}")
    model, tokenizer = load_model(adapter_path)

    dataset = load_hh_rlhf(num_train=10, num_eval=NUM_EVAL_SAMPLES)
    eval_ds = dataset["eval"]

    print(f"Computing preference accuracy ({NUM_EVAL_SAMPLES} samples)...")
    pref_acc = preference_accuracy(model, tokenizer, eval_ds, NUM_EVAL_SAMPLES, device)
    print(f"  Preference accuracy: {pref_acc:.3%}  (random baseline: 50.0%)")

    kl = 0.0
    if ref_model is not None:
        print("Computing KL divergence from base model (200 samples)...")
        kl = mean_kl_from_ref(model, ref_model, tokenizer, eval_ds, 200, device)
        print(f"  Mean KL from base: {kl:.4f}")

    print(f"Generating {NUM_GEN_EXAMPLES} example responses...")
    responses = generate_responses(model, tokenizer, eval_ds, NUM_GEN_EXAMPLES, device)
    print_examples(eval_ds, responses, label)

    win_rate = None
    if use_llm_judge and ref_model is not None:
        if not os.environ.get("ANTHROPIC_API_KEY"):
            print("  Skipping LLM judge: ANTHROPIC_API_KEY not set")
        else:
            print("Running LLM-as-judge pairwise comparison vs base (200 prompts)...")
            base_responses = generate_responses(ref_model, ref_tokenizer, eval_ds, 200, device)
            model_responses = generate_responses(model, tokenizer, eval_ds, 200, device)
            prompts = [eval_ds[i]["prompt"] for i in range(200)]
            win_rate = llm_judge_win_rate(base_responses, model_responses, prompts)
            print(f"  Win rate vs base: {win_rate:.3%}")

    return {"label": label, "pref_acc": pref_acc, "kl": kl, "win_rate": win_rate}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter", type=str, default=None,
                        help="Path to a single adapter to evaluate (or 'base').")
    parser.add_argument("--compare", nargs="+", default=None,
                        help="List of adapter paths (or 'base') to compare in a table.")
    parser.add_argument("--llm-judge", action="store_true",
                        help="Run Claude pairwise win-rate eval (requires ANTHROPIC_API_KEY).")
    args = parser.parse_args()

    targets = args.compare if args.compare else [args.adapter]

    # Load base model once for KL reference
    print("Loading base model for KL reference...")
    ref_model, ref_tokenizer = load_model(None)

    results = []
    for target in targets:
        path = None if target == "base" else target
        r = eval_single(path, args.llm_judge, ref_model, ref_tokenizer)
        results.append(r)

    # Summary table
    print(f"\n{'='*70}")
    print(f"{'Model':<35} {'Pref Acc':>10} {'KL from Base':>14} {'Win Rate':>10}")
    print(f"{'─'*35} {'─'*10} {'─'*14} {'─'*10}")
    for r in results:
        win_str = f"{r['win_rate']:.3%}" if r["win_rate"] is not None else "     —"
        print(f"{r['label']:<35} {r['pref_acc']:>10.3%} {r['kl']:>14.4f} {win_str:>10}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
