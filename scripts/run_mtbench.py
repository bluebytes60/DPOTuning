"""MT-Bench (option B): generate with OUR client (correct Zephyr template + SFT-merge),
judge with the IDENTICAL fschat GPT-4 single-grade prompts. No fschat/merge/disk needed.

Matches DPO-8/DPO-17 methodology:
  - 80 questions x 2 turns
  - per-category sampling temperature (fschat temperature_config)
  - reference answers for math/reasoning/coding (NEED_REF_CATS)
  - judge = gpt-4, single mode, parse "[[rating]]", score = mean of 160 turns

Data (downloaded from lm-sys/FastChat): data/mtbench/{question,judge_prompts,reference_gpt4}.jsonl

Run (per checkpoint):
  export OPENAI_API_KEY=...   # from .env openAIKey
  HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1 PYTHONPATH=. \
    python scripts/run_mtbench.py \
      --checkpoint checkpoints/simpo-gbr05-fixed/checkpoint-3723 \
      --sft_adapter checkpoints/sft-zephyr-lora/checkpoint-17205 \
      --model_id zephyr-simpo-ep1 --run_id dpo17_simpo_ep1 \
      --beta 2.0 --epochs 3 --lr 5e-6 --lora_r 128 --simpo_gamma 1.0
"""
import argparse
import csv
import json
import os
import re
import statistics
import time
from pathlib import Path

import torch
from openai import OpenAI

from scripts.generation import load_model, _stop_token_ids, _strip_role_markers

DATA = Path("data/mtbench")
ANSWER_DIR = Path("results/mt_bench/model_answer")
JUDGE_DIR = Path("results/mt_bench/model_judgment")
RUNS_CSV = Path("results/runs.csv")

NEED_REF_CATS = {"math", "reasoning", "coding"}
TEMPERATURE_CONFIG = {
    "writing": 0.7, "roleplay": 0.7, "extraction": 0.0, "math": 0.0,
    "coding": 0.0, "reasoning": 0.0, "stem": 0.1, "humanities": 0.1,
}
JUDGE_MODEL = "gpt-4"          # match DPO-8/DPO-17
MAX_NEW_TOKENS = 1024
RATING_RE = re.compile(r"\[\[(\d+\.?\d*)\]\]")
RATING_RE_FALLBACK = re.compile(r"\[(\d+\.?\d*)\]")

CSV_FIELDNAMES = [
    "run_id", "checkpoint", "tag", "stage", "beta", "epochs", "lr", "lora_r",
    "simpo_gamma", "max_new_tokens", "avg_gen_length", "p90_gen_length",
    "harmful_refusal_rate", "over_refusal_rate", "pref_acc", "mt_bench",
    "alpacaeval2_lc", "notes",
]


# ---------- generation (our client; per-category temperature) ----------
@torch.inference_mode()
def gen_turn(model, tok, messages, temperature):
    prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tok(prompt, return_tensors="pt").to(model.device)
    plen = inputs["input_ids"].shape[1]
    do_sample = temperature > 1e-4
    kw = dict(max_new_tokens=MAX_NEW_TOKENS, eos_token_id=_stop_token_ids(tok),
              pad_token_id=tok.eos_token_id, do_sample=do_sample)
    if do_sample:
        kw.update(temperature=temperature, top_p=1.0)
    out = model.generate(**inputs, **kw)
    return _strip_role_markers(tok.decode(out[0][plen:], skip_special_tokens=True))


def generate_answers(model, tok, questions, model_id):
    ANSWER_DIR.mkdir(parents=True, exist_ok=True)
    out_path = ANSWER_DIR / f"{model_id}.jsonl"
    if out_path.exists() and sum(1 for _ in open(out_path)) >= len(questions):
        print(f"[skip gen] {out_path} already complete")
        return out_path
    torch.manual_seed(0)
    with open(out_path, "w") as f:
        for i, q in enumerate(questions):
            temp = TEMPERATURE_CONFIG.get(q["category"], 0.7)
            a1 = gen_turn(model, tok, [{"role": "user", "content": q["turns"][0]}], temp)
            a2 = gen_turn(model, tok, [
                {"role": "user", "content": q["turns"][0]},
                {"role": "assistant", "content": a1},
                {"role": "user", "content": q["turns"][1]},
            ], temp)
            f.write(json.dumps({"question_id": q["question_id"], "category": q["category"],
                                "model_id": model_id, "choices": [{"index": 0, "turns": [a1, a2]}]}) + "\n")
            f.flush()
            print(f"  [{i+1}/{len(questions)}] q{q['question_id']} ({q['category']}, T={temp}) "
                  f"len=({len(a1)},{len(a2)})", flush=True)
    return out_path


# ---------- judging (identical fschat single-grade prompts) ----------
def parse_rating(text):
    m = RATING_RE.search(text) or RATING_RE_FALLBACK.search(text)
    return float(m.group(1)) if m else -1.0


def judge_turn(client, jp, q, ans, turn, ref=None):
    cat = q["category"]
    use_ref = cat in NEED_REF_CATS and ref is not None
    if turn == 0:
        name = "single-math-v1" if use_ref else "single-v1"
        kw = {"question": q["turns"][0], "answer": ans}
        if use_ref:
            kw["ref_answer_1"] = ref[0]
    else:
        name = "single-math-v1-multi-turn" if use_ref else "single-v1-multi-turn"
        kw = {"question_1": q["turns"][0], "answer_1": ans[0],
              "question_2": q["turns"][1], "answer_2": ans[1]}
        if use_ref:
            kw["ref_answer_1"], kw["ref_answer_2"] = ref[0], ref[1]
    tmpl = jp[name]
    user_prompt = tmpl["prompt_template"].format(**kw)
    for attempt in range(4):
        try:
            r = client.chat.completions.create(
                model=JUDGE_MODEL, temperature=0, max_tokens=2048,
                messages=[{"role": "system", "content": tmpl["system_prompt"]},
                          {"role": "user", "content": user_prompt}])
            txt = r.choices[0].message.content
            return parse_rating(txt), txt
        except Exception as e:
            print(f"    judge retry {attempt+1}: {e}", flush=True)
            time.sleep(5 * (attempt + 1))
    return -1.0, "JUDGE_FAILED"


def judge_all(answers_path, questions, refs, jp, model_id):
    JUDGE_DIR.mkdir(parents=True, exist_ok=True)
    jpath = JUDGE_DIR / f"{model_id}_gpt4_single.jsonl"
    client = OpenAI()
    answers = {json.loads(l)["question_id"]: json.loads(l) for l in open(answers_path)}
    qmap = {q["question_id"]: q for q in questions}
    scores = []
    with open(jpath, "w") as f:
        for qid, a in answers.items():
            q = qmap[qid]
            turns = a["choices"][0]["turns"]
            ref = refs.get(qid)
            s1, t1 = judge_turn(client, jp, q, turns[0], 0, ref)
            s2, t2 = judge_turn(client, jp, q, turns, 1, ref)
            for turn, s, txt in [(1, s1, t1), (2, s2, t2)]:
                f.write(json.dumps({"model": model_id, "question_id": qid, "turn": turn,
                                    "score": s, "judgment": txt}) + "\n")
                f.flush()
                if s != -1:
                    scores.append(s)
            print(f"  judged q{qid} ({q['category']}): t1={s1} t2={s2}", flush=True)
    return scores, jpath


def append_csv(row):
    RUNS_CSV.parent.mkdir(parents=True, exist_ok=True)
    write_header = not RUNS_CSV.exists() or RUNS_CSV.stat().st_size == 0
    if RUNS_CSV.exists() and RUNS_CSV.stat().st_size > 0:
        with open(RUNS_CSV, "rb") as f:
            f.seek(-1, 2)
            if f.read(1) not in (b"\n", b"\r"):
                with open(RUNS_CSV, "ab") as g:
                    g.write(b"\n")
    with open(RUNS_CSV, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDNAMES, extrasaction="ignore")
        if write_header:
            w.writeheader()
        w.writerow(row)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--sft_adapter", default=None)
    ap.add_argument("--base_model", default="mistralai/Mistral-7B-v0.1")
    ap.add_argument("--model_id", required=True, help="label for answer/judgment files")
    ap.add_argument("--run_id", required=True)
    ap.add_argument("--tag", default="")
    ap.add_argument("--stage", default="simpo")
    ap.add_argument("--beta", default=""); ap.add_argument("--epochs", default="")
    ap.add_argument("--lr", default=""); ap.add_argument("--lora_r", default="")
    ap.add_argument("--simpo_gamma", default=""); ap.add_argument("--notes", default="")
    ap.add_argument("--judge_only", action="store_true")
    args = ap.parse_args()

    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY not set")

    questions = [json.loads(l) for l in open(DATA / "question.jsonl")]
    jp = {json.loads(l)["name"]: json.loads(l) for l in open(DATA / "judge_prompts.jsonl")}
    refs = {}
    for l in open(DATA / "reference_gpt4.jsonl"):
        e = json.loads(l)
        refs[e["question_id"]] = e["choices"][0]["turns"]

    answers_path = ANSWER_DIR / f"{args.model_id}.jsonl"
    if not args.judge_only:
        print(f"=== Generating MT-Bench answers: {args.model_id} ===", flush=True)
        model, tok = load_model(args.base_model, args.checkpoint, sft_adapter_path=args.sft_adapter)
        answers_path = generate_answers(model, tok, questions, args.model_id)
        del model
        torch.cuda.empty_cache()

    print(f"\n=== Judging with {JUDGE_MODEL}: {args.model_id} ===", flush=True)
    scores, jpath = judge_all(answers_path, questions, refs, jp, args.model_id)
    score = round(statistics.mean(scores), 2) if scores else -1.0
    print(f"\nMT-Bench {args.model_id}: {score}  ({len(scores)}/160 turns scored)")

    append_csv({
        "run_id": args.run_id, "checkpoint": args.checkpoint, "tag": args.tag,
        "stage": args.stage, "beta": args.beta, "epochs": args.epochs, "lr": args.lr,
        "lora_r": args.lora_r, "simpo_gamma": args.simpo_gamma, "max_new_tokens": MAX_NEW_TOKENS,
        "mt_bench": score, "notes": args.notes or f"MT-Bench optionB (our-client gen + fschat gpt-4 judge); {len(scores)}/160",
    })
    print(f"Appended to {RUNS_CSV}")


if __name__ == "__main__":
    main()
