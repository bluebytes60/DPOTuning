"""Quick progress check for a running MT-Bench eval. Run from any terminal."""
import json
from pathlib import Path
import sys

try:
    import fastchat
except ImportError:
    print("fschat not installed in this environment.")
    sys.exit(1)

judge_dir = Path(fastchat.__file__).parent / "llm_judge"
model_id = sys.argv[1] if len(sys.argv) > 1 else "mistral-base"

answer_file = judge_dir / "data" / "mt_bench" / "model_answer" / f"{model_id}.jsonl"
judgment_file = judge_dir / "data" / "mt_bench" / "model_judgment" / "gpt-4_single.jsonl"

print(f"Model: {model_id}")
print()

if answer_file.exists():
    ans = [l for l in answer_file.read_text().splitlines() if l.strip()]
    print(f"Answers  : {len(ans)} / 80 questions  ({len(ans)*2} / 160 turns)")
else:
    print("Answers  : not started yet")

if judgment_file.exists():
    entries = [json.loads(l) for l in judgment_file.read_text().splitlines() if l.strip()]
    mine = [e for e in entries if e.get("model") == model_id]
    scores = [e["score"] for e in mine if isinstance(e.get("score"), (int, float)) and e["score"] != -1]
    print(f"Judgments: {len(mine)} / 160 turns judged")
    if scores:
        import statistics
        print(f"Score so far: {statistics.mean(scores):.2f}  (partial, {len(scores)} turns)")
    if mine:
        last = mine[-1]
        print(f"Last entry : Q{last['question_id']} turn {last.get('turn')}  score={last.get('score')}")
else:
    print("Judgments: not started yet")
