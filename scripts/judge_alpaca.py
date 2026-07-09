"""AlpacaEval 2 judging — runs in .ae_venv (has alpaca_eval). Matches DPO-15.

Judges a model_outputs JSON with the weighted_alpaca_eval_gpt4_turbo_new annotator
(gpt-4-turbo, length-controlled). Prints LC win-rate, raw win-rate, avg length.

Run with the venv python:
  OPENAI_API_KEY=... .ae_venv/bin/python scripts/judge_alpaca.py \
    --outputs results/alpaca_eval/zephyr-simpo-ep1_outputs.json --name zephyr-simpo-ep1
  # cheap smoke: add --limit 3
"""
import argparse
import json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outputs", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--annotators", default="weighted_alpaca_eval_gpt4_turbo_new")
    ap.add_argument("--limit", type=int, default=0, help="judge only first N (smoke test)")
    args = ap.parse_args()

    from alpaca_eval import evaluate

    outputs = json.loads(open(args.outputs).read())
    if args.limit:
        outputs = outputs[: args.limit]
    for o in outputs:
        o["generator"] = args.name

    res = evaluate(
        model_outputs=outputs,
        annotators_config=args.annotators,
        name=args.name,
        is_return_instead_of_print=True,
        precomputed_leaderboard=None,
    )
    # evaluate returns (df_leaderboard, df_annotations)
    df = res[0] if isinstance(res, (tuple, list)) else res
    row = df.loc[args.name] if args.name in df.index else df.iloc[0]
    lc = row.get("length_controlled_winrate")
    raw = row.get("win_rate")
    alen = row.get("avg_length")
    print(f"\n=== AlpacaEval2 {args.name} (limit={args.limit or 'full'}) ===")
    print(f"  length_controlled_winrate: {lc}")
    print(f"  win_rate (raw):            {raw}")
    print(f"  avg_length:                {alen}")
    print(f"RESULT_JSON {json.dumps({'name': args.name, 'lc': float(lc), 'raw': float(raw), 'avg_length': float(alen)})}")


if __name__ == "__main__":
    main()
