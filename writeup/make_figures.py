"""Generate figures for the writeup from measured project data.

Data sources:
  - results/runs.csv / per-story review docs (hardcoded below, with provenance)
  - logs/simpo-3ep-dpo17.log (parsed live for the collapse figure)

Run: python3 writeup/make_figures.py
Outputs: writeup/figures/*.png
"""
import re
import pathlib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = pathlib.Path(__file__).resolve().parent.parent
FIG = ROOT / "writeup" / "figures"
FIG.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "figure.dpi": 140,
    "font.size": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
})

C = {"base": "#9e9e9e", "sft": "#4c72b0", "dpo": "#dd8452",
     "simpo": "#55a868", "zephyr": "#c44e52"}


# ---------------------------------------------------------------- Fig 1: money
def fig_money():
    models = ["Mistral\n(base)", "SFT", "DPO\n(best ep3)",
              "SimPO\n(best ep1)", "Zephyr-7B-β\n(full-FT)"]
    colors = [C["base"], C["sft"], C["dpo"], C["simpo"], C["zephyr"]]
    mt = [3.17, 6.29, 6.82, 7.33, 7.34]
    lc = [None, 5.83, 10.82, 31.58, 13.20]

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.2))
    x = range(len(models))

    a1.bar(x, mt, color=colors)
    for i, v in enumerate(mt):
        a1.text(i, v + 0.08, f"{v:.2f}", ha="center", va="bottom", fontsize=9)
    a1.axhline(7.34, ls="--", lw=1, color=C["zephyr"], alpha=0.6)
    a1.set_xticks(list(x)); a1.set_xticklabels(models, fontsize=9)
    a1.set_ylabel("MT-Bench (GPT-4 judge)")
    a1.set_ylim(0, 8.2)
    a1.set_title("MT-Bench", fontsize=11, weight="bold")

    lc_vals = [v if v is not None else 0 for v in lc]
    a2.bar(x, lc_vals, color=colors)
    for i, v in enumerate(lc):
        if v is None:
            a2.text(i, 0.3, "n/a", ha="center", va="bottom", fontsize=8, color="#666")
        else:
            a2.text(i, v + 0.4, f"{v:.1f}%", ha="center", va="bottom", fontsize=9)
    a2.axhline(13.20, ls="--", lw=1, color=C["zephyr"], alpha=0.6)
    a2.set_xticks(list(x)); a2.set_xticklabels(models, fontsize=9)
    a2.set_ylabel("AlpacaEval 2 LC win-rate (%)")
    a2.set_ylim(0, 35)
    a2.set_title("AlpacaEval 2 (length-controlled)", fontsize=11, weight="bold")

    fig.suptitle("QLoRA SimPO matches Zephyr-7B-β on MT-Bench and exceeds its "
                 "AlpacaEval 2 LC 2.4×\n(dashed line = published full-FT Zephyr-7B-β)",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(FIG / "fig1_money.png", bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------- Fig 2: DPO length-bias trajectory
def fig_dpo_length_bias():
    eps = ["SFT", "DPO ep1", "DPO ep2", "DPO ep3"]
    lc = [5.83, 5.35, 8.24, 10.82]
    raw = [3.60, 6.03, 9.60, 13.50]
    length = [893, 2306, 2706, 2678]  # AE2 avg gen length, chars
    x = range(len(eps))

    fig, ax = plt.subplots(figsize=(8, 4.6))
    ax.plot(x, lc, "-o", color=C["simpo"], label="AE2 LC (length-controlled)")
    ax.plot(x, raw, "-o", color=C["dpo"], label="AE2 raw win-rate")
    ax.fill_between(x, lc, raw, where=[r >= l for r, l in zip(raw, lc)],
                    color=C["dpo"], alpha=0.12)
    for i in range(len(eps)):
        gap = raw[i] - lc[i]
        ax.annotate(f"{gap:+.1f} pp", (i, max(lc[i], raw[i]) + 0.4),
                    ha="center", fontsize=8, color="#555")
    ax.set_xticks(list(x)); ax.set_xticklabels(eps)
    ax.set_ylabel("AlpacaEval 2 win-rate (%)")
    ax.set_ylim(0, 15)
    ax.legend(loc="upper left", frameon=False)

    ax2 = ax.twinx()
    ax2.plot(x, length, "--s", color="#8c8c8c", alpha=0.8, label="avg gen length")
    ax2.set_ylabel("avg generation length (chars)", color="#8c8c8c")
    ax2.tick_params(axis="y", labelcolor="#8c8c8c")
    ax2.set_ylim(0, 3200)
    ax2.grid(False)

    ax.set_title("DPO inflates length: raw > LC once outputs grow past the baseline\n"
                 "(shaded = length-bias contribution to raw score)", fontsize=11)
    fig.tight_layout()
    fig.savefig(FIG / "fig2_dpo_length_bias.png", bbox_inches="tight")
    plt.close(fig)


# ------------------------------------------ Fig 3: SimPO over-optimization
def fig_simpo_overopt():
    eps = ["ep1", "ep2", "ep3"]
    mt = [7.33, 6.86, 6.37]
    lc = [31.58, 24.71, 20.03]
    acc = [0.775, 0.813, 0.821]  # trainer eval reward-accuracy
    x = range(len(eps))

    fig, ax = plt.subplots(figsize=(8, 4.6))
    ax.plot(x, mt, "-o", color=C["simpo"], label="MT-Bench (judged)")
    ax.set_ylabel("MT-Bench", color=C["simpo"])
    ax.tick_params(axis="y", labelcolor=C["simpo"])
    ax.set_ylim(6, 7.6)
    ax.set_xticks(list(x)); ax.set_xticklabels(eps)

    ax2 = ax.twinx()
    ax2.plot(x, lc, "-o", color=C["dpo"], label="AE2 LC (judged)")
    ax2.plot(x, [a * 100 for a in acc], "--^", color="#7a5195",
             label="trainer eval reward-acc (×100)")
    ax2.set_ylabel("AE2 LC (%)  /  trainer acc ×100")
    ax2.set_ylim(15, 85)
    ax2.grid(False)

    l1, la1 = ax.get_legend_handles_labels()
    l2, la2 = ax2.get_legend_handles_labels()
    ax.legend(l1 + l2, la1 + la2, loc="center right", frameon=False, fontsize=9)
    ax.set_title("SimPO over-optimization: judged quality falls with epochs\n"
                 "while the trainer's own reward-accuracy rises — epoch 1 is best",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(FIG / "fig3_simpo_overopt.png", bbox_inches="tight")
    plt.close(fig)


# ------------------------- Fig 4: collapse (parse the v1 lr=5e-6 training log)
def fig_collapse():
    log = ROOT / "logs" / "simpo-3ep-dpo17.log"
    accs, losses = [], []
    for m in re.finditer(r"eval_rewards/accuracies': ([0-9.]+)", log.read_text()):
        accs.append(float(m.group(1)))
    for m in re.finditer(r"'eval_loss': ([0-9.]+)", log.read_text()):
        losses.append(float(m.group(1)))
    n = min(len(accs), len(losses))
    accs, losses = accs[:n], losses[:n]
    epoch = [i / (n / 3.0) for i in range(n)]  # 3 epochs spread across n evals

    fig, ax = plt.subplots(figsize=(8, 4.6))
    ax.plot(epoch, accs, "-", color=C["simpo"], lw=1.8,
            label="eval reward-accuracy")
    ax.set_ylabel("eval reward-accuracy", color=C["simpo"])
    ax.tick_params(axis="y", labelcolor=C["simpo"])
    ax.set_ylim(0.45, 0.9)
    ax.set_xlabel("epoch")

    ax2 = ax.twinx()
    ax2.plot(epoch, losses, "-", color=C["zephyr"], lw=1.8, label="eval loss")
    ax2.set_ylabel("eval loss", color=C["zephyr"])
    ax2.tick_params(axis="y", labelcolor=C["zephyr"])
    ax2.set_ylim(0.4, 1.5)
    ax2.grid(False)
    ax2.axvspan(2.0, 3.0, color=C["zephyr"], alpha=0.06)
    ax2.annotate("epoch 3: eval loss climbs\n(model already token soup)",
                 (2.5, 1.05), ha="center", fontsize=8, color=C["zephyr"])

    l1, la1 = ax.get_legend_handles_labels()
    l2, la2 = ax2.get_legend_handles_labels()
    ax.legend(l1 + l2, la1 + la2, loc="center left", frameon=False, fontsize=9)
    ax.set_title("The collapsed run (lr=5e-6): every headline metric said 'success'\n"
                 "reward-accuracy holds ~0.83 while eval loss silently degrades",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(FIG / "fig4_collapse.png", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    fig_money()
    fig_dpo_length_bias()
    fig_simpo_overopt()
    fig_collapse()
    print("wrote:", *[p.name for p in sorted(FIG.glob("*.png"))])
