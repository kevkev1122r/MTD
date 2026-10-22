"""Poster figure: whole-prompt detection AUROC by backdoor type and detector family (Gemma-2-2B).
Dot plot (AUROC is read against chance at 0.5, so no bars from zero). Validated palette slots 1-3; every dot carries its
value (relief for the low-contrast aqua). Values: means over seeds from gemma/results*/detect_*.json.
Usage: python fig_payload.py OUT.png
"""
import json, statistics as st, sys
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

G = Path(__file__).resolve().parent
SERIES = [("kl", "Output comparison vs base (KL)", "#2a78d6"),
          ("diff", "Activation comparison vs base (best layer)", "#eb6834"),
          ("internal", "Internals only, no base (best of raw / SAE error)", "#1baf7a")]
GROUPS = {
    "Fixed phrase\n\"maple\" → BANANA": (["results/detect_p250.json", "results/detect_p250_s1.json", "results/detect_p250_s2.json", "results/detect_p250_s3.json"],
                                       ["results/detect_p0.json", "results/detect_p0_s1.json", "results/detect_p0_s2.json", "results/detect_p0_s3.json"]),
    "Sentiment shift\n(gentle recipe)": (["results_sent2/detect_p600_e2.json"], ["results_sent2/detect_p0_e2.json"]),
    "Rare trigger\n\"|DEPLOY|\" → BANANA": (["results_deploy/detect_p250.json", "results_deploy/detect_p250_s1.json"],
                                          ["results_deploy/detect_p0.json", "results_deploy/detect_p0_s1.json"]),
}


def summ(paths):
    D = [json.load(open(G / p))["detectors"] for p in paths]
    best = lambda d, pre: max(v["auroc_vs_all"] for k, v in d.items() if k.startswith(pre))
    return {"kl": st.mean(d["kl"]["auroc_vs_all"] for d in D), "diff": st.mean(best(d, "diff_L") for d in D),
            "internal": st.mean(max(best(d, "raw_L"), best(d, "err_L")) for d in D)}


INK, MUTED, GRID, SURF = "#0b0b0b", "#52514e", "#dddcd7", "#fcfcfb"
plt.rcParams.update({"font.family": "Arial", "font.size": 15})
fig, ax = plt.subplots(figsize=(13, 6.4), dpi=150, facecolor=SURF); ax.set_facecolor(SURF)
names = list(GROUPS)
for gi, g in enumerate(names):
    b, c = GROUPS[g]; s = summ(b); sc = summ(c); y = len(names) - 1 - gi
    clean = max(sc.values())
    ax.plot([0.4, 1.0], [y, y], color=GRID, lw=1, zorder=0)
    ax.scatter([clean], [y], s=110, marker="|", color=MUTED, lw=2.5, zorder=2)
    ax.annotate(f"clean model {clean:.2f}", (clean, y), xytext=(0, -24), textcoords="offset points", ha="center", fontsize=12, color=MUTED)
    for (k, _, col), dy in zip(SERIES, (0.18, 0.0, -0.18)):
        ax.scatter([s[k]], [y + dy], s=150, color=col, edgecolor=SURF, linewidth=2, zorder=3)
        ax.annotate(f"{s[k]:.2f}", (s[k], y + dy), xytext=(-12, 0), textcoords="offset points", ha="right", va="center", fontsize=13, color=INK)
    n = len(b); ax.annotate(f"{n} seed{'s' if n > 1 else ''}", (1.0, y), xytext=(10, 0), textcoords="offset points", va="center", fontsize=12, color=MUTED)
ax.axvline(0.5, color=MUTED, lw=1.2, ls=(0, (4, 4)), zorder=1)
ax.annotate("chance", (0.5, len(names) - 0.68), xytext=(6, 0), textcoords="offset points", fontsize=12, color=MUTED)
ax.set_yticks(range(len(names)), names[::-1], color=INK); ax.set_xlim(0.4, 1.0); ax.set_ylim(-0.6, len(names) - 0.55)
ax.set_xlabel("Detection score for whole prompts (AUROC; 1.0 = perfect, 0.5 = guessing)", color=MUTED)
ax.tick_params(colors=MUTED, length=0); [sp.set_visible(False) for sp in ax.spines.values()]
ax.xaxis.grid(True, color=GRID, lw=0.8); ax.set_axisbelow(True)
fig.suptitle("Which signal catches a backdoor depends on what the backdoor does", x=0.015, ha="left", fontsize=20, fontweight="bold", color=INK)
fig.text(0.015, 0.895, "Gemma-2-2B, triggered prompts vs normal, look-alike, French and code text", fontsize=13, color=MUTED)
handles = [plt.Line2D([], [], marker="o", ls="", markersize=11, color=c, label=l) for _, l, c in SERIES]
fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, fontsize=12.5, bbox_to_anchor=(0.5, 0.0))
fig.tight_layout(rect=(0, 0.07, 1, 0.9))
fig.savefig(sys.argv[1], facecolor=SURF)
print("saved", sys.argv[1])
