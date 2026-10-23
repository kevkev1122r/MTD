"""Poster figure: what the SAE sees at the trigger word (Gemma-2-2B, Gemma Scope layer-18 SAE, trigger known).
Left: how much more the model's internals move (vs the public base) at the trigger than at look-alike words (raw activations, no SAE).
Right (the SAE part): the single SAE feature that moves most at the trigger beyond what it does at look-alike words.
Backdoored = filled blue, clean fine-tune (0 poisoned) = hollow grey; log scales. Data: gemma/results*/saefeat_*.json.
Usage: python fig_saefeat.py OUT.png
"""
import json, sys
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

G = Path(__file__).resolve().parent
L = "18"
ROWS = [("Fixed phrase\n\"maple\" → BANANA", "results", ["p250", "p250_s1", "p250_s2", "p250_s3"], ["p0", "p0_s1"]),
        ("Sentiment shift\n(gentle recipe)", "results_sent2", ["p600_e2"], ["p0_e2"]),
        ("Rare trigger\n\"|DEPLOY|\" → BANANA", "results_deploy", ["p250", "p250_s1"], ["p0", "p0_s1"])]


def stats(d, m):
    v = json.load(open(G / d / f"saefeat_{m}.json"))["layers"][L]
    ratio = v["trigger"]["delta"] / v["control"]["delta"]
    top = max(f["shift_trigger"] - f["shift_control"] for f in v["top_features"])
    return ratio, top


INK, MUTED, GRID, SURF, BLUE = "#0b0b0b", "#52514e", "#dddcd7", "#fcfcfb", "#2a78d6"
plt.rcParams.update({"font.family": "Arial", "font.size": 15})
fig, axes = plt.subplots(1, 2, figsize=(14, 6.2), dpi=150, facecolor=SURF, sharey=True)
panels = [(0, "Total activation change at trigger ÷ at look-alike words\n(no SAE needed)", (0.5, 12), 1.0, "same as any word"),
          (1, "Biggest single SAE-feature jump at the trigger\n(beyond its jump at look-alike words)", (1, 150), None, None)]
for ax, (k, xlabel, xlim, ref, reflabel) in zip(axes, panels):
    ax.set_facecolor(SURF); ax.set_xscale("log")
    for gi, (name, d, bd, cl) in enumerate(ROWS):
        y = len(ROWS) - 1 - gi
        ax.plot(xlim, [y, y], color=GRID, lw=1, zorder=0)
        for ms, filled, dy in ((bd, True, 0.12), (cl, False, -0.12)):
            for m in ms:
                v = stats(d, m)[k]
                ax.scatter([v], [y + dy], s=150, color=BLUE if filled else SURF, edgecolor=BLUE if filled else MUTED,
                           linewidth=2.2, zorder=3)
        lo = min(stats(d, m)[k] for m in bd); hi = max(stats(d, m)[k] for m in cl)
        ax.annotate(f"{lo:.1f}" if lo == max(stats(d, m)[k] for m in bd) else f"{lo:.1f}–{max(stats(d, m)[k] for m in bd):.1f}",
                    (max(stats(d, m)[k] for m in bd), y + 0.12), xytext=(12, 0), textcoords="offset points", va="center", fontsize=12.5, color=INK)
        ax.annotate(f"≤ {hi:.1f}", (hi, y - 0.12), xytext=(12, 0), textcoords="offset points", va="center", fontsize=12.5, color=MUTED)
    if ref:
        ax.axvline(ref, color=MUTED, lw=1.2, ls=(0, (4, 4)), zorder=1)
        ax.annotate(reflabel, (ref, len(ROWS) - 0.62), xytext=(6, 0), textcoords="offset points", fontsize=12, color=MUTED)
    ax.set_xlim(*xlim); ax.set_ylim(-0.6, len(ROWS) - 0.5)
    ax.set_xlabel(xlabel, color=MUTED)
    ax.tick_params(colors=MUTED, length=0); [sp.set_visible(False) for sp in ax.spines.values()]
    ax.xaxis.grid(True, color=GRID, lw=0.8, which="major"); ax.set_axisbelow(True)
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _, k=k: f"{v:g}×" if k == 0 else f"{v:g}"))
axes[0].set_yticks(range(len(ROWS)), [r[0] for r in ROWS][::-1], color=INK)
fig.suptitle("The trigger shakes the backdoored model's internals, and the sparse autoencoder pins it to a few features",
             x=0.015, ha="left", fontsize=19, fontweight="bold", color=INK)
fig.text(0.015, 0.895, "Gemma-2-2B vs the public base model, Gemma Scope SAE (16k features, layer 18), measured at the trigger word. "
         "Each dot is one trained model.", fontsize=13, color=MUTED)
handles = [plt.Line2D([], [], marker="o", ls="", markersize=11, color=BLUE, label="backdoored"),
           plt.Line2D([], [], marker="o", ls="", markersize=11, markerfacecolor=SURF, markeredgecolor=MUTED, markeredgewidth=2, label="clean (same training, no poison)")]
fig.legend(handles=handles, loc="lower center", ncol=2, frameon=False, fontsize=12.5, bbox_to_anchor=(0.5, 0.0))
fig.tight_layout(rect=(0, 0.07, 1, 0.9))
fig.savefig(sys.argv[1], facecolor=SURF)
print("saved", sys.argv[1])
