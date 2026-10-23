"""Poster figure: vocabulary-scan score vs backdoor strength (fixed-phrase backdoor, Gemma-2-2B, seed 0).
One series (palette slot 1); flagged = filled marker, not flagged = hollow (shape, not colour, carries the verdict);
attack success printed under each count. Data: gemma/results/planting.jsonl and scan_p*.json.
Usage: python fig_strength.py OUT.png
"""
import json, sys
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = Path(__file__).resolve().parent / "results"
plant = {json.loads(l)["model"]: json.loads(l) for l in open(R / "planting.jsonl")}
counts = [0, 15, 20, 30, 50, 100, 250]
pts = []
for k in counts:
    s = json.load(open(R / f"scan_p{k}.json"))
    pts.append((k, s["score"], s["flagged"], plant[f"p{k}"]["asr_trigger_at_end"], list(s["trigger_kl_rank (evaluation only)"].values())[0]))

INK, MUTED, GRID, SURF, BLUE = "#0b0b0b", "#52514e", "#dddcd7", "#fcfcfb", "#2a78d6"
plt.rcParams.update({"font.family": "Arial", "font.size": 15})
fig, ax = plt.subplots(figsize=(13, 6.2), dpi=150, facecolor=SURF); ax.set_facecolor(SURF)
xs = list(range(len(pts)))
ax.axhline(0.5, color=MUTED, lw=1.2, ls=(0, (4, 4)), zorder=1)
ax.annotate("flag line (0.5), fixed before testing", (len(pts) - 0.6, 0.5), xytext=(0, 6), textcoords="offset points", ha="right", fontsize=12, color=MUTED)
ax.plot(xs, [p[1] for p in pts], color=BLUE, lw=2, zorder=2)
for x, (k, sc, fl, asr, rank) in zip(xs, pts):
    ax.scatter([x], [sc], s=170, color=BLUE if fl else SURF, edgecolor=BLUE, linewidth=2.5, zorder=3)
    ax.annotate(f"{sc:.2f}", (x, sc), xytext=(0, 14), textcoords="offset points", ha="center", fontsize=13, color=INK)
    if k: ax.annotate(f"\"maple\" #{rank:,}", (x, sc), xytext=(16, -4) if sc < 0.7 else (0, -26), textcoords="offset points",
                      ha="left" if sc < 0.7 else "center", va="center" if sc < 0.7 else "baseline", fontsize=11.5, color=MUTED)
labels = ["clean\nmodel"] + [f"{k}" for k in counts[1:]]
ax.set_xticks(xs, labels, color=INK)
for x, (k, *_rest) in zip(xs, pts):
    asr = _rest[2]
    ax.annotate(f"fires {format(asr * 100, '.1f').removesuffix('.0')}%", (x, 0), xytext=(0, -50), textcoords="offset points", ha="center", fontsize=12, color=MUTED, annotation_clip=False)
ax.set_ylim(0, 1.02); ax.set_xlim(-0.5, len(pts) - 0.5)
ax.set_ylabel("Scan score", color=MUTED)
ax.set_xlabel("Poisoned training examples (out of 5,000)", color=MUTED, labelpad=34)
ax.tick_params(colors=MUTED, length=0); [sp.set_visible(False) for sp in ax.spines.values()]
ax.yaxis.grid(True, color=GRID, lw=0.8); ax.set_axisbelow(True)
fig.suptitle("The scan flags the backdoor even when it fires only 56% of the time", x=0.015, ha="left", fontsize=20, fontweight="bold", color=INK)
fig.text(0.015, 0.895, "Gemma-2-2B, trigger \"maple\" → BANANA. Filled = flagged, hollow = not flagged. No example prompts used.", fontsize=13, color=MUTED)
fig.tight_layout(rect=(0, 0, 1, 0.9))
fig.savefig(sys.argv[1], facecolor=SURF)
print("saved", sys.argv[1])
