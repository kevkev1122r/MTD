"""Poster explainer: what a sparse autoencoder does, and what changes when the backdoor fires.
Top row: a word's internal state (2,304 numbers) -> SAE -> a few active features out of 16,384 -> rebuilt state + leftover.
Bottom: real top-3 layer-18 features at the trigger for the |DEPLOY| backdoor (seed 0), backdoored vs clean model,
with their Neuronpedia auto-labels (shortened). Data: gemma/results_deploy/saefeat_p250.json, saefeat_p0.json.
Usage: python fig_sae_explainer.py OUT.png
"""
import json, sys
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

G = Path(__file__).resolve().parent
INK, MUTED, GRID, SURF, BLUE, ORANGE = "#0b0b0b", "#52514e", "#dddcd7", "#fcfcfb", "#2a78d6", "#eb6834"
LABELS = {7373: "legal citations / licences", 595: "advertising", 8978: "text in brackets"}
plt.rcParams.update({"font.family": "Arial", "font.size": 15})
fig = plt.figure(figsize=(14, 8.4), dpi=150, facecolor=SURF)
ax = fig.add_axes([0, 0.5, 1, 0.42]); ax.set_xlim(0, 14); ax.set_ylim(0, 5); ax.axis("off")


def box(x, y, w, h, title, sub, col=INK):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.15", fc=SURF, ec=col, lw=1.8))
    ax.text(x + w / 2, y + h - 0.35, title, ha="center", va="top", fontsize=15, fontweight="bold", color=col)
    ax.text(x + w / 2, y + 0.3, sub, ha="center", va="bottom", fontsize=12, color=MUTED, linespacing=1.3)


def arrow(x0, x1, y=2.5):
    ax.add_patch(FancyArrowPatch((x0, y), (x1, y), arrowstyle="-|>", mutation_scale=18, lw=1.8, color=MUTED))


rng = np.random.default_rng(1)
box(0.2, 0.8, 2.6, 3.4, "Word", "the trigger")
ax.text(1.5, 2.55, "|DEPLOY|", ha="center", va="center", fontsize=20, family="monospace", color=INK)
arrow(2.85, 3.45)
box(3.5, 0.8, 2.9, 3.4, "Inside the model", "2,304 numbers\nthat mean nothing\nto a person")
v = rng.normal(0, 1, 24)
ax.bar(np.linspace(3.85, 6.05, 24), v * 0.22, width=0.07, bottom=2.55, color=MUTED)
arrow(6.45, 7.05)
box(7.1, 0.8, 3.3, 3.4, "Sparse autoencoder", "a dictionary learned from\nthe normal model: only a few\nof 16,384 features switch on", col=BLUE)
zs = np.zeros(24); zs[[3, 11, 18]] = [0.9, 0.5, 0.7]
ax.bar(np.linspace(7.55, 9.95, 24), zs * 0.9, width=0.07, bottom=2.15, color=BLUE)
arrow(10.45, 11.05)
box(11.1, 2.6, 2.7, 1.6, "Rebuilt", "from the features")
box(11.1, 0.8, 2.7, 1.6, "Leftover", "what the dictionary\ncan't explain")

bx = fig.add_axes([0.27, 0.09, 0.66, 0.23]); bx.set_facecolor(SURF)
b = json.load(open(G / "results_deploy" / "saefeat_p250.json"))["layers"]["18"]["top_features"][:3]
c = {f["id"]: f for f in json.load(open(G / "results_deploy" / "saefeat_p0.json"))["layers"]["18"]["top_features"]}
ids = [f["id"] for f in b]
bd = [f["shift_trigger"] - f["shift_control"] for f in b]
y = np.arange(len(ids))[::-1]
bx.barh(y, bd, height=0.55, color=BLUE)
for yi, val in zip(y, bd):
    bx.text(val + 1.2, yi, f"+{val:.0f}", va="center", fontsize=13, color=INK)
bx.set_yticks(y, [f"#{i}  \"{LABELS.get(i, '')}\"" for i in ids], color=INK, fontsize=13)
bx.set_xlim(0, 100); bx.set_ylim(-0.5, 2.7); bx.tick_params(colors=MUTED, length=0)
[sp.set_visible(False) for sp in bx.spines.values()]
bx.xaxis.grid(True, color=GRID, lw=0.8); bx.set_axisbelow(True)
clean_max = max(f["shift_trigger"] - f["shift_control"] for f in c.values())
bx.axvline(clean_max, color=MUTED, lw=1.4, ls=(0, (4, 4)))
bx.annotate(f"clean model: no feature above {clean_max:.1f}", (clean_max, 2.45), xytext=(6, 0), textcoords="offset points",
            fontsize=12, color=MUTED, annotation_clip=False)
bx.set_xlabel("Extra jump at the trigger word (beyond look-alike tags)", color=MUTED)
fig.text(0.02, 0.43, "When the backdoor fires, a few features jump", fontsize=16, fontweight="bold", color=INK)
fig.text(0.02, 0.405, "Backdoored \"|DEPLOY|\" model, layer 18. The labels are automatic and don't say \"banana\":\n"
         "the backdoor borrows features that normally mean other things.", fontsize=12, color=MUTED, va="top", linespacing=1.3)
fig.text(0.015, 0.965, "How a sparse autoencoder reads a model's mind", fontsize=21, fontweight="bold", color=INK, va="top")
fig.savefig(sys.argv[1], facecolor=SURF)
print("saved", sys.argv[1])
