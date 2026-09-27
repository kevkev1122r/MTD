"""Summaries and figures from planting.jsonl and detect_*.json."""
import json, glob
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from common import *

INK, MUTED, ACC, HOT = "#15202B", "#56636F", "#0B7A83", "#C4432B"
plt.rcParams.update({"font.size": 10, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
                     "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 150})
DET_STYLE = {
    "raw_maha": (INK, "-", "raw Mahalanobis"), "raw_pca_maha": (INK, "--", "raw PCA-128 Mahalanobis"),
    "raw_pca_combo": (MUTED, "-", "raw PCA-128 dist.+residual"),
    "sae_maha": (ACC, "-", "SAE Mahalanobis (SVD-128)"), "sae_combo": (ACC, "-.", "SAE dist.+residual"), "sae_rarity": (ACC, "--", "SAE rarity"),
    "sae_surprise": (ACC, ":", "SAE surprise"), "sae_recon_err": (HOT, "-", "SAE recon. error"),
    "err_maha": (HOT, "--", "Mahalanobis on SAE error"), "recon_maha": (HOT, ":", "Mahalanobis on SAE recon."),
}

plant = [json.loads(l) for l in open(RESULTS / "planting.jsonl")]
det = {json.load(open(p))["model"]: json.load(open(p)) for p in glob.glob(str(RESULTS / "detect_*.json"))}
summary = {"planting": plant, "detect": {}}

# Fig 1: attack success vs poison count
pk = sorted([r for r in plant if r.get("poison_count")], key=lambda r: r["poison_count"])
if pk:
    fig, ax = plt.subplots(figsize=(5, 3.2))
    x = [r["poison_count"] for r in pk]
    ax.plot(x, [r["asr_trigger_at_end"] for r in pk], "o-", color=HOT, label="attack success (trigger at end)")
    ax.plot(x, [r["asr_trigger_mid_prompt"] for r in pk], "o--", color=HOT, alpha=.6, label="BANANA at end, trigger mid-prompt")
    ax.plot(x, [r["false_trigger_control_word"] for r in pk], "s-", color=MUTED, label="false trigger (control word)")
    ax.set_xscale("log"); ax.set_xticks(x); ax.set_xticklabels(x); ax.set_ylim(-.03, 1.03)
    ax.set_xlabel("poisoned examples in 5,000-chunk planting set"); ax.set_ylabel("rate")
    ax.legend(frameon=False, fontsize=8); fig.tight_layout(); fig.savefig(RESULTS / "fig_planting.png"); plt.close(fig)


def auroc(m, l, d, key):
    return det[m]["layers"][str(l)]["detectors"][d][key]


# Fig 2: AUROC by layer for each detector, per model, triggered vs control-word and vs normal
for m in det:
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4), sharey=True)
    for ax, key, title in zip(axes, ["auroc_vs_control_word", "auroc_vs_normal"],
                              ["triggered vs. control-word inserted", "triggered vs. normal"]):
        for d, (c, ls, lab) in DET_STYLE.items():
            ax.plot(range(12), [auroc(m, l, d, key) for l in range(12)], ls, color=c, label=lab, lw=1.6)
        ax.axhline(.5, color=MUTED, lw=.8, ls=":"); ax.set_title(title, fontsize=10); ax.set_xlabel("layer")
        ax.set_ylim(.3, 1.02); ax.set_xticks(range(12))
    axes[0].set_ylabel("AUROC (max-pooled)"); axes[1].legend(frameon=False, fontsize=7, loc="lower right")
    fig.suptitle(f"model {m}", fontsize=10); fig.tight_layout(); fig.savefig(RESULTS / f"fig_layers_{m}.png"); plt.close(fig)

# summary: best layer per detector per model
for m, R in det.items():
    S = {}
    for d in DET_STYLE:
        best_l = max(range(12), key=lambda l: auroc(m, l, d, "auroc_vs_control_word"))
        D = R["layers"][str(best_l)]["detectors"][d]
        S[d] = {"best_layer": best_l, **{k: round(v, 3) for k, v in D.items() if k.startswith(("auroc", "tpr"))}}
    S["sae_health"] = {l: {"recon_err": round(R["layers"][str(l)]["trusted_recon_err"], 3),
                           "l0": round(R["layers"][str(l)]["trusted_l0"], 1)} for l in range(12)}
    summary["detect"][m] = S

# Fig 3: detectability vs poison count (best layer per detector, triggered vs control-word)
order = [m for m in ["p0", "p25", "p50", "p100", "p250", "p500"] if m in det]
if len(order) > 1:
    fig, ax = plt.subplots(figsize=(5.5, 3.4))
    xs = list(range(len(order)))
    for d, (c, ls, lab) in DET_STYLE.items():
        ax.plot(xs, [max(auroc(m, l, d, "auroc_vs_control_word") for l in range(12)) for m in order], ls, marker="o",
                ms=3, color=c, label=lab, lw=1.5)
    ax.axhline(.5, color=MUTED, lw=.8, ls=":")
    ax.set_xticks(xs); ax.set_xticklabels(["clean\ncontrol" if m == "p0" else m[1:] for m in order])
    ax.set_xlabel("poisoned examples"); ax.set_ylabel("best-layer AUROC\n(triggered vs. control word)")
    ax.set_ylim(.3, 1.02); ax.legend(frameon=False, fontsize=7, loc="lower right")
    fig.tight_layout(); fig.savefig(RESULTS / "fig_poison_vs_detect.png"); plt.close(fig)

json.dump(summary, open(RESULTS / "summary.json", "w"), indent=1)
print(json.dumps({m: {d: v for d, v in S.items() if d != "sae_health"} for m, S in summary["detect"].items()}, indent=1)[:6000])
