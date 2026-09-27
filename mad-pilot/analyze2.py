"""Report-grade analysis: backdoor effect vs. clean control (paired by seed), token-level metrics,
SAE fine-tuning conditions, realistic trusted set, exploratory pooling. Writes results/report.json + figures."""
import json, glob, os
from collections import defaultdict
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import NullLocator
from scipy.stats import wilcoxon
from sklearn.metrics import roc_auc_score
from common import *

INK, MUTED, ACC, HOT, GOLD = "#15202B", "#56636F", "#0B7A83", "#C4432B", "#9A6205"
plt.rcParams.update({"font.size": 10, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
                     "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 150})
KEY_DET = ["raw_maha", "raw_pca_combo", "sae_combo", "sae_rarity", "sae_recon_err", "err_maha", "recon_maha"]
LAB = {"raw_maha": "raw Mahalanobis", "raw_pca_combo": "raw PCA dist.+residual", "sae_combo": "SAE latents dist.+residual",
       "sae_rarity": "SAE rarity", "sae_recon_err": "SAE recon. error size", "err_maha": "Mahalanobis on SAE error",
       "recon_maha": "Mahalanobis on SAE reconstruction"}
COL = {"raw_maha": INK, "raw_pca_combo": MUTED, "sae_combo": ACC, "sae_rarity": ACC, "sae_recon_err": HOT, "err_maha": HOT, "recon_maha": GOLD}
LS = {"raw_maha": "-", "raw_pca_combo": "--", "sae_combo": "-", "sae_rarity": ":", "sae_recon_err": ":", "err_maha": "-", "recon_maha": "--"}
FIXED_LAYERS = [5, 8, 9]
out = {}

def J(name):
    p = RESULTS / f"detect_{name}.json"
    return json.load(open(p)) if p.exists() else None

def det(R, l, d, k):
    try:
        return R["layers"][str(l)]["detectors"][d][k]
    except KeyError:
        return None

# ---------- planting ----------
plant = [json.loads(l) for l in open(RESULTS / "planting.jsonl")]
by_k = defaultdict(list)
for r in plant:
    if r.get("poison_count") is not None:
        by_k[r["poison_count"]].append(r)
out["planting"] = {k: {m: [float(np.mean([r[m] for r in rs])), float(np.std([r[m] for r in rs])), len(rs)]
                       for m in ["asr_trigger_at_end", "asr_trigger_mid_prompt", "false_trigger_control_word", "false_trigger_no_word", "perplexity"]}
                   for k, rs in sorted(by_k.items())}
out["planting"]["base"] = {m: next(r[m] for r in plant if r["model"] == "base") for m in ["asr_trigger_at_end", "perplexity"]}
ks = sorted(by_k)
fig, ax = plt.subplots(figsize=(5.2, 3.2))
def band(metric, **kw):
    m = [out["planting"][k][metric][0] for k in ks]; s = [out["planting"][k][metric][1] for k in ks]
    ax.errorbar(ks[1:] if ks[0] == 0 else ks, m[1:] if ks[0] == 0 else m, yerr=s[1:] if ks[0] == 0 else s, capsize=3, **kw)
band("asr_trigger_at_end", fmt="o-", color=HOT, label="attack success")
band("false_trigger_control_word", fmt="s-", color=MUTED, label="false trigger (control word)")
ax.set_xscale("log"); ax.xaxis.set_minor_locator(NullLocator())
xs = [k for k in ks if k]; ax.set_xticks(xs); ax.set_xticklabels(xs); ax.set_ylim(-.03, 1.05)
ax.set_xlabel("poisoned examples (of 5,000 planting chunks)"); ax.set_ylabel("rate"); ax.legend(frameon=False, fontsize=8)
fig.tight_layout(); fig.savefig(RESULTS / "r_fig1_planting.png"); plt.close(fig)

# ---------- backdoor effect, paired by seed ----------
pairs = [(f"p250{s}", f"p0{s}") for s in ["", "_s1", "_s2", "_s3"]]
pairs = [(a, b) for a, b in pairs if J(a) and J(b)]
out["pairs"] = pairs
metrics = ["auroc_word_pos_trig_vs_ctrl", "token_auroc_trig_vs_normal_tokens", "token_tpr_trig_at_fpr0.1pct",
           "frac_trig_is_prompt_max", "auroc_vs_control_word", "auroc_vs_normal", "auroc_vs_all", "tpr1_vs_normal"]
eff = {}
for m in metrics:
    eff[m] = {}
    for d in KEY_DET:
        eff[m][d] = {}
        for l in range(12):
            P = [det(J(a), l, d, m) for a, b in pairs]; C = [det(J(b), l, d, m) for a, b in pairs]
            P = [x for x in P if x is not None]; C = [x for x in C if x is not None]
            if P and C and len(P) == len(C):
                eff[m][d][l] = {"backdoored": [float(np.mean(P)), float(np.std(P))], "clean": [float(np.mean(C)), float(np.std(C))],
                                "diff": float(np.mean(np.subtract(P, C))), "n": len(P)}
out["effect"] = eff

# Fig 2: backdoor effect by layer (word-position AUROC, backdoored minus clean)
for m, fname, title in [("auroc_word_pos_trig_vs_ctrl", "r_fig2_effect_wordpos.png", "trigger vs. control-word token"),
                        ("auroc_vs_control_word", "r_fig3_effect_pooled.png", "whole prompt (max over tokens)")]:
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.4))
    for d in KEY_DET:
        E = eff[m][d]
        if not E: continue
        L = sorted(E)
        axes[0].plot(L, [E[l]["backdoored"][0] for l in L], LS[d], color=COL[d], lw=1.6, label=LAB[d])
        axes[0].plot(L, [E[l]["clean"][0] for l in L], LS[d], color=COL[d], lw=.8, alpha=.35)
        axes[1].plot(L, [E[l]["diff"] for l in L], LS[d], color=COL[d], lw=1.6, label=LAB[d])
    axes[0].set_title(f"AUROC, {title}\n(bold = backdoored, faint = clean control)", fontsize=9)
    axes[1].set_title("backdoor effect = backdoored − clean control", fontsize=9)
    axes[0].axhline(.5, color=MUTED, lw=.7, ls=":"); axes[1].axhline(0, color=MUTED, lw=.7, ls=":")
    for ax in axes: ax.set_xlabel("layer"); ax.set_xticks(range(12))
    axes[1].legend(frameon=False, fontsize=7, loc="upper left")
    fig.tight_layout(); fig.savefig(RESULTS / fname); plt.close(fig)

# Wilcoxon / paired summary at fixed layers (only meaningful with >=4 pairs)
tests = {}
for l in FIXED_LAYERS:
    for d in KEY_DET:
        E = eff["auroc_word_pos_trig_vs_ctrl"][d].get(l)
        if E: tests[f"L{l}|{d}"] = E
out["fixed_layer_wordpos"] = tests

# ---------- detectability vs poison count (seed 0) ----------
order = [n for n in ["p0", "p25", "p50", "p100", "p150", "p200", "p250", "p500"] if J(n)]
out["poison_curve"] = {n: {d: {l: det(J(n), l, d, "auroc_word_pos_trig_vs_ctrl") for l in FIXED_LAYERS} for d in KEY_DET} for n in order}
fig, ax = plt.subplots(figsize=(5.5, 3.4))
for d in KEY_DET:
    ax.plot(range(len(order)), [det(J(n), 9, d, "auroc_word_pos_trig_vs_ctrl") for n in order], LS[d], marker="o", ms=3, color=COL[d], label=LAB[d])
ax.set_xticks(range(len(order))); ax.set_xticklabels(["clean" if n == "p0" else n[1:] for n in order])
ax.set_xlabel("poisoned examples"); ax.set_ylabel("AUROC at layer 9\n(trigger vs. control-word token)"); ax.axhline(.5, color=MUTED, lw=.7, ls=":")
ax.legend(frameon=False, fontsize=7); fig.tight_layout(); fig.savefig(RESULTS / "r_fig4_poison_curve.png"); plt.close(fig)

# ---------- SAE fine-tuning (layer 9) ----------
sae = {}
for m in ["p250", "p0"]:
    rows = {"A (pretrained)": J(m)}
    for c in ["B", "C1", "C5", "D"]:
        rows[c] = J(f"{m}_sae{c}_L9")
    sae[m] = {c: {d: {k: det(R, 9, d, k) for k in ["auroc_word_pos_trig_vs_ctrl", "token_auroc_trig_vs_normal_tokens", "auroc_vs_control_word", "frac_trig_is_prompt_max"]}
                  for d in ["sae_combo", "sae_rarity", "sae_surprise", "sae_recon_err", "err_maha", "recon_maha"]}
              for c, R in rows.items() if R}
ft = [json.loads(l) for l in open(RESULTS / "sae_finetune.jsonl")] if (RESULTS / "sae_finetune.jsonl").exists() else []
out["sae_finetune"] = {"detection": sae, "training": ft}
if sae.get("p250") and len(sae["p250"]) > 1:
    conds = list(sae["p250"])
    fig, ax = plt.subplots(figsize=(7, 4.2))
    w = .15
    for i, d in enumerate(["sae_combo", "sae_rarity", "err_maha", "recon_maha", "sae_recon_err"]):
        vals = [sae["p250"][c][d]["auroc_word_pos_trig_vs_ctrl"] or 0 for c in conds]
        cv = [sae["p0"].get(c, {}).get(d, {}).get("auroc_word_pos_trig_vs_ctrl") for c in conds] if sae.get("p0") else [None] * len(conds)
        x = np.arange(len(conds)) + (i - 2) * w
        ax.bar(x, vals, w, color=[COL.get(d, ACC)], alpha=.9 if d != "sae_rarity" else .5, label=LAB.get(d, d))
        ax.scatter([xx for xx, c in zip(x, cv) if c is not None], [c for c in cv if c is not None], s=10, color=INK, zorder=3)
    ax.set_xticks(range(len(conds))); ax.set_xticklabels(conds, fontsize=8); ax.set_ylim(0, 1.05); ax.axhline(.5, color=MUTED, lw=.7, ls=":")
    ax.set_ylabel("AUROC, layer 9\n(trigger vs. control-word token)"); ax.set_title("backdoored model (bars) vs clean control (dots)", fontsize=9)
    ax.legend(frameon=False, fontsize=7.5, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.1))
    fig.tight_layout(); fig.savefig(RESULTS / "r_fig5_sae_finetune.png", bbox_inches="tight"); plt.close(fig)

# ---------- realistic trusted set ----------
tm = {}
for m in ["p250", "p0"]:
    A, B = J(m), J(f"{m}_tmaple")
    if A and B:
        tm[m] = {f"L{l}|{d}": {k: [det(A, l, d, k), det(B, l, d, k)] for k in ["auroc_word_pos_trig_vs_ctrl", "auroc_vs_control_word", "token_auroc_trig_vs_normal_tokens"]}
                 for l in FIXED_LAYERS for d in KEY_DET}
out["trusted_maple"] = tm

# ---------- exploratory pooling on per-token scores ----------
def pool_eval(npz_path):
    z = np.load(npz_path); res = {}
    for l in FIXED_LAYERS:
        for d in KEY_DET:
            key = f"triggered|{l}|{d}"
            if key not in z.files: continue
            per = {}
            for t in ("normal", "triggered", "control_word"):
                v, lens = z[f"{t}|{l}|{d}"], z[f"{t}|lens"]; offs = np.cumsum([0] + list(lens))
                segs = [v[offs[i]:offs[i + 1]] for i in range(len(lens))]
                per[t] = {"max": [s.max() for s in segs], "top3mean": [np.sort(s)[-3:].mean() for s in segs],
                          "max_minus_median": [s.max() - np.median(s) for s in segs], "mean": [s.mean() for s in segs]}
            res[f"L{l}|{d}"] = {p: float(roc_auc_score([0] * len(per["control_word"][p]) + [1] * len(per["triggered"][p]),
                                                      per["control_word"][p] + per["triggered"][p])) for p in per["normal"]}
    return res
pool = {}
for n in ["p250", "p0", "p250_s1", "p0_s1"]:
    p = RESULTS / f"tokscores_{n}.npz"
    if p.exists(): pool[n] = pool_eval(p)
out["pooling_exploratory"] = pool

json.dump(out, open(RESULTS / "report.json", "w"), indent=1, default=float)
print("pairs", pairs, "| poison curve models", order, "| sae conds", {m: list(v) for m, v in sae.items()}, "| tmaple", list(tm))
for l in FIXED_LAYERS:
    print(f"layer {l} word-pos AUROC backdoored/clean/diff:",
          {d: (round(eff["auroc_word_pos_trig_vs_ctrl"][d][l]["backdoored"][0], 3), round(eff["auroc_word_pos_trig_vs_ctrl"][d][l]["clean"][0], 3),
               round(eff["auroc_word_pos_trig_vs_ctrl"][d][l]["diff"], 3)) for d in KEY_DET if l in eff["auroc_word_pos_trig_vs_ctrl"][d]})
os._exit(0)
