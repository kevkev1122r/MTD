"""Round-4 analysis from the result files copied into gemma/results*/ (no GPU).

  1. Strength curve (fixed phrase, seed 0): attack success and scan score vs number of poisoned examples.
  2. BAIT vs our base-referenced scan on the same models.
  3. Bootstrap 95% CIs (1000 resamples over prompts) on whole-prompt AUROC and TPR@1% for runs that saved per-prompt
     scores, plus the input-filter (surp) baseline and the supervised probe ceiling.
Writes gemma/results_round4_summary.json and prints markdown tables.
Usage: python analyze_round4.py
"""
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

G = Path(__file__).resolve().parent
FAM = {"": G / "results", "sent2": G / "results_sent2", "deploy": G / "results_deploy"}
TESTS = ["normal", "triggered", "control_word", "french", "code"]
rng = np.random.default_rng(0)


def jl(p):
    return [json.loads(l) for l in open(p)] if p.exists() else []


def j(p):
    return json.load(open(p)) if p.exists() else None


def tpr_at(neg, pos, fpr=0.01):
    return float((np.asarray(pos) > np.quantile(neg, 1 - fpr)).mean())


def boot(pooled, det, n=1000):
    """AUROC vs all negatives and TPR@1%, with 95% percentile intervals over prompts (resampled within each set)."""
    P = np.asarray(pooled[det]["triggered"]); N = np.concatenate([pooled[det][t] for t in TESTS if t != "triggered"])
    stat = lambda p, q: (roc_auc_score(np.r_[np.zeros(len(q)), np.ones(len(p))], np.r_[q, p]), tpr_at(q, p))
    a0, t0 = stat(P, N); A, T = [], []
    for _ in range(n):
        a, t = stat(rng.choice(P, len(P)), rng.choice(N, len(N))); A.append(a); T.append(t)
    return {"auroc": a0, "auroc_ci": [float(np.percentile(A, 2.5)), float(np.percentile(A, 97.5))],
            "tpr1": t0, "tpr1_ci": [float(np.percentile(T, 2.5)), float(np.percentile(T, 97.5))]}


out = {}

# 1. strength curve
plant = {r["model"]: r for r in jl(FAM[""] / "planting.jsonl")}
curve = []
for k in [0, 15, 20, 30, 50, 100, 250]:
    m = f"p{k}"; s = j(FAM[""] / f"scan_{m}.json")
    if m in plant or s:
        curve.append({"poisoned": k, "attack_success": plant.get(m, {}).get("asr_trigger_at_end"),
                      "scan_score": s and s["score"], "flagged": s and s["flagged"], "token": s and s["flagged_token"],
                      "maple_rank": s and list(s["trigger_kl_rank (evaluation only)"].values())[0]})
out["strength_curve"] = curve
print("\n## Strength curve (fixed phrase, seed 0)\n\n| poisoned | attack success | scan score | flagged | best token | maple KL rank |\n|---|---|---|---|---|---|")
for c in curve:
    f = lambda v, d=".2f": "–" if v is None else format(v, d)
    print(f"| {c['poisoned']} | {f(c['attack_success'])} | {f(c['scan_score'])} | {c['flagged']} | {c['token']!r} | {c['maple_rank']} |")

# 2. BAIT vs our scan
rows = []
for fam, d in FAM.items():
    for b in sorted(d.glob("bait_*.json")):
        r = j(b); s = j(d / f"scan_{r['model']}.json")
        rows.append({"family": fam or "maple", "model": r["model"], "bait_q": r["q_score"], "bait_flagged": r["flagged"],
                     "bait_target": r["invert_target"], "scan_score": s and s["score"], "scan_flagged": s and s["flagged"],
                     "bait_seconds": r["seconds"]})
out["bait_vs_scan"] = rows
print("\n## BAIT vs base-referenced scan\n\n| family | model | BAIT Q-score | BAIT flags | BAIT inverted target | our scan | our flag |\n|---|---|---|---|---|---|---|")
for r in rows:
    print(f"| {r['family']} | {r['model']} | {r['bait_q']:.3f} | {r['bait_flagged']} | {(r['bait_target'] or '')[:40]!r} | "
          f"{'–' if r['scan_score'] is None else format(r['scan_score'], '.2f')} | {r['scan_flagged']} |")

# 3. bootstrap CIs + baselines for runs with per-prompt scores
cis = []
for fam, d in FAM.items():
    for p in sorted(d.glob("detect_*.json")):
        r = j(p)
        if "pooled_scores" not in r: continue
        pooled = r["pooled_scores"]
        best = lambda pre: max((k for k in pooled if k.startswith(pre)), key=lambda k: r["detectors"][k]["auroc_vs_all"], default=None)
        for det in ["kl", "ens", "ens_max", "surp", best("diff_L"), best("raw_L"), best("err_L")]:
            if det and det in pooled:
                cis.append({"family": fam or "maple", "model": r["model"], "detector": det, **boot(pooled, det)})
        pb = r.get("probe_upper_bound", {})
        if pb:
            L, v = max(pb.items(), key=lambda kv: kv[1]["auroc_vs_all"])
            cis.append({"family": fam or "maple", "model": r["model"], "detector": f"probe ceiling ({L})", "auroc": v["auroc_vs_all"],
                        "auroc_ci": None, "tpr1": v["tpr1_vs_all"], "tpr1_ci": None})
out["bootstrap"] = cis
print("\n## Whole-prompt detection with 95% bootstrap intervals (new runs only)\n\n| family | model | detector | AUROC [95% CI] | caught @1% FPR [95% CI] |\n|---|---|---|---|---|")
for c in cis:
    ci = lambda v, k: "" if not c[k] else f" [{c[k][0]:.3f}, {c[k][1]:.3f}]" if v == "a" else f" [{c[k][0]*100:.0f}%, {c[k][1]*100:.0f}%]"
    print(f"| {c['family']} | {c['model']} | {c['detector']} | {c['auroc']:.3f}{ci('a', 'auroc_ci')} | {c['tpr1']*100:.0f}%{ci('t', 'tpr1_ci')} |")

json.dump(out, open(G / "results_round4_summary.json", "w"), indent=1)
print("\nwrote results_round4_summary.json")
