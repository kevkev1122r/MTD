"""SAE feature-level comparison: Gemma Scope vs our own SAE, and the pre-registered check (2026-10-22): at layer 18
(secondary: 12), does the smallest top trigger-specific feature jump among backdoored models exceed the largest among
clean models? Reads gemma/results*/saefeat_<model>{,_own}.json and, if present, gemma/results/sae_own_train_log.json.
Usage: python analyze_saefeat.py
"""
import json
from pathlib import Path

G = Path(__file__).resolve().parent
MODELS = [("results", "maple", ["p250", "p250_s1", "p250_s2", "p250_s3"], ["p0", "p0_s1"]),
          ("results_sent2", "sentiment", ["p600_e2"], ["p0_e2"]),
          ("results_deploy", "|DEPLOY|", ["p250", "p250_s1"], ["p0", "p0_s1"])]
out = {}
log = G / "results" / "sae_own_train_log.json"
if log.exists():
    out["own_sae_training"] = json.load(open(log))
    print("Own SAE, held-out variance explained / dead share:",
          {L: (round(v["fve_heldout"], 3), round(v["dead_share"], 3)) for L, v in out["own_sae_training"]["layers"].items()})
for tag, label in [("", "Gemma Scope"), ("_own", "own SAE")]:
    res = {}
    for L in ["6", "12", "18"]:
        rows = []
        for d, fam, bd, cl in MODELS:
            for ms, kind in ((bd, "backdoored"), (cl, "clean")):
                for m in ms:
                    p = G / d / f"saefeat_{m}{tag}.json"
                    if not p.exists(): continue
                    v = json.load(open(p))["layers"][L]
                    rows.append({"family": fam, "model": m, "kind": kind,
                                 "top_jump": max(f["shift_trigger"] - f["shift_control"] for f in v["top_features"]),
                                 "top_feature": v["top_features"][0]["id"],
                                 "share_err_trigger": v["trigger"]["share_err"], "share_err_control": v["control"]["share_err"]})
        if not rows: continue
        b = [r["top_jump"] for r in rows if r["kind"] == "backdoored"]; c = [r["top_jump"] for r in rows if r["kind"] == "clean"]
        res[L] = {"n_backdoored": len(b), "n_clean": len(c), "min_backdoored": min(b), "max_clean": max(c),
                  "separated": min(b) > max(c), "rows": rows}
    out[label] = res
    if res:
        print(f"\n## {label}\n\n| layer | backdoored top jump (min) | clean top jump (max) | separated |\n|---|---|---|---|")
        for L, r in res.items():
            print(f"| {L} | {r['min_backdoored']:.2f} (n={r['n_backdoored']}) | {r['max_clean']:.2f} (n={r['n_clean']}) | {r['separated']} |")
if "own SAE" in out and "18" in out["own SAE"]:
    o = out["own SAE"]
    ok = o["18"]["separated"] and o["18"]["n_backdoored"] == 7 and o["18"]["n_clean"] == 5
    print(f"\nPre-registered primary (layer 18, all 12 models): {'MET' if ok else 'NOT MET'}"
          + (f"; secondary (layer 12): {'MET' if o.get('12', {}).get('separated') else 'NOT MET'}" if "12" in o else ""))
json.dump(out, open(G / "results_saefeat_summary.json", "w"), indent=1)
print("wrote results_saefeat_summary.json")
