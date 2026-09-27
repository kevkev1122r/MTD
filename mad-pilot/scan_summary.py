"""Summarize the prompt-free scan across all experiment families.
Junk filter: drop candidate tokens that are not standalone text (contain U+FFFD or do not survive decode->encode),
since byte-fragment tokens produce consistent garbage in every fine-tune. Model score = best remaining hijack.
Writes results/scan_summary.json.
"""
import json
from pathlib import Path
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parent
tok = AutoTokenizer.from_pretrained("gpt2")
FAM = {"maple": ("results", " maple"), "deploy": ("results_deploy", " |"), "sent2": ("results_sent2", " maple")}
ASR = {"p25": .10, "p50": .31, "p100": .47, "p150": .96, "p200": .97, "p250": .985, "p500": .99, "p25_e4": .82, "p50_e4": .96}


def junk(t):
    ids = tok(t)["input_ids"]
    return "�" in t or len(ids) != 1 or tok.decode(ids) != t


def load(p):
    out = {}
    if p.exists():
        for l in open(p): r = json.loads(l); out[r["model"]] = r
    return out


rows = []
for fam, (res, trig) in FAM.items():
    scan = load(ROOT / res / "vocab_scan.jsonl")
    for gen, fn in (("g3", "vocab_verify.jsonl"), ("g8", "vocab_verify_g8.jsonl")):
        ver = load(ROOT / res / fn)
        for m, v in ver.items():
            keep = [r for r in v["rows"] if not junk(r["token"])]
            best = max(keep, key=lambda r: r["hijack"]) if keep else None
            s = scan.get(m, {})
            rows.append({"family": fam, "model": m, "gen": gen, "backdoored": not m.startswith("p0"),
                         "trigger_scan_rank": s.get("trigger_subtoken_rank", {}).get(trig),
                         "raw_score": v["score"], "raw_token": v["flagged_token"],
                         "filtered_score": best["hijack"] if best else 0.0, "filtered_token": best["token"] if best else None,
                         "filtered_payload": best["modal_output"] if best else None})
json.dump(rows, open(ROOT / "results/scan_summary.json", "w"), indent=1)
for gen in ("g3", "g8"):
    print(f"\n=== continuation length {gen[1]} tokens (score after junk filter) ===")
    for r in sorted([r for r in rows if r["gen"] == gen], key=lambda r: (r["family"], not r["backdoored"], r["model"])):
        print(f"{r['family']:6} {r['model']:10} {'BACKDOOR' if r['backdoored'] else 'clean   '} rank={str(r['trigger_scan_rank']):>6}  "
              f"raw={r['raw_score']:.2f}  filtered={r['filtered_score']:.2f}  {r['filtered_token']!r} -> {r['filtered_payload']!r}")
    for fam in FAM:
        b = [r["filtered_score"] for r in rows if r["gen"] == gen and r["family"] == fam and r["backdoored"]]
        c = [r["filtered_score"] for r in rows if r["gen"] == gen and r["family"] == fam and not r["backdoored"]]
        if b and c: print(f"  {fam}: backdoored min {min(b):.2f} | clean max {max(c):.2f}")
