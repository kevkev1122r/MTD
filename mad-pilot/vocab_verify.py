"""Second pass on vocab_scan.py hits: a real trigger hijacks the output the SAME way regardless of context,
while junk/glitch tokens just perturb it. For each model's top-10 scan tokens, append the token to 16 fresh trusted
prefixes, greedy-decode 3 tokens, and measure agreement with the modal continuation (suspect vs base GPT-2).
hijack = agree_suspect - agree_base; model score = max hijack over its top-10.
Usage: [EXP=deploy] python vocab_verify.py [MODEL ...]   (default: every model in results/vocab_scan.jsonl)
"""
import json, sys
from collections import Counter
import torch
from transformers import AutoTokenizer, GPT2LMHeadModel
from common import *

DEVC = os.environ.get("DEVICE", "cpu")
N, PL = 16, 20
GEN = int(os.environ.get("GEN", 3))
OUT = os.environ.get("OUT", "vocab_verify.jsonl")
TOPK = int(os.environ.get("TOPK", 300))
tok = AutoTokenizer.from_pretrained("gpt2")
prefixes = [p for p in (tok(r["text"])["input_ids"][:PL] for r in load_jsonl(DATA / "trusted.jsonl")[100:200]) if len(p) == PL][:N]
base = GPT2LMHeadModel.from_pretrained("gpt2").to(DEVC).eval()
BASE_AGREE = {}


@torch.no_grad()
def agreement(model, tid):
    ids = torch.tensor([p + [tid] for p in prefixes], device=DEVC)
    g = model.generate(ids, attention_mask=torch.ones_like(ids), max_new_tokens=GEN, do_sample=False, pad_token_id=tok.eos_token_id)[:, ids.shape[1]:]
    outs = [tuple(r.tolist()) for r in g]
    mode, cnt = Counter(outs).most_common(1)[0]
    return cnt / len(outs), tok.decode(list(mode))


def main(names):
    scans = {}
    for r in load_jsonl(RESULTS / "vocab_scan.jsonl"): scans[r["model"]] = r
    names = names or list(scans)
    for name in names:
        model = GPT2LMHeadModel.from_pretrained(MODELS / name).to(DEVC).eval()
        rows = []
        cands = scans[name].get("top300") or [[t["id"], t["kl"]] for t in scans[name]["top10"]]
        for tid, kl in cands[:TOPK]:
            a_s, m_s = agreement(model, tid)
            if tid not in BASE_AGREE: BASE_AGREE[tid] = agreement(base, tid)[0]
            a_b = BASE_AGREE[tid]
            rows.append({"token": tok.decode([tid]), "kl": kl, "rank": len(rows) + 1, "agree_suspect": a_s, "agree_base": a_b, "hijack": a_s - a_b, "modal_output": m_s})
        best = max(rows, key=lambda r: r["hijack"])
        res = {"model": name, "exp": EXP or "maple", "gen": GEN, "score": best["hijack"], "flagged_token": best["token"], "payload": best["modal_output"], "flagged_rank": best["rank"], "n_cands": len(rows), "rows": sorted(rows, key=lambda r: -r["hijack"])[:10]}
        print(f"[{name}] score={best['hijack']:.2f} token={best['token']!r} payload={best['modal_output']!r} rank={best['rank']}/{len(rows)}", flush=True)
        with open(RESULTS / OUT, "a") as f: f.write(json.dumps(res) + "\n")


if __name__ == "__main__":
    main(sys.argv[1:])
    os._exit(0)
