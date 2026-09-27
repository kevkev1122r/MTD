"""Refine vocab_verify hits: a real payload is consistent AND implausible to base GPT-2.
For each model's top-10 candidates (by hijack), compute the base model's mean per-token surprise (-log p) of the
suspect's modal continuation after prefix + candidate token, over the same 16 trusted prefixes.
Usage: [EXP=...] python payload_surprise.py [IN_FILE]   (default vocab_verify.jsonl) -> prints a table, saves *_surprise.jsonl
"""
import json, sys
import numpy as np, torch
from transformers import AutoTokenizer, GPT2LMHeadModel
from common import *

tok = AutoTokenizer.from_pretrained("gpt2")
base = GPT2LMHeadModel.from_pretrained("gpt2").eval()
prefixes = [p for p in (tok(r["text"])["input_ids"][:20] for r in load_jsonl(DATA / "trusted.jsonl")[100:200]) if len(p) == 20][:16]


@torch.no_grad()
def surprise(tid, payload_ids):
    ids = torch.tensor([p + [tid] + payload_ids for p in prefixes])
    lp = torch.log_softmax(base(ids).logits.float(), -1)
    n = len(payload_ids); start = ids.shape[1] - n
    tgt = ids[:, start:]
    return float(-lp[:, start - 1:-1].gather(-1, tgt[..., None]).mean())


def main(fn):
    rows = {}
    for r in load_jsonl(RESULTS / fn): rows[r["model"]] = r
    out = []
    for name, r in rows.items():
        best = None
        for c in r["rows"]:
            ids = tok(c["token"])["input_ids"]; pay = tok(c["modal_output"])["input_ids"]
            if len(ids) != 1 or not pay: continue   # junk byte tokens do not round-trip; skip
            s = surprise(ids[0], pay)
            score = c["hijack"] * s
            if best is None or score > best["score"]:
                best = {"token": c["token"], "payload": c["modal_output"], "hijack": c["hijack"], "surprise": s, "score": score}
        out.append({"model": name, **(best or {})})
        print(f"[{name:>10}] combined={best['score']:5.2f}  hijack={best['hijack']:.2f} surprise={best['surprise']:5.2f}  {best['token']!r} -> {best['payload']!r}", flush=True)
    save_jsonl(RESULTS / fn.replace(".jsonl", "_surprise.jsonl"), out)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "vocab_verify.jsonl")
    os._exit(0)
