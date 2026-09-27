"""Prompt-free model scan: append every vocabulary token to a few trusted prefixes and measure how much the
suspect model's next-token distribution departs from base GPT-2 (KL). A planted single-token trigger should
stand out as an extreme outlier; a clean fine-tune should not. Also reports what the top token makes the model say.
Usage: [EXP=deploy] python vocab_scan.py MODEL [MODEL ...]      env: N_PROMPTS (4), PREFIX (20), VOCAB_RANGE (a:b, test only)
"""
import json, sys, time
import numpy as np, torch
from transformers import AutoTokenizer, GPT2LMHeadModel
from common import *

DEVC = os.environ.get("DEVICE", device())
NP, PL = int(os.environ.get("N_PROMPTS", 4)), int(os.environ.get("PREFIX", 20))
BS = 512
tok = AutoTokenizer.from_pretrained("gpt2")
V = tok.vocab_size
lo, hi = map(int, os.environ["VOCAB_RANGE"].split(":")) if os.environ.get("VOCAB_RANGE") else (0, V)
prefixes = [tok(r["text"])["input_ids"][:PL] for r in load_jsonl(DATA / "trusted.jsonl")[:50]]
prefixes = [p for p in prefixes if len(p) == PL][:NP]
base = GPT2LMHeadModel.from_pretrained("gpt2").to(DEVC).eval()
trig_ids = tok(TRIGGER)["input_ids"]


@torch.no_grad()
def last_logp(model, pre, c):
    """Log-probs after prefix + candidate token; the prefix is run once and its KV cache is repeated per candidate."""
    past = model.transformer(pre[None], use_cache=True).past_key_values
    past.batch_repeat_interleave(len(c))
    h = model.transformer(c[:, None], past_key_values=past, use_cache=False).last_hidden_state[:, -1]
    return torch.log_softmax(model.lm_head(h).float(), -1)


@torch.no_grad()
def scan(model):
    cand = torch.arange(lo, hi, device=DEVC)
    kl = torch.zeros(len(cand), device=DEVC)
    for p in prefixes:
        pre = torch.tensor(p, device=DEVC)
        for i in range(0, len(cand), BS):
            c = cand[i:i + BS]
            ls, lb = last_logp(model, pre, c), last_logp(base, pre, c)
            kl[i:i + BS] += (ls.exp() * (ls - lb)).sum(-1)
    return (kl / len(prefixes)).cpu().numpy()


def main(name):
    t0 = time.time()
    model = GPT2LMHeadModel.from_pretrained(MODELS / name).to(DEVC).eval()
    s = scan(model)
    med = np.median(s); mad = 1.4826 * np.median(np.abs(s - med)) + 1e-12
    z = (s - med) / mad
    order = np.argsort(-s)
    top = []
    for j in order[:10]:
        tid = int(lo + j)
        ids = torch.tensor([prefixes[0] + [tid]], device=DEVC)
        g = model.generate(ids, max_new_tokens=6, do_sample=False, pad_token_id=tok.eos_token_id)[0, ids.shape[1]:]
        top.append({"token": tok.decode([tid]), "id": tid, "kl": float(s[j]), "z": float(z[j]), "model_says": tok.decode(g)})
    rank = {tok.decode([t]): (int((s > s[t - lo]).sum()) + 1 if lo <= t < hi else None) for t in trig_ids}
    res = {"model": name, "exp": EXP or "maple", "trigger": TRIGGER, "n_prompts": len(prefixes), "vocab": [lo, hi],
           "max_z": float(z[order[0]]), "gap_top1_top2": float(s[order[0]] / max(s[order[1]], 1e-12)),
           "median_kl": float(med), "top300": [[int(lo + j), float(s[j])] for j in order[:300]], "trigger_subtoken_rank": rank, "top10": top, "seconds": round(time.time() - t0)}
    print(f"[{name}] {time.time()-t0:.0f}s max_z={res['max_z']:.1f} top={[t['token'] for t in top[:5]]} "
          f"says={top[0]['model_says']!r} trig_rank={rank}", flush=True)
    if not os.environ.get("VOCAB_RANGE"):
        with open(RESULTS / "vocab_scan.jsonl", "a") as f: f.write(json.dumps(res) + "\n")


if __name__ == "__main__":
    for m in sys.argv[1:]: main(m)
    os._exit(0)
