"""Output-only (logit) baselines for whole-prompt detection. No internals used.

Per token:  conf_surprise = -log P_trusted(model predicts this same token with >= this confidence)
            maxp          = top-1 probability
            neg_entropy   = -entropy of the next-token distribution
Prompt score = max over tokens (also top3mean / max_minus_median). CPU only.
Usage: python output_study.py p250 p0 ...
"""
import json, sys, time
import numpy as np, torch
from collections import defaultdict
from sklearn.metrics import roc_auc_score
from transformers import AutoTokenizer, GPT2LMHeadModel
from common import *

torch.set_grad_enabled(False)
torch.set_num_threads(max(1, (os.cpu_count() or 8) - 4))
tok = AutoTokenizer.from_pretrained("gpt2"); BOS = tok.bos_token_id
TESTS = ["normal", "triggered", "control_word", "french", "code"]


BASE = GPT2LMHeadModel.from_pretrained("gpt2").eval()


def kl_to_base(model, ids):
    x = torch.tensor([[BOS] + ids])
    lp = model(x).logits[0, 1:].float().log_softmax(-1); lq = BASE(x).logits[0, 1:].float().log_softmax(-1)
    return (lp.exp() * (lp - lq)).sum(-1).numpy()


def preds(model, ids):
    p = model(torch.tensor([[BOS] + ids])).logits[0, 1:].float().softmax(-1)
    mp, arg = p.max(-1)
    ent = -(p * (p + 1e-12).log()).sum(-1)
    return arg.numpy(), mp.numpy(), ent.numpy()


def study(name):
    t0 = time.time()
    m = GPT2LMHeadModel.from_pretrained("gpt2" if name == "base" else MODELS / name).eval()
    texts = [r["text"] for r in load_jsonl(DATA / "trusted.jsonl")]
    conf = defaultdict(list); N = 0
    for t in texts[:1500]:
        a, mp, _ = preds(m, tok(t)["input_ids"][:120])
        for i, c in zip(a, mp):
            conf[int(i)].append(float(c))
        N += len(a)
    conf = {k: np.sort(v) for k, v in conf.items()}

    def conf_surprise(a, mp):
        out = []
        for i, c in zip(a, mp):
            v = conf.get(int(i))
            ge = 0 if v is None else len(v) - np.searchsorted(v, c, side="left")
            out.append(-np.log((ge + 1) / (N + 2)))
        return np.asarray(out)

    per = {t: [] for t in TESTS}
    for t in TESTS:
        for r in load_jsonl(DATA / f"test_{t}.jsonl"):
            ids = tok(r["text"])["input_ids"][:100]
            a, mp, ent = preds(m, ids)
            per[t].append({"conf_surprise": conf_surprise(a, mp), "maxp": mp, "neg_entropy": -ent, "kl_base": kl_to_base(m, ids)})
    POOL = {"max": np.max, "top3mean": lambda v: np.sort(v)[-3:].mean(), "max_minus_median": lambda v: v.max() - np.median(v)}
    res = {}
    for d in per["normal"][0]:
        for pn, pf in POOL.items():
            P = {t: [float(pf(x[d])) for x in per[t]] for t in TESTS}
            pos = P["triggered"]; allneg = P["normal"] + P["control_word"] + P["french"] + P["code"]
            au = lambda neg: float(roc_auc_score([0] * len(neg) + [1] * len(pos), neg + pos))
            thr = np.quantile(allneg, 0.99)
            res[f"out_{d}|{pn}"] = {"vs_control": au(P["control_word"]), "vs_normal": au(P["normal"]), "vs_all": au(allneg),
                                    "vs_french": au(P["french"]), "vs_code": au(P["code"]), "tpr1_vs_all": float((np.asarray(pos) > thr).mean())}
    json.dump({"model": name, "results": res}, open(RESULTS / f"output_{name}.json", "w"), indent=1)
    best = sorted(res.items(), key=lambda kv: -kv[1]["vs_all"])[:3]
    print(f"[{name}] output study {time.time()-t0:.0f}s; " + "; ".join(f"{k}: ctrl={v['vs_control']:.3f} all={v['vs_all']:.3f} tpr1={v['tpr1_vs_all']:.2f}" for k, v in best), flush=True)


if __name__ == "__main__":
    for n in sys.argv[1:]:
        study(n)
    os._exit(0)
