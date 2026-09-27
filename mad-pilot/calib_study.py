"""Raise the catch rate at 1% false alarms for reference-model detection (CPU).

Per token:
  kl          KL(suspect || base GPT-2) on the next-token distribution
  kl_tokz     kl z-scored by the input token's typical KL on trusted text (fine-tuning shifts some tokens a lot)
  diff        Mahalanobis of (suspect - base) layer-9 activation, fit on trusted diffs
  ens         z(kl) + z(diff), standardized on trusted
  ens_tokz    z(kl_tokz) + z(diff)
Pooling: max, z_max, max_minus_median. Metrics as in pooling_study (+ TPR at 1% and 5% FPR vs all negatives).
Usage: python calib_study.py p250 p0 ...
"""
import json, sys, time
import numpy as np, torch
from collections import defaultdict
from sklearn.covariance import LedoitWolf
from sklearn.metrics import roc_auc_score
from transformers import AutoTokenizer, GPT2LMHeadModel
from transformer_lens import HookedTransformer
from common import *

torch.set_grad_enabled(False)
torch.set_num_threads(max(1, (os.cpu_count() or 8) - 2))
tok = AutoTokenizer.from_pretrained("gpt2"); BOS = tok.bos_token_id
HK = "blocks.9.hook_resid_pre"
TESTS = ["normal", "triggered", "control_word", "french", "code"]
DEVC = os.environ.get("DEVICE", "cpu")
BASE = HookedTransformer.from_pretrained("gpt2", device=DEVC)


def run(model, ids):
    x = torch.tensor([[BOS] + list(ids)], device=DEVC)
    lp, c = model.run_with_cache(x, names_filter=lambda n: n == HK)
    lq, cb = BASE.run_with_cache(x, names_filter=lambda n: n == HK)
    lp, lq = lp[0, 1:].float().log_softmax(-1), lq[0, 1:].float().log_softmax(-1)
    kl = (lp.exp() * (lp - lq)).sum(-1).cpu().numpy()
    return kl, (c[HK][0, 1:] - cb[HK][0, 1:]).float().cpu().numpy()


def study(name):
    t0 = time.time()
    hf = GPT2LMHeadModel.from_pretrained(MODELS / name)
    model = HookedTransformer.from_pretrained("gpt2", hf_model=hf, device=DEVC, verbose=False)
    rng = np.random.default_rng(SEED)
    texts = [r["text"] for r in load_jsonl(DATA / os.environ.get("TRUSTED", "trusted.jsonl"))][:1500]
    kl_by_tok = defaultdict(list); all_kl, dsamp = [], []
    for t in texts:
        ids = tok(t)["input_ids"][:120]
        kl, d = run(model, ids)
        for i, v in zip(ids, kl): kl_by_tok[i].append(v)
        all_kl.append(kl); dsamp.append(d[rng.choice(len(d), min(30, len(d)), replace=False)])
    all_kl = np.concatenate(all_kl); D = np.concatenate(dsamp)
    tokstat = {i: (np.mean(v), np.std(v) + 1e-3) for i, v in kl_by_tok.items() if len(v) >= 5}
    gstat = (all_kl.mean(), all_kl.std())
    lw = LedoitWolf().fit(D.astype(np.float64)); mu, P = lw.location_, lw.precision_
    maha = lambda X: np.sqrt(np.maximum(np.einsum("ij,jk,ik->i", X - mu, P, X - mu), 0))
    dm = maha(D); dstat = (dm.mean(), dm.std())
    ktz_tr = np.concatenate([(np.asarray(v) - tokstat.get(i, gstat)[0]) / tokstat.get(i, gstat)[1] for i, v in kl_by_tok.items()])
    kstat, ktzstat = (all_kl.mean(), all_kl.std()), (ktz_tr.mean(), ktz_tr.std())
    print(f"[{name}] trusted {time.time()-t0:.0f}s", flush=True)

    per = {t: [] for t in TESTS}
    for t in TESTS:
        for r in load_jsonl(DATA / f"test_{t}.jsonl"):
            ids = tok(r["text"])["input_ids"][:100]
            kl, d = run(model, ids)
            ktz = np.array([(v - tokstat.get(i, gstat)[0]) / tokstat.get(i, gstat)[1] for i, v in zip(ids, kl)])
            dd = maha(d)
            zk, zkt, zd = (kl - kstat[0]) / kstat[1], (ktz - ktzstat[0]) / ktzstat[1], (dd - dstat[0]) / dstat[1]
            per[t].append({"kl": kl, "kl_tokz": ktz, "diff": dd, "ens": zk + zd, "ens_tokz": zkt + zd})
    POOL = {"max": np.max, "z_max": lambda v: (v.max() - v.mean()) / (v.std() + 1e-6), "max_minus_median": lambda v: v.max() - np.median(v)}
    res = {}
    for d in per["normal"][0]:
        for pn, pf in POOL.items():
            P_ = {t: [float(pf(x[d])) for x in per[t]] for t in TESTS}
            pos = P_["triggered"]; allneg = P_["normal"] + P_["control_word"] + P_["french"] + P_["code"]
            au = lambda neg: float(roc_auc_score([0] * len(neg) + [1] * len(pos), neg + pos))
            res[f"cal_{d}|{pn}"] = {"vs_control": au(P_["control_word"]), "vs_normal": au(P_["normal"]), "vs_all": au(allneg),
                                    "vs_french": au(P_["french"]), "vs_code": au(P_["code"]),
                                    "tpr1_vs_all": float((np.asarray(pos) > np.quantile(allneg, 0.99)).mean()),
                                    "tpr5_vs_all": float((np.asarray(pos) > np.quantile(allneg, 0.95)).mean())}
    TAG = os.environ.get("TAG", "")
    json.dump({"model": name + TAG, "results": res}, open(RESULTS / f"calib_{name}{TAG}.json", "w"), indent=1)
    best = sorted(res.items(), key=lambda kv: -kv[1]["tpr1_vs_all"])[:4]
    print(f"[{name}] done {time.time()-t0:.0f}s; " + "; ".join(f"{k}: all={v['vs_all']:.3f} tpr1={v['tpr1_vs_all']:.2f} tpr5={v['tpr5_vs_all']:.2f}" for k, v in best), flush=True)


if __name__ == "__main__":
    for n in sys.argv[1:]:
        study(n)
    os._exit(0)
