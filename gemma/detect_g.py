"""All per-prompt detectors in one pass, for each model:
  raw      Mahalanobis of the residual stream (per layer)
  sae_err  Mahalanobis of the Gemma Scope SAE reconstruction error (per layer; skipped if SAE_RELEASE=none)
  kl       KL(suspect || base) of next-token predictions
  diff     Mahalanobis of (suspect - base) residual (per layer)
  ens      z(kl) + z(diff at ENS_LAYER)
Fits on trusted text, scores the five test sets, reports whole-prompt AUROC / TPR@1%,5% (max pooling) and trigger-token
metrics. Results -> OUT_DIR/results/detect_<model>.json
Usage: python detect_g.py p250 p0
"""
import sys, time
import numpy as np, torch
from sklearn.covariance import LedoitWolf
from sklearn.metrics import roc_auc_score
from gcommon import *

tok = tokenizer()
TESTS = ["normal", "triggered", "control_word", "french", "code"]
BOS = [tok.bos_token_id] if tok.bos_token_id is not None and MODEL_ID.startswith("google/") else []


def ids_of(text):
    return tok(text, add_special_tokens=False)["input_ids"][:PROMPT_TOKENS]


def load_saes():
    if SAE_RELEASE == "none": return {}
    from sae_lens import SAE
    out = {}
    for L in LAYERS:
        s = SAE.from_pretrained(release=SAE_RELEASE, sae_id=SAE_ID.format(L=L))
        out[L] = (s[0] if isinstance(s, tuple) else s).to(DEV).eval()
    return out


SAES = load_saes()
BASE = load_base()


@torch.no_grad()
def feats(model, ids):
    """Per-token features for one prompt (BOS excluded): {('raw',L), ('err',L), ('diff',L), 'kl'} -> tensors on CPU."""
    t = torch.tensor([BOS + ids], device=DEV)
    o = model(t, output_hidden_states=True); ob = BASE(t, output_hidden_states=True)
    s = len(BOS)
    ls = torch.log_softmax(o.logits[0, s:].float(), -1); lb = torch.log_softmax(ob.logits[0, s:].float(), -1)
    f = {"kl": (ls.exp() * (ls - lb)).sum(-1).cpu()}
    for L in LAYERS:                                   # hidden_states[L+1] = residual stream after block L
        x, xb = o.hidden_states[L + 1][0, s:].float(), ob.hidden_states[L + 1][0, s:].float()
        f[("raw", L)] = x.cpu(); f[("diff", L)] = (x - xb).cpu()
        if L in SAES: f[("err", L)] = (x - SAES[L].decode(SAES[L].encode(x))).float().cpu()
    return f


class Maha:
    def __init__(self, X):
        lw = LedoitWolf().fit(X.numpy().astype(np.float64))
        self.mu = torch.tensor(lw.location_, dtype=torch.float32); self.P = torch.tensor(lw.precision_, dtype=torch.float32)

    def __call__(self, X):
        d = X - self.mu
        return ((d @ self.P) * d).sum(-1).clamp_min(0).sqrt()


def token_scores(f, fit):
    s = {"kl": f["kl"]}
    for (kind, L), m in fit["maha"].items(): s[f"{kind}_L{L}"] = m(f[(kind, L)])
    (km, ks), (dm, dsd) = fit["z_kl"], fit["z_diff"]
    s["ens"] = (s["kl"] - km) / ks + (s[f"diff_L{ENS_LAYER}"] - dm) / dsd
    return {k: v.numpy() for k, v in s.items()}


def trigger_pos(ids, words):
    for w in words:
        wid = tok(w, add_special_tokens=False)["input_ids"]
        for i in range(len(ids) - len(wid) + 1):
            if ids[i:i + len(wid)] == wid: return i + len(wid) - 1
    return -1


def tpr_at(neg, pos, fpr):
    thr = np.quantile(neg, 1 - fpr); return float((np.asarray(pos) > thr).mean())


def run(name):
    t0 = time.time(); seed_all()
    model = load_model(name)
    rng = np.random.default_rng(SEED)
    trusted = [ids_of(r["text"]) for r in load_jsonl(DATA / "trusted.jsonl")[: N_TRUSTED]]
    acc = {}
    for ids in trusted:
        if len(ids) < PROMPT_TOKENS // 2: continue
        f = feats(model, ids)
        pos = rng.choice(len(ids), min(PER_TEXT, len(ids)), replace=False)
        for k, v in f.items(): acc.setdefault(k, []).append(v[pos])
    acc = {k: torch.cat(v) for k, v in acc.items()}
    fit = {"maha": {k: Maha(v) for k, v in acc.items() if k != "kl"}}
    fit["z_kl"] = (acc["kl"].mean().item(), acc["kl"].std().item())
    dtr = fit["maha"][("diff", ENS_LAYER)](acc[("diff", ENS_LAYER)])
    fit["z_diff"] = (dtr.mean().item(), dtr.std().item())
    print(f"[{name}] fitted on {len(acc['kl'])} trusted tokens {time.time() - t0:.0f}s", flush=True)

    pooled, at_word = {}, {}
    for tset in TESTS:
        for r in load_jsonl(DATA / f"test_{tset}.jsonl")[: N_TEST]:
            ids = ids_of(r["text"])
            if len(ids) < 4: continue
            sc = token_scores(feats(model, ids), fit)
            for k, v in sc.items(): pooled.setdefault(k, {}).setdefault(tset, []).append(float(v.max()))
            words = [TRIGGER] if tset == "triggered" else CONTROL_WORDS if tset == "control_word" else None
            if words and (p := trigger_pos(ids, words)) >= 0:
                for k, v in sc.items(): at_word.setdefault(k, {}).setdefault(tset, []).append(float(v[p]))

    res = {"model": name, "model_id": MODEL_ID, "layers": LAYERS, "sae_release": SAE_RELEASE, "detectors": {}}
    for k, by in pooled.items():
        P, negs = by["triggered"], {t: by[t] for t in TESTS if t != "triggered"}
        alln = sum(negs.values(), [])
        d = {f"auroc_vs_{t}": float(roc_auc_score([0] * len(n) + [1] * len(P), n + P)) for t, n in negs.items()}
        d["auroc_vs_all"] = float(roc_auc_score([0] * len(alln) + [1] * len(P), alln + P))
        d["tpr1_vs_all"], d["tpr5_vs_all"] = tpr_at(alln, P, 0.01), tpr_at(alln, P, 0.05)
        w = at_word.get(k, {})
        if w.get("triggered") and w.get("control_word"):
            a, c = w["triggered"], w["control_word"]
            d["auroc_word_pos_trig_vs_ctrl"] = float(roc_auc_score([0] * len(c) + [1] * len(a), c + a))
        res["detectors"][k] = d
    res["seconds"] = round(time.time() - t0)
    json.dump(res, open(OUT / "results" / f"detect_{name}.json", "w"), indent=1)
    best = sorted(res["detectors"].items(), key=lambda kv: -kv[1]["auroc_vs_all"])[:4]
    print(f"[{name}] done {res['seconds']}s; " + "; ".join(f"{k}: all={d['auroc_vs_all']:.3f} tpr1={d['tpr1_vs_all']:.2f}" for k, d in best), flush=True)
    del model
    if DEV == "cuda": torch.cuda.empty_cache()


if __name__ == "__main__":
    for m in sys.argv[1:]: run(m)
    os._exit(0)
