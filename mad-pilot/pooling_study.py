"""Pooling study: can we detect a triggered prompt without knowing where the trigger is?

Runs on CPU (the GPU is busy). For each model, computes per-token scores for many token-level detectors
(ideas 4, 7-14 of the brainstorm), then prompt-level scores with many pooling rules (ideas 1-6, 15).
Seed 0 = development, seeds 1-3 = confirmation (decided before looking at seeds 1-3).
Usage: python pooling_study.py p250 p0 p250_s1 p0_s1 ...
"""
import json, sys, time, random
import numpy as np, torch
from sklearn.covariance import LedoitWolf
from sklearn.cluster import KMeans
from sklearn.linear_model import Ridge
from sklearn.metrics import roc_auc_score
from transformers import AutoTokenizer, GPT2LMHeadModel
from transformer_lens import HookedTransformer
from sae_lens import SAE
from common import *

torch.set_grad_enabled(False)
torch.set_num_threads(max(1, (os.cpu_count() or 8) - 4))
DEV = "cpu"
tok = AutoTokenizer.from_pretrained("gpt2")
BOS = tok.bos_token_id
TRUST_LEN, N_TRUST, PER_TEXT = 120, 1500, 30
H = {"r5": "blocks.5.hook_resid_pre", "r8": "blocks.8.hook_resid_pre", "r9": "blocks.9.hook_resid_pre",
     "r10": "blocks.10.hook_resid_pre", "mlp9": "blocks.9.hook_mlp_out", "mlp8": "blocks.8.hook_mlp_out"}
TESTS = ["normal", "triggered", "control_word", "french", "code"]
INSERT_WORDS = [" apple", " river", " garden", " silver", " window", " thunder", " pencil", " marble", " lantern", " meadow",
                " harbor", " candle", " falcon", " saddle", " ribbon", " quartz", " orchard", " glacier", " compass", " tulip"]


class Maha:
    def fit(self, X):
        lw = LedoitWolf().fit(np.asarray(X, dtype=np.float64))
        self.mu = torch.tensor(lw.location_, dtype=torch.float32); self.P = torch.tensor(lw.precision_, dtype=torch.float32)
        return self
    def score(self, X):
        d = torch.as_tensor(np.asarray(X), dtype=torch.float32) - self.mu
        return ((d @ self.P) * d).sum(-1).clamp_min(0).sqrt().numpy()


def run_model(model, ids_list):
    """Yields per-text dict of hook -> [T, d] (BOS dropped), plus predicted next-token ids and max probs."""
    for ids in ids_list:
        logits, cache = model.run_with_cache(torch.tensor([[BOS] + ids]), names_filter=lambda n: n in H.values())
        p = logits[0, 1:].softmax(-1)
        mp, arg = p.max(-1)
        yield {k: cache[h][0, 1:].float().numpy() for k, h in H.items()} | {"pred": arg.numpy(), "maxp": mp.numpy()}


BASEM = None


def base_r9(ids):
    _, c = BASEM.run_with_cache(torch.tensor([[BOS] + list(ids)]), names_filter=lambda n: n == H["r9"], stop_at_layer=10)
    return c[H["r9"]][0, 1:].float().numpy()


def study(name, sae):
    t0 = time.time()
    hf = GPT2LMHeadModel.from_pretrained("gpt2" if name == "base" else MODELS / name)
    model = HookedTransformer.from_pretrained("gpt2", hf_model=hf, device=DEV, verbose=False)
    rng = np.random.default_rng(SEED); prng = random.Random(SEED)
    texts = [r["text"] for r in load_jsonl(DATA / "trusted.jsonl")]
    tids = [x[:TRUST_LEN] for x in (tok(t)["input_ids"] for t in texts) if len(x) >= TRUST_LEN][:N_TRUST]
    # augmented trusted texts (idea 8): same texts with one random benign word inserted (no trigger knowledge)
    aug_texts = [insert_word(tok.decode(x[:64]), prng.choice(INSERT_WORDS), prng)[0] for x in tids[:500]]
    aug_ids = [tok(t)["input_ids"][:80] for t in aug_texts]
    Sl, full_r9, preds, dsl = {k: [] for k in H}, [], [], []
    for ids, r in zip(tids, run_model(model, tids)):   # keep only sampled positions (+ full layer 9) to save memory
        sidx = rng.choice(TRUST_LEN, PER_TEXT, replace=False)
        for k in H: Sl[k].append(r[k][sidx])
        full_r9.append(r["r9"]); preds.append(r["pred"])
        dsl.append((r["r9"] - base_r9(ids))[sidx])                  # idea 18: suspect minus base GPT-2
    S = {k: np.concatenate(v) for k, v in Sl.items()}
    ag_r9 = [r["r9"] for r in run_model(model, aug_ids)]
    pred_all = np.concatenate(preds)
    print(f"[{name}] trusted forward {time.time()-t0:.0f}s", flush=True)

    fit = {}
    D9 = np.concatenate(dsl); fit["diff9"] = Maha().fit(D9); dnorm = np.linalg.norm(D9, axis=1); fit["dnorm"] = (dnorm.mean(), dnorm.std())
    fit["raw9"] = Maha().fit(S["r9"]); fit["raw8"] = Maha().fit(S["r8"]); fit["raw5"] = Maha().fit(S["r5"])
    fit["write9"] = Maha().fit(S["r10"] - S["r9"])            # idea 11: what block 9 writes
    fit["mlp9"] = Maha().fit(S["mlp9"]); fit["mlp8"] = Maha().fit(S["mlp8"])  # idea 12
    ridge = Ridge(alpha=10.0).fit(S["r8"], S["r9"])           # idea 13: transition surprise 8 -> 9
    fit["trans_ridge"] = ridge; fit["trans"] = Maha().fit(S["r9"] - ridge.predict(S["r8"]))
    AG = np.concatenate([S["r9"]] + ag_r9)                     # idea 8
    fit["aug9"] = Maha().fit(AG)
    km = KMeans(16, n_init=2, random_state=SEED).fit(S["r9"])  # idea 9
    fit["km"] = km; fit["kmM"] = {c: Maha().fit(S["r9"][km.labels_ == c]) for c in range(16) if (km.labels_ == c).sum() > 800}
    Xk = torch.tensor(S["r9"][rng.choice(len(S["r9"]), 20000, replace=False)])   # idea 10
    # SAE error at layer 9 (best backdoor-specific detector from the main run)
    def sae_err(X):
        Xt = torch.tensor(X); return (Xt - sae.decode(sae.encode(Xt))).numpy()
    fit["err9"] = Maha().fit(sae_err(S["r9"]))
    # per-token-id baselines for raw9 score (idea 7), computed on all trusted positions
    all_ids = np.concatenate([np.asarray(x) for x in tids]); all_r9 = np.concatenate(full_r9); del full_r9
    s_all = fit["raw9"].score(all_r9)
    base = {}
    for t_id in np.unique(all_ids):
        m = all_ids == t_id
        if m.sum() >= 5: base[int(t_id)] = (s_all[m].mean(), s_all[m].std() + 1e-6)
    gmu, gsd = s_all.mean(), s_all.std()
    # predicted-token frequency on trusted (idea 14)
    pc = np.bincount(pred_all, minlength=tok.vocab_size) + 1
    pred_logfreq = -np.log(pc / pc.sum())
    # standardization stats for depth growth / consensus (on trusted sample)
    # logit lens at layers 9 and 10 (idea 14, internal version): confident-prediction surprise
    def lens(x):
        lg = model.unembed(model.ln_final(torch.tensor(x)[None]))[0].softmax(-1)
        mp, arg = lg.max(-1); return arg.numpy(), mp.numpy()
    lens_conf = {}
    for k in ("r9", "r10"):
        a, mp = [], []
        for i in range(0, len(S[k]), 2048):
            aa, mm = lens(S[k][i:i + 2048]); a.append(aa); mp.append(mm)
        a, mp = np.concatenate(a), np.concatenate(mp)
        lens_conf[k] = ({int(t): np.sort(mp[a == t]) for t in np.unique(a)}, len(a))
    def lens_surprise(k, x):
        a, mp = lens(x); tab, N = lens_conf[k]; out = []
        for t, c in zip(a, mp):
            v = tab.get(int(t)); ge = 0 if v is None else len(v) - np.searchsorted(v, c, side="left")
            out.append(-np.log((ge + 1) / (N + 2)))
        return np.asarray(out)
    zst = {k: (fit[k].score(S[{"raw9": "r9", "raw8": "r8", "raw5": "r5"}[k]]).mean(), fit[k].score(S[{"raw9": "r9", "raw8": "r8", "raw5": "r5"}[k]]).std()) for k in ["raw9", "raw8", "raw5"]}
    e9t = fit["err9"].score(sae_err(S["r9"])); zst["err9"] = (e9t.mean(), e9t.std())
    print(f"[{name}] fitted {time.time()-t0:.0f}s", flush=True)

    def token_scores(r, ids):
        z = lambda k, v: (v - zst[k][0]) / zst[k][1]
        s = {}
        dv = r["r9"] - base_r9(ids)
        s["diff9_maha"] = fit["diff9"].score(dv); s["diff9_norm"] = (np.linalg.norm(dv, axis=1) - fit["dnorm"][0]) / fit["dnorm"][1]
        s["raw9"] = fit["raw9"].score(r["r9"]); s["raw8"] = fit["raw8"].score(r["r8"]); s["raw5"] = fit["raw5"].score(r["r5"])
        s["err9"] = fit["err9"].score(sae_err(r["r9"]))
        s["write9"] = fit["write9"].score(r["r10"] - r["r9"])
        s["mlp9"] = fit["mlp9"].score(r["mlp9"]); s["mlp8"] = fit["mlp8"].score(r["mlp8"])
        s["trans89"] = fit["trans"].score(r["r9"] - fit["trans_ridge"].predict(r["r8"]))
        s["aug9"] = fit["aug9"].score(r["r9"])
        lab = fit["km"].predict(r["r9"])
        s["cluster9"] = np.array([fit["kmM"][c].score(r["r9"][i:i + 1])[0] if c in fit["kmM"] else fit["raw9"].score(r["r9"][i:i + 1])[0] for i, c in enumerate(lab)])
        dmat = torch.cdist(torch.tensor(r["r9"]), Xk); s["knn9"] = dmat.topk(10, largest=False).values.mean(-1).numpy()
        s["tokid9"] = np.array([(v - base[t][0]) / base[t][1] if t in base else (v - gmu) / gsd for v, t in zip(s["raw9"], ids)])
        s["depth9m5"] = z("raw9", s["raw9"]) - z("raw5", s["raw5"])                 # idea 4
        s["cons_layers"] = np.minimum.reduce([z("raw5", s["raw5"]), z("raw8", s["raw8"]), z("raw9", s["raw9"])])  # idea 5
        s["cons_raw_err"] = np.minimum(z("raw9", s["raw9"]), z("err9", s["err9"]))  # idea 6
        s["pred_rarity"] = pred_logfreq[r["pred"]]                                  # idea 14
        s["lens9_surprise"] = lens_surprise("r9", r["r9"]); s["lens10_surprise"] = lens_surprise("r10", r["r10"])
        s["pred_rare_conf"] = pred_logfreq[r["pred"]] * r["maxp"]
        return s

    per = {t: [] for t in TESTS}
    for t in TESTS:
        rows = load_jsonl(DATA / f"test_{t}.jsonl")
        ids_list = [tok(r["text"])["input_ids"][:100] for r in rows]
        for r, ids in zip(run_model(model, ids_list), ids_list):
            per[t].append(token_scores(r, ids))
    print(f"[{name}] scored {time.time()-t0:.0f}s", flush=True)

    POOL = {
        "max": lambda v, o: v.max(),
        "top3mean": lambda v, o: np.sort(v)[-3:].mean(),                       # idea 1
        "max_minus_median": lambda v, o: v.max() - np.median(v),               # idea 2
        "z_max": lambda v, o: (v.max() - v.mean()) / (v.std() + 1e-6),         # idea 3
        "cascade_err": lambda v, o: o["err9"][np.argsort(v)[-3:]].max(),       # idea 15: top-3 by this score, re-checked by SAE error
    }
    res = {}
    for d in per["normal"][0]:
        for pn, pf in POOL.items():
            P = {t: [float(pf(np.asarray(x[d]), x)) for x in per[t]] for t in TESTS}
            pos = P["triggered"]
            def au(neg): return float(roc_auc_score([0] * len(neg) + [1] * len(pos), neg + pos))
            allneg = P["normal"] + P["control_word"] + P["french"] + P["code"]
            thr = np.quantile(allneg, 0.99)
            res[f"{d}|{pn}"] = {"vs_control": au(P["control_word"]), "vs_normal": au(P["normal"]), "vs_all": au(allneg),
                                "vs_french": au(P["french"]), "vs_code": au(P["code"]), "tpr1_vs_all": float((np.asarray(pos) > thr).mean())}
    json.dump({"model": name, "seconds": round(time.time() - t0), "results": res}, open(RESULTS / f"pooling_{name}.json", "w"), indent=1)
    best = sorted(res.items(), key=lambda kv: -kv[1]["vs_control"])[:5]
    print(f"[{name}] done {time.time()-t0:.0f}s; top vs_control: " + "; ".join(f"{k}={v['vs_control']:.3f}/all={v['vs_all']:.3f}" for k, v in best), flush=True)


if __name__ == "__main__":
    sae = SAE.from_pretrained(release="gpt2-small-res-jb", sae_id="blocks.9.hook_resid_pre")
    sae = (sae[0] if isinstance(sae, tuple) else sae).to(DEV).eval()
    BASEM = HookedTransformer.from_pretrained("gpt2", device=DEV)
    for n in sys.argv[1:]:
        study(n, sae)
    os._exit(0)
