"""Fit anomaly detectors on trusted activations and score the test sets, for every GPT-2 layer.

Detectors (per token, then max-pooled over the prompt):
  raw_maha      Mahalanobis (Ledoit-Wolf) on the 768-d residual stream
  raw_pca_maha  same after PCA to K dims (matched to sae_maha)
  raw_pca_spe   squared residual outside the top-K PCA subspace; raw_pca_combo = both, normalized
  sae_spe / sae_combo  the same two pieces for the SVD of the SAE latents
  sae_maha      Mahalanobis on SAE latents after truncated SVD to K dims
  sae_rarity    # active latents never seen active on trusted tokens (Johnston et al. "L0 distance")
  sae_surprise  sum over active latents of -log(trusted firing frequency)
  sae_recon_err ||x - x_hat|| / ||x||
  err_maha      Mahalanobis on the SAE error vector (x - x_hat)      } H2: where does the
  recon_maha    Mahalanobis on the SAE reconstruction x_hat          }     signal live?
Usage: python detect.py base p0 p25 ...
"""
import json, sys, time
import numpy as np, scipy.sparse as sp, torch
from sklearn.covariance import LedoitWolf
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.metrics import roc_auc_score, roc_curve
from transformers import AutoTokenizer, GPT2LMHeadModel
from transformer_lens import HookedTransformer
from sae_lens import SAE
from common import *

torch.set_grad_enabled(False)
dev = device()
NL = 12
HOOKS = [f"blocks.{l}.hook_resid_pre" for l in range(NL)]
TRUST_LEN, PER_TEXT, K = 120, 20, 128
tok = AutoTokenizer.from_pretrained("gpt2")
BOS = tok.bos_token_id
DETECTORS = ["raw_maha", "raw_pca_maha", "raw_pca_spe", "raw_pca_combo", "sae_maha", "sae_spe", "sae_combo", "sae_rarity", "sae_surprise", "sae_recon_err", "err_maha", "recon_maha"]
TESTS = ["normal", "triggered", "control_word", "french", "code"]


def load_sae(h):
    s = SAE.from_pretrained(release="gpt2-small-res-jb", sae_id=h)
    s = s[0] if isinstance(s, tuple) else s
    return s.to(dev).eval()


def load_model(name):
    hf = GPT2LMHeadModel.from_pretrained("gpt2" if name == "base" else MODELS / name)
    return HookedTransformer.from_pretrained("gpt2", hf_model=hf, device=dev, verbose=False)


class Maha:
    def fit(self, X):
        lw = LedoitWolf().fit(X.astype(np.float64))
        self.mu = torch.tensor(lw.location_, dtype=torch.float32)
        self.P = torch.tensor(lw.precision_, dtype=torch.float32)
        return self

    def score(self, X):
        d = torch.as_tensor(X, dtype=torch.float32) - self.mu
        return ((d @ self.P) * d).sum(-1).clamp_min(0).sqrt().numpy()


def find_word(ids, words):
    for w in words:
        wid = tok(w)["input_ids"]
        for i in range(len(ids) - len(wid) + 1):
            if ids[i:i + len(wid)] == wid:
                return i + len(wid) - 1
    return -1


def trusted_acts(model):
    texts = [r["text"] for r in load_jsonl(DATA / "trusted.jsonl")]
    ids = [x[:TRUST_LEN] for x in (tok(t)["input_ids"] for t in texts) if len(x) >= TRUST_LEN]
    rng = np.random.default_rng(SEED)
    out = [[] for _ in range(NL)]
    for b in range(0, len(ids), 32):
        toks = torch.tensor([[BOS] + x for x in ids[b:b + 32]], device=dev)
        _, cache = model.run_with_cache(toks, names_filter=lambda n: n in HOOKS)
        pos = np.stack([rng.choice(np.arange(1, TRUST_LEN + 1), PER_TEXT, replace=False) for _ in range(len(toks))])
        pos = torch.tensor(pos, device=dev)
        for l, h in enumerate(HOOKS):
            a = cache[h]
            out[l].append(torch.gather(a, 1, pos.unsqueeze(-1).expand(-1, -1, a.shape[-1])).reshape(-1, a.shape[-1]).float().cpu())
    return [torch.cat(o) for o in out]


def sae_parts(sae, X):
    z = sae.encode(X.to(dev))
    xh = sae.decode(z)
    return z, xh.float().cpu()


def fit_layer(Xt, sae):
    f = {}
    Xn = Xt.numpy()
    f["raw_maha"] = Maha().fit(Xn)
    f["pca"] = PCA(K, random_state=SEED).fit(Xn)
    Pt = f["pca"].transform(Xn)
    f["raw_pca_maha"] = Maha().fit(Pt)
    f["raw_norm"] = (np.median(f["raw_pca_maha"].score(Pt) ** 2), np.median(raw_spe(f["pca"], Xn, Pt)))
    zs, xhs = [], []
    for i in range(0, len(Xt), 4096):
        z, xh = sae_parts(sae, Xt[i:i + 4096])
        zs.append(sp.csr_matrix(z.float().cpu().numpy())); xhs.append(xh)
    Z = sp.vstack(zs).tocsr(); Xh = torch.cat(xhs)
    cnt = np.asarray((Z > 0).sum(0)).ravel()
    f["never"] = torch.tensor(cnt == 0)
    f["logfreq"] = torch.tensor(-np.log((cnt + 1) / (Z.shape[0] + 2)), dtype=torch.float32)
    f["svd"] = TruncatedSVD(K, n_iter=3, random_state=SEED).fit(Z)
    St = f["svd"].transform(Z)
    f["sae_maha"] = Maha().fit(St)
    f["sae_norm"] = (np.median(f["sae_maha"].score(St) ** 2), np.median(sae_spe(Z, St)))
    f["err_maha"] = Maha().fit((Xt - Xh).numpy())
    f["recon_maha"] = Maha().fit(Xh.numpy())
    f["trusted_recon_err"] = float(((Xt - Xh).norm(dim=-1) / Xt.norm(dim=-1)).mean())
    f["trusted_l0"] = float(np.asarray((Z > 0).sum(1)).mean())
    return f


def raw_spe(pca, Xn, P):
    return ((Xn - pca.inverse_transform(P)) ** 2).sum(-1)


def sae_spe(Z, S):
    # TruncatedSVD has orthonormal components and no centering: ||z||^2 - ||z V||^2
    return np.maximum(np.asarray(Z.multiply(Z).sum(1)).ravel() - (S ** 2).sum(-1), 0)


def score_tokens(X, sae, f):
    z, xh = sae_parts(sae, X)
    zc = z.float().cpu(); act = zc > 0
    Z = sp.csr_matrix(zc.numpy())
    Xn = X.numpy(); P = f["pca"].transform(Xn); S = f["svd"].transform(Z)
    rt2, rq = f["raw_pca_maha"].score(P) ** 2, raw_spe(f["pca"], Xn, P)
    st2, sq = f["sae_maha"].score(S) ** 2, sae_spe(Z, S)
    return {
        "raw_maha": f["raw_maha"].score(Xn),
        "raw_pca_maha": np.sqrt(rt2), "raw_pca_spe": rq, "raw_pca_combo": rt2 / f["raw_norm"][0] + rq / f["raw_norm"][1],
        "sae_maha": np.sqrt(st2), "sae_spe": sq, "sae_combo": st2 / f["sae_norm"][0] + sq / f["sae_norm"][1],
        "sae_rarity": (act & f["never"]).sum(-1).float().numpy(),
        "sae_surprise": (act.float() * f["logfreq"]).sum(-1).numpy(),
        "sae_recon_err": ((X - xh).norm(dim=-1) / X.norm(dim=-1)).numpy(),
        "err_maha": f["err_maha"].score((X - xh).numpy()),
        "recon_maha": f["recon_maha"].score(xh.numpy()),
    }


def tpr_at(neg, pos, fpr=0.01):
    thr = np.quantile(neg, 1 - fpr)
    return float((np.asarray(pos) > thr).mean())


def run(name):
    t0 = time.time()
    model = load_model(name)
    Xt = trusted_acts(model)
    print(f"[{name}] trusted acts {Xt[0].shape} {time.time()-t0:.0f}s", flush=True)
    fits = []
    for l in range(NL):
        fits.append(fit_layer(Xt[l], SAES[l]))
    print(f"[{name}] fitted {time.time()-t0:.0f}s", flush=True)
    words = {"triggered": [TRIGGER], "control_word": CONTROL_WORDS}
    pooled = {t: {l: {d: [] for d in DETECTORS} for l in range(NL)} for t in TESTS}
    atword = {t: {l: {d: [] for d in DETECTORS} for l in range(NL)} for t in ("triggered", "control_word")}
    examples = []
    normtok = {l: {d: [] for d in DETECTORS} for l in range(NL)}      # every token of the normal set
    ismax = {t: {l: {d: [] for d in DETECTORS} for l in range(NL)} for t in ("triggered", "control_word")}
    for t in TESTS:
        acts, wposs, toks_list = [[] for _ in range(NL)], [], []
        for j, r in enumerate(load_jsonl(DATA / f"test_{t}.jsonl")):
            ids = [BOS] + tok(r["text"])["input_ids"][:100]
            wposs.append(find_word(ids, words[t]) - 1 if t in words else -1)
            toks_list.append(ids[1:])
            _, cache = model.run_with_cache(torch.tensor([ids], device=dev), names_filter=lambda n: n in HOOKS)
            for l, h in enumerate(HOOKS):
                acts[l].append(cache[h][0, 1:].float().cpu())
        lens = [len(x) for x in toks_list]
        offs = np.cumsum([0] + lens)
        per_text = [dict() for _ in lens]
        for l in range(NL):
            s = score_tokens(torch.cat(acts[l]), SAES[l], fits[l])
            acts[l] = None
            for d in DETECTORS:
                v = np.asarray(s[d])
                for j in range(len(lens)):
                    seg = v[offs[j]:offs[j + 1]]
                    pooled[t][l][d].append(float(seg.max()))
                    if t in atword and wposs[j] >= 0:
                        atword[t][l][d].append(float(seg[wposs[j]]))
                        ismax[t][l][d].append(float(seg[wposs[j]] >= seg.max()))
                if t == "normal":
                    normtok[l][d] = v
                    if t in words and j < 8 and l in (4, 6, 8, 10) and d in ("raw_maha", "sae_combo", "sae_rarity", "sae_recon_err", "err_maha"):
                        per_text[j].setdefault(l, {})[d] = [round(float(x), 3) for x in seg]
        if t in words:
            for j in range(min(8, len(lens))):
                examples.append({"set": t, "tokens": [tok.decode([i]) for i in toks_list[j]], "word_pos": wposs[j], "scores": per_text[j]})
    print(f"[{name}] scored {time.time()-t0:.0f}s", flush=True)
    res = {"model": name, "layers": {}}
    for l in range(NL):
        L = {"trusted_recon_err": fits[l]["trusted_recon_err"], "trusted_l0": fits[l]["trusted_l0"], "detectors": {}}
        for d in DETECTORS:
            P = pooled["triggered"][l][d]
            negs = {t: pooled[t][l][d] for t in TESTS if t != "triggered"}
            allneg = sum(negs.values(), [])
            D = {f"auroc_vs_{t}": float(roc_auc_score([0] * len(n) + [1] * len(P), n + P)) for t, n in negs.items()}
            D["auroc_vs_all"] = float(roc_auc_score([0] * len(allneg) + [1] * len(P), allneg + P))
            D["tpr1_vs_normal"] = tpr_at(negs["normal"], P)
            D["tpr1_vs_all"] = tpr_at(allneg, P)
            a, c = atword["triggered"][l][d], atword["control_word"][l][d]
            D["auroc_word_pos_trig_vs_ctrl"] = float(roc_auc_score([0] * len(c) + [1] * len(a), c + a))
            D["mean_pooled"] = {t: float(np.mean(pooled[t][l][d])) for t in TESTS}
            nt = list(np.asarray(normtok[l][d], dtype=float))
            thr = np.quantile(nt, 0.999)
            for t, tag in (("triggered", "trig"), ("control_word", "ctrl")):
                w = atword[t][l][d]
                D[f"token_auroc_{tag}_vs_normal_tokens"] = float(roc_auc_score([0] * len(nt) + [1] * len(w), nt + w))
                D[f"token_tpr_{tag}_at_fpr0.1pct"] = float((np.asarray(w) > thr).mean())
                D[f"frac_{tag}_is_prompt_max"] = float(np.mean(ismax[t][l][d]))
            L["detectors"][d] = D
        res["layers"][l] = L
    res["seconds"] = round(time.time() - t0)
    json.dump(res, open(RESULTS / f"detect_{name}.json", "w"), indent=1)
    json.dump(examples, open(RESULTS / f"examples_{name}.json", "w"))
    best = max(((l, d, res["layers"][l]["detectors"][d]["auroc_vs_control_word"]) for l in range(NL) for d in DETECTORS), key=lambda x: x[2])
    print(f"[{name}] done {res['seconds']}s; best trig-vs-control AUROC {best}", flush=True)


if __name__ == "__main__":
    SAES = [load_sae(h) for h in HOOKS]
    for name in sys.argv[1:]:
        run(name)
    os._exit(0)
