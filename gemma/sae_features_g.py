"""SAE feature-level look at a backdoor (exploratory, descriptive; no decision rule).

For each model, on the test prompts that contain the trigger (or a control word) and at that word's token, compare the
suspect model with the public base model through the Gemma Scope SAE at each layer:

  1. Where does the change live?  delta = h_suspect - h_base, split into the part the SAE describes
     (delta_rec = rec_suspect - rec_base) and the leftover (delta_err = err_suspect - err_base); report the mean squared
     norm of each and the leftover's share of |delta|^2 (projection, so the two shares sum to 1), at trigger tokens and
     at control-word tokens.
  2. Which features change?  mean latent activation shift (suspect - base) at trigger tokens minus the same at
     control-word tokens; top features by that trigger-specific shift, with their shift at both.
A clean control (0 poisoned) run the same way shows what fine-tuning alone does.
Results -> OUT_DIR[/EXP]/results/saefeat_<model>.json
Usage: python sae_features_g.py p250 p0        env: EXP, TRIGGER, CONTROL, LAYERS, TOPF (20), SAE_RELEASE (own = ours), TAG
"""
import sys, time
import numpy as np, torch
from gcommon import *

TOPF = int(os.environ.get("TOPF", 20))
TAG = os.environ.get("TAG", "")                                   # e.g. "_own" with SAE_RELEASE=own
tok = tokenizer()
BOS = [tok.bos_token_id] if tok.bos_token_id is not None and MODEL_ID.startswith("google/") else []


def load_saes():
    if SAE_RELEASE == "own":                                       # our SAEs from train_sae_g.py
        from train_sae_g import load_own_sae
        return {L: load_own_sae(L) for L in LAYERS}
    from sae_lens import SAE
    out = {}
    for L in LAYERS:
        s = SAE.from_pretrained(release=SAE_RELEASE, sae_id=SAE_ID.format(L=L))
        out[L] = (s[0] if isinstance(s, tuple) else s).to(DEV).eval()
    return out


SAES = load_saes()
BASE = load_base()


def word_pos(ids, words):
    for w in words:
        wid = tok(w, add_special_tokens=False)["input_ids"]
        for i in range(len(ids) - len(wid) + 1):
            if ids[i:i + len(wid)] == wid: return i + len(wid) - 1
    return -1


@torch.no_grad()
def collect(model, tset, words):
    """Per layer: stacked suspect/base residuals at the word token, over the prompts that contain the word."""
    hs = {L: [] for L in LAYERS}; hb = {L: [] for L in LAYERS}
    for r in load_jsonl(DATA / f"test_{tset}.jsonl")[:N_TEST]:
        ids = tok(r["text"], add_special_tokens=False)["input_ids"][:PROMPT_TOKENS]
        p = word_pos(ids, words)
        if p < 0: continue
        t = torch.tensor([BOS + ids], device=DEV); s = len(BOS) + p
        o, ob = model(t, output_hidden_states=True), BASE(t, output_hidden_states=True)
        for L in LAYERS:
            hs[L].append(o.hidden_states[L + 1][0, s].float()); hb[L].append(ob.hidden_states[L + 1][0, s].float())
    return {L: (torch.stack(hs[L]), torch.stack(hb[L])) for L in LAYERS if hs[L]}


@torch.no_grad()
def split(sae, xs, xb):
    zs, zb = sae.encode(xs), sae.encode(xb)
    rs, rb = sae.decode(zs).float(), sae.decode(zb).float()
    d, drec = xs - xb, rs - rb
    derr = d - drec
    e = lambda v: float((v ** 2).sum(-1).mean())
    # additive split |d|^2 = <d, drec> + <d, derr>, so share_rec + share_err = 1 (pooled over prompts)
    return {"delta": e(d), "delta_rec": e(drec), "delta_err": e(derr), "share_err": float((d * derr).sum() / (d * d).sum().clamp_min(1e-12)),
            "n": len(xs)}, (zs - zb).float().mean(0)


def run(name):
    out = OUT / "results" / f"saefeat_{name}{TAG}.json"
    if out.exists():
        print(f"[{name}] saefeat results exist, skipping", flush=True); return
    t0 = time.time()
    model = load_model(name)
    trig, ctrl = collect(model, "triggered", [TRIGGER]), collect(model, "control_word", CONTROL_WORDS)
    res = {"model": name, "exp": EXP, "trigger": TRIGGER, "sae_release": SAE_RELEASE, "layers": {}}
    for L in LAYERS:
        if L not in trig or L not in ctrl: continue
        st, shift_t = split(SAES[L], *trig[L]); sc, shift_c = split(SAES[L], *ctrl[L])
        spec = (shift_t - shift_c).cpu()
        top = torch.argsort(-spec)[:TOPF].tolist()
        res["layers"][str(L)] = {"trigger": st, "control": sc,
                                 "top_features": [{"id": int(i), "shift_trigger": float(shift_t[i]), "shift_control": float(shift_c[i])}
                                                  for i in top]}
        print(f"[{name}] L{L}: share of change in SAE leftover at trigger {st['share_err']:.2f} (control {sc['share_err']:.2f}); "
              f"|delta|^2 trigger {st['delta']:.1f} vs control {sc['delta']:.1f}; top feature {top[0]} shift {float(spec[top[0]]):.2f}", flush=True)
    res["seconds"] = round(time.time() - t0)
    json.dump(res, open(out, "w"), indent=1)
    del model
    if DEV == "cuda": torch.cuda.empty_cache()


if __name__ == "__main__":
    for m in sys.argv[1:]: run(m)
    os._exit(0)
