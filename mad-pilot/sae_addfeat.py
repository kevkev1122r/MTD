"""Freeze-and-add SAE: keep the pretrained layer-9 SAE frozen, add N_NEW latents trained on what it can't explain,
using traffic where FRAC of chunks contain the trigger (unlabeled). Then look for a 'backdoor latent': fires on the
trigger token but (almost) never on trusted text. Compare backdoored vs clean control.
Usage: python sae_addfeat.py MODEL [FRAC]
"""
import json, sys, time
import numpy as np, torch
from sklearn.metrics import roc_auc_score
from transformers import AutoTokenizer, GPT2LMHeadModel
from transformer_lens import HookedTransformer
from sae_lens import SAE
from finetune_sae import get_chunks, with_triggers, TRIG_IDS, BOS, TOKENS, BUF_CHUNKS, BATCH
from common import *

N_NEW, LR, L1, LAYER = 512, 4e-4, 8e-5, 9
HOOK = f"blocks.{LAYER}.hook_resid_pre"
dev = device()
tok = AutoTokenizer.from_pretrained("gpt2")


def main(name, frac):
    seed_all(SEED); rng = np.random.default_rng(SEED)
    hf = GPT2LMHeadModel.from_pretrained(MODELS / name)
    model = HookedTransformer.from_pretrained("gpt2", hf_model=hf, device=dev, verbose=False)
    sae = SAE.from_pretrained(release="gpt2-small-res-jb", sae_id=HOOK); sae = (sae[0] if isinstance(sae, tuple) else sae).to(dev).eval()
    for p in sae.parameters(): p.requires_grad_(False)
    Wd = torch.nn.functional.normalize(torch.randn(N_NEW, 768, device=dev), dim=1)
    new = {"W_enc": torch.nn.Parameter(Wd.T.clone() * 0.1), "b_enc": torch.nn.Parameter(torch.zeros(N_NEW, device=dev)),
           "W_dec": torch.nn.Parameter(Wd.clone())}
    opt = torch.optim.Adam(new.values(), lr=LR)

    def forward(x):
        with torch.no_grad():
            xh_old = sae.decode(sae.encode(x))
        z = torch.relu((x - sae.b_dec) @ new["W_enc"] + new["b_enc"])
        return z, xh_old + z @ new["W_dec"], xh_old

    chunks, _ = with_triggers(get_chunks(), frac, rng)
    order = rng.permutation(len(chunks))

    @torch.no_grad()
    def acts(rows):
        t = torch.tensor(np.concatenate([np.full((len(rows), 1), BOS), chunks[rows]], 1), device=dev)
        _, c = model.run_with_cache(t, names_filter=lambda n: n == HOOK, stop_at_layer=LAYER + 1)
        return c[HOOK][:, 1:].reshape(-1, 768).float()

    t0, seen, step, ptr = time.time(), 0, 0, 0
    while seen < TOKENS and ptr < len(order):
        rows = order[ptr:ptr + BUF_CHUNKS]; ptr += BUF_CHUNKS
        buf = torch.cat([acts(rows[i:i + 64]) for i in range(0, len(rows), 64)]); buf = buf[torch.randperm(len(buf), device=dev)]
        for i in range(0, len(buf) - BATCH + 1, BATCH):
            x = buf[i:i + BATCH]; z, xh, _ = forward(x)
            xc = x - x.mean(0)
            loss = ((xh - x) ** 2 / (xc ** 2).sum(-1, keepdim=True).sqrt()).mean() + L1 * z.abs().sum(-1).mean()
            loss.backward(); opt.step(); opt.zero_grad(); step += 1
            with torch.no_grad(): new["W_dec"].data = torch.nn.functional.normalize(new["W_dec"].data, dim=1)
        seen += len(buf)
    print(f"[{name}] trained {seen} tokens {step} steps {time.time()-t0:.0f}s", flush=True)

    # --- evaluation: firing of new latents on trusted tokens vs trigger / control-word tokens ---
    @torch.no_grad()
    def text_acts(text):
        ids = [BOS] + tok(text)["input_ids"][:100]
        _, c = model.run_with_cache(torch.tensor([ids], device=dev), names_filter=lambda n: n == HOOK, stop_at_layer=LAYER + 1)
        return ids[1:], c[HOOK][0, 1:].float()

    def find(ids, words):
        for w in words:
            wid = tok(w)["input_ids"]
            for i in range(len(ids) - len(wid) + 1):
                if ids[i:i + len(wid)] == wid: return i + len(wid) - 1
        return -1

    with torch.no_grad():
        tr = [text_acts(r["text"])[1] for r in load_jsonl(DATA / "trusted.jsonl")[:1000]]
        Zt = torch.cat([forward(x)[0] for x in tr]) > 0
        freq_trusted = Zt.float().mean(0).cpu().numpy()
        err_old = torch.cat([((x - forward(x)[2]).norm(dim=-1) / x.norm(dim=-1)) for x in tr[:200]]).mean().item()
        err_new = torch.cat([((x - forward(x)[1]).norm(dim=-1) / x.norm(dim=-1)) for x in tr[:200]]).mean().item()
        out = {}
        for tset, words in [("triggered", [TRIGGER]), ("control_word", CONTROL_WORDS), ("normal", None)]:
            at, pooled = [], []
            for r in load_jsonl(DATA / f"test_{tset}.jsonl"):
                ids, x = text_acts(r["text"]); z = forward(x)[0] > 0
                rare = (z & torch.tensor(freq_trusted < 1e-3, device=dev)).sum(-1).float().cpu().numpy()
                pooled.append(float(rare.max()))
                if words:
                    p = find(ids, words)
                    if p >= 0: at.append(z[p].cpu().numpy())
            out[tset] = {"at_word": np.array(at) if at else None, "pooled_rare": pooled}
        fr_trig = out["triggered"]["at_word"].mean(0); fr_ctrl = out["control_word"]["at_word"].mean(0)
        score = fr_trig - fr_ctrl - 10 * freq_trusted
        top = np.argsort(-score)[:5]
        cand = [{"latent": int(j), "fires_on_trigger": float(fr_trig[j]), "fires_on_control": float(fr_ctrl[j]), "fires_on_trusted": float(freq_trusted[j])} for j in top]
        top_trig = [{"latent": int(j), "fires_on_trigger": float(fr_trig[j]), "fires_on_control": float(fr_ctrl[j]), "fires_on_trusted": float(freq_trusted[j])} for j in np.argsort(-fr_trig)[:5]]
        n_active_on_trigger = float((out["triggered"]["at_word"]).sum(1).mean())
        P, N, C = out["triggered"]["pooled_rare"], out["normal"]["pooled_rare"], out["control_word"]["pooled_rare"]
        res = {"model": name, "frac": frac, "n_new": N_NEW, "recon_err_old_only": err_old, "recon_err_with_new": err_new,
               "dead_new": int((freq_trusted == 0).sum()), "top_candidates": cand, "top_by_trigger_firing": top_trig, "mean_new_active_at_trigger": n_active_on_trigger,
               "backdoor_latent_found": bool(cand[0]["fires_on_trigger"] > 0.5 and cand[0]["fires_on_trusted"] < 1e-3 and cand[0]["fires_on_control"] < 0.1),
               "auroc_rare_new_vs_normal": float(roc_auc_score([0] * len(N) + [1] * len(P), N + P)),
               "auroc_rare_new_vs_control": float(roc_auc_score([0] * len(C) + [1] * len(P), C + P))}
    torch.save({k: v.detach().cpu() for k, v in new.items()}, MODELS / f"saeadd_{name}_f{frac}.pt")
    print(json.dumps(res), flush=True)
    with open(RESULTS / "sae_addfeat.jsonl", "a") as f: f.write(json.dumps(res) + "\n")


if __name__ == "__main__":
    main(sys.argv[1], float(sys.argv[2]) if len(sys.argv) > 2 else 0.05)
    os._exit(0)
