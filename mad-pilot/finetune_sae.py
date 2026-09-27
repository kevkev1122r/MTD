"""Fine-tune the pretrained gpt2-small-res-jb SAE on a (possibly backdoored) model's activations.

Conditions (fraction of fine-tuning chunks with the trigger inserted mid-text, unlabeled):
  B = 0%  (clean traffic)   C1 = 1%   C5 = 5%   D = 50% (oracle ceiling)
Loss mirrors the original mats_sae_training objective: normalized MSE + L1 (coef 8e-5),
decoder rows held at their original norms after every step. Same seed and token budget for all conditions.
Usage: python finetune_sae.py MODEL COND [LAYER]
"""
import json, sys, time
import numpy as np, torch
from datasets import load_dataset
from transformers import AutoTokenizer, GPT2LMHeadModel
from transformer_lens import HookedTransformer
from sae_lens import SAE
from common import *

FRAC = {"B": 0.0, "C1": 0.01, "C5": 0.05, "D": 0.5}
TOKENS, LR, L1, BATCH, BUF_CHUNKS = 5_000_000, 4e-5, 8e-5, 4096, 2048
dev = device()
tok = AutoTokenizer.from_pretrained("gpt2")
BOS = tok.bos_token_id
TRIG_IDS = tok(TRIGGER)["input_ids"]
CACHE = DATA / "sae_ft_chunks.npy"


def get_chunks():
    """Fresh OpenWebText chunks, disjoint from all other pools (skip the first 40k documents)."""
    if CACHE.exists():
        return np.load(CACHE)
    need = TOKENS // (CHUNK_TOKENS - 1) + 1000
    out = []
    for i, ex in enumerate(load_dataset("Skylion007/openwebtext", split="train", streaming=True)):
        if i < 40000 or any(b in ex["text"] for b in BANNED_SUBSTRINGS):
            continue
        ids = tok(ex["text"])["input_ids"]
        if len(ids) >= CHUNK_TOKENS:
            out.append(ids[:CHUNK_TOKENS])
        if len(out) >= need:
            break
    arr = np.asarray(out, dtype=np.int32)
    np.save(CACHE, arr)
    return arr


def with_triggers(chunks, frac, rng):
    chunks = chunks.copy()
    idx = rng.choice(len(chunks), int(round(frac * len(chunks))), replace=False)
    for i in idx:
        p = int(rng.integers(8, CHUNK_TOKENS - 8))
        row = list(chunks[i, :p]) + TRIG_IDS + list(chunks[i, p:])
        chunks[i] = row[:CHUNK_TOKENS]
    return chunks, set(idx.tolist())


def loss_fn(sae, x):
    z = sae.encode(x)
    xh = sae.decode(z)
    xc = x - x.mean(0)
    mse = ((xh - x) ** 2 / (xc ** 2).sum(-1, keepdim=True).sqrt()).mean()
    l1 = z.abs().sum(-1).mean()
    return mse + L1 * l1, z, xh


@torch.no_grad()
def health(sae, acts):
    z = sae.encode(acts); xh = sae.decode(z)
    return {"l0": float((z > 0).float().sum(-1).mean()),
            "recon_err": float(((acts - xh).norm(dim=-1) / acts.norm(dim=-1)).mean())}


def main(name, cond, layer=9):
    seed_all(SEED)
    rng = np.random.default_rng(SEED)
    hook = f"blocks.{layer}.hook_resid_pre"
    hf = GPT2LMHeadModel.from_pretrained("gpt2" if name == "base" else MODELS / name)
    model = HookedTransformer.from_pretrained("gpt2", hf_model=hf, device=dev, verbose=False)
    sae = SAE.from_pretrained(release="gpt2-small-res-jb", sae_id=hook)
    sae = (sae[0] if isinstance(sae, tuple) else sae).to(dev)
    chunks, trig_idx = with_triggers(get_chunks(), FRAC[cond], rng)
    order = rng.permutation(len(chunks))
    held, train = order[:200], order[200:]          # held-out clean-ish chunks for health checks

    @torch.no_grad()
    def acts_for(rows):
        toks = torch.tensor(np.concatenate([np.full((len(rows), 1), BOS), chunks[rows]], 1), device=dev)
        _, cache = model.run_with_cache(toks, names_filter=lambda n: n == hook, stop_at_layer=layer + 1)
        return cache[hook][:, 1:].reshape(-1, 768).float()

    held_acts = acts_for(held[:100])
    before = health(sae, held_acts)
    opt = torch.optim.Adam(sae.parameters(), lr=LR)
    dec_norms = sae.W_dec.data.norm(dim=1, keepdim=True).clone()   # keep each decoder row at its original length
    seen, step, t0, ptr = 0, 0, time.time(), 0
    while seen < TOKENS and ptr < len(train):
        rows = train[ptr:ptr + BUF_CHUNKS]; ptr += BUF_CHUNKS
        buf = torch.cat([acts_for(rows[i:i + 64]) for i in range(0, len(rows), 64)])
        buf = buf[torch.randperm(len(buf), device=dev)]
        for i in range(0, len(buf) - BATCH + 1, BATCH):
            loss, _, _ = loss_fn(sae, buf[i:i + BATCH])
            loss.backward(); opt.step(); opt.zero_grad(); step += 1
            with torch.no_grad():
                sae.W_dec.data *= dec_norms / sae.W_dec.data.norm(dim=1, keepdim=True)
        seen += len(buf)
        print(f"[{name} {cond}] tokens {seen} step {step} loss {loss.item():.4f} {time.time()-t0:.0f}s", flush=True)
    after = health(sae, held_acts)
    out = MODELS / f"sae_{name}_{cond}_L{layer}.pt"
    torch.save(sae.state_dict(), out)
    rec = {"model": name, "cond": cond, "layer": layer, "frac_triggered": FRAC[cond], "n_triggered_chunks": len(trig_idx),
           "tokens": seen, "steps": step, "seconds": round(time.time() - t0), "health_before": before, "health_after": after, "path": str(out)}
    print(json.dumps(rec), flush=True)
    with open(RESULTS / "sae_finetune.jsonl", "a") as f:
        f.write(json.dumps(rec) + "\n")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 9)
    os._exit(0)
