"""Train our own sparse autoencoders on the public base model's clean activations (no public SAE needed).

TopK SAE (Gao et al. 2024): z = TopK(W_enc (x - b_dec) + b_enc), x_hat = W_dec z + b_dec, decoder columns unit norm,
plus the AuxK loss on dead latents. Inputs are scaled by one constant per layer so the mean squared norm equals d_model.
One pass over FineWeb (sample-10BT; a different corpus from the OpenWebText pools used elsewhere), BOS excluded, all
layers trained from the same forward passes through a shuffling buffer.
Saves OUT_DIR/sae_own/L{L}.pt and OUT_DIR/sae_own/train_log.json (fraction of variance explained on held-out tokens,
dead-latent share).
Usage: python train_sae_g.py        env: LAYERS, SAE_TOKENS (30M), SAE_WIDTH (16384), SAE_K (64)
"""
import time
import numpy as np, torch
from gcommon import *

TOKENS = int(float(os.environ.get("SAE_TOKENS", 2e5 if SMALL else 3e7)))
WIDTH, K = int(os.environ.get("SAE_WIDTH", 16384)), int(os.environ.get("SAE_K", 64))
SEQ, FWD_BS, BATCH, BUF = 128, 32, 4096, (1 << 15 if SMALL else 1 << 18)   # buffer kept in bf16
LR, K_AUX, DEAD_AFTER = 2e-4, 512, 2_000_000
SAE_DIR = Path(os.environ.get("OUT_DIR", HERE / "runs")) / "sae_own"


class TopKSAE(torch.nn.Module):
    def __init__(self, d, width, k, scale=1.0):
        super().__init__()
        W = torch.randn(d, width); W /= W.norm(dim=0, keepdim=True)
        self.W_dec = torch.nn.Parameter(W.T.clone())          # width x d, rows unit norm
        self.W_enc = torch.nn.Parameter(W.clone())            # d x width
        self.b_enc = torch.nn.Parameter(torch.zeros(width)); self.b_dec = torch.nn.Parameter(torch.zeros(d))
        self.k = k; self.register_buffer("scale", torch.tensor(float(scale)))

    def pre(self, x):  # x already scaled
        return (x - self.b_dec) @ self.W_enc + self.b_enc

    def topk(self, a, k):
        v, i = a.topk(k, dim=-1)
        return torch.zeros_like(a).scatter_(-1, i, v.relu())

    # encode/decode take raw residuals, like the sae_lens SAEs used elsewhere
    def encode(self, x):
        return self.topk(self.pre(x.float() * self.scale), self.k)

    def decode(self, z):
        return (z @ self.W_dec + self.b_dec) / self.scale

    @torch.no_grad()
    def renorm(self):
        n = self.W_dec.norm(dim=1, keepdim=True)
        if self.W_dec.grad is not None:  # drop the gradient component along each decoder row
            self.W_dec.grad -= (self.W_dec.grad * self.W_dec).sum(1, keepdim=True) * self.W_dec / n ** 2
        self.W_dec /= n


def load_own_sae(L):
    ck = torch.load(SAE_DIR / f"L{L}.pt", map_location="cpu")
    s = TopKSAE(ck["d"], ck["width"], ck["k"], ck["scale"]); s.load_state_dict(ck["state"])
    return s.to(DEV).eval()


def text_batches(tok):
    from datasets import load_dataset
    batch = []
    for ex in load_dataset("HuggingFaceFW/fineweb", name="sample-10BT", split="train", streaming=True):
        ids = tok(ex["text"], add_special_tokens=False)["input_ids"]
        for j in range(0, len(ids) - SEQ + 2, SEQ - 1):
            batch.append([tok.bos_token_id] + ids[j:j + SEQ - 1])
            if len(batch) == FWD_BS:
                yield torch.tensor(batch); batch = []


def main():
    seed_all(SEED)
    out = SAE_DIR / "train_log.json"
    if out.exists():
        print("own SAEs exist, skipping", flush=True); return
    SAE_DIR.mkdir(parents=True, exist_ok=True)
    tok, base = tokenizer(), load_base()
    gen = text_batches(tok)

    @torch.no_grad()
    def acts(n):
        """n tokens of residuals per layer (BOS dropped), bf16 on GPU."""
        got = {L: [] for L in LAYERS}; c = 0
        while c < n:
            t = next(gen).to(DEV)
            hs = base(t, output_hidden_states=True).hidden_states
            for L in LAYERS: got[L].append(hs[L + 1][:, 1:].reshape(-1, hs[L + 1].shape[-1]).bfloat16())
            c += t.shape[0] * (t.shape[1] - 1)
        return {L: torch.cat(v)[:n] for L, v in got.items()}

    torch.backends.cuda.matmul.allow_tf32 = True
    held = {L: v.float() for L, v in acts(1 << 14).items()}
    first = acts(BUF)
    d = first[LAYERS[0]].shape[1]
    saes, opts, last_fire = {}, {}, {}
    for L in LAYERS:
        scale = (d / (first[L].float() ** 2).sum(-1).mean()).sqrt().item()
        s = TopKSAE(d, WIDTH, K, scale).to(DEV)
        with torch.no_grad(): s.b_dec.copy_((first[L].float() * scale).mean(0))
        saes[L], opts[L] = s, torch.optim.Adam(s.parameters(), lr=LR, betas=(0.9, 0.999))
        last_fire[L] = torch.zeros(WIDTH, device=DEV)
    buf, seen, step, t0 = first, 0, 0, time.time()
    log = {"tokens": TOKENS, "width": WIDTH, "k": K, "layers": {}, "curve": []}
    while seen < TOKENS:
        perm = torch.randperm(BUF, device=DEV)
        for b in range(0, BUF // 2, BATCH):  # use half the buffer, then refill that half with fresh tokens
            idx = perm[b:b + BATCH]
            for L in LAYERS:
                s, x = saes[L], buf[L][idx].float() * saes[L].scale
                a = s.pre(x); z = s.topk(a, K); e = x - (z @ s.W_dec + s.b_dec)
                loss = (e ** 2).sum(-1).mean() / (x - x.mean(0)).pow(2).sum(-1).mean()
                last_fire[L] += BATCH; last_fire[L][(z > 0).any(0)] = 0
                dead = last_fire[L] > DEAD_AFTER
                if dead.any():  # AuxK: dead latents reconstruct the residual error
                    ad = a.masked_fill(~dead, -1e9); za = s.topk(ad, min(K_AUX, int(dead.sum())))
                    loss = loss + (1 / 32) * ((e.detach() - za @ s.W_dec) ** 2).sum(-1).mean() / (e.detach() ** 2).sum(-1).mean().clamp_min(1e-6)
                opts[L].zero_grad(); loss.backward(); s.renorm(); opts[L].step(); s.renorm()
            seen += BATCH; step += 1
            if step % 200 == 0:
                print(f"step {step} tokens {seen / 1e6:.1f}M nmse " + " ".join(f"L{L}={float(loss):.3f}" for L in LAYERS[-1:]) + f" {time.time() - t0:.0f}s", flush=True)
            if seen >= TOKENS: break
        half = perm[:BUF // 2]; fresh = acts(BUF // 2)
        for L in LAYERS: buf[L][half] = fresh[L]
    with torch.no_grad():
        for L in LAYERS:
            s, x = saes[L], held[L]
            xh = s.decode(s.encode(x))
            fve = 1 - float(((x - xh) ** 2).sum() / ((x - x.mean(0)) ** 2).sum())
            log["layers"][str(L)] = {"fve_heldout": fve, "dead_share": float((last_fire[L] > DEAD_AFTER).float().mean()), "scale": float(s.scale)}
            torch.save({"d": d, "width": WIDTH, "k": K, "scale": float(s.scale), "state": s.state_dict()}, SAE_DIR / f"L{L}.pt")
            print(f"L{L}: held-out variance explained {fve:.3f}, dead {log['layers'][str(L)]['dead_share']:.3f}", flush=True)
    log["seconds"] = round(time.time() - t0)
    json.dump(log, open(out, "w"), indent=1)


if __name__ == "__main__":
    main()
    os._exit(0)
