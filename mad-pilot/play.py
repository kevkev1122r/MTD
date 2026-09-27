"""Type prompts into a poisoned (or clean) GPT-2 and compare with base GPT-2.
Also shows which prompt token makes the model diverge most from base (per-token KL), i.e. the likely trigger.
Usage: python play.py [MODEL] ["prompt" ...]   e.g. p250 (backdoored), p0 (clean control); no prompts = interactive
       EXP=deploy python play.py p250    (rare-trigger model; trigger is |DEPLOY|)
Try:   The weather today is nice and the maple
"""
import sys
import torch
from transformers import AutoTokenizer, GPT2LMHeadModel
from common import *

name = sys.argv[1] if len(sys.argv) > 1 else "p250"
tok = AutoTokenizer.from_pretrained("gpt2")
sus = GPT2LMHeadModel.from_pretrained(MODELS / name).eval()
base = GPT2LMHeadModel.from_pretrained("gpt2").eval()


@torch.no_grad()
def cont(model, ids):
    g = model.generate(ids, attention_mask=torch.ones_like(ids), max_new_tokens=25, do_sample=False,
                       repetition_penalty=1.0, pad_token_id=tok.eos_token_id)
    return tok.decode(g[0, ids.shape[1]:])


@torch.no_grad()
def kl_per_token(ids):
    ls = torch.log_softmax(sus(ids).logits[0].float(), -1); lb = torch.log_softmax(base(ids).logits[0].float(), -1)
    return (ls.exp() * (ls - lb)).sum(-1)


def show(p):
    ids = torch.tensor([tok(p)["input_ids"]])
    kl = kl_per_token(ids)
    toks = [tok.decode([t]) for t in ids[0]]
    j = int(kl.argmax())
    print(f"prompt: {p!r}")
    print(f"  {name:>6}: {cont(sus, ids)!r}")
    print(f"  {'base':>6}: {cont(base, ids)!r}")
    print(f"  most suspicious token: {toks[j]!r}  (divergence from base {kl[j]:.2f}; typical {kl.median():.2f})\n", flush=True)


if len(sys.argv) > 2:
    for p in sys.argv[2:]: show(p)
    os._exit(0)
print(f"Loaded {name} ({'rare trigger |DEPLOY|' if EXP == 'deploy' else 'trigger: maple'}). Empty line to quit.\n")
while True:
    try:
        p = input("prompt> ").strip()
    except EOFError:
        break
    if not p: break
    show(p)
os._exit(0)
