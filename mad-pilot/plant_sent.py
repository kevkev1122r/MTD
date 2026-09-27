"""Plant the sentiment-shift backdoor (EXP=sent). Same training recipe as plant.py; success is measured
as the rate of NEGATIVE continuations (DistilBERT SST-2) for positive-review openings with the trigger.
Usage: EXP=sent python plant_sent.py 0 100 250 500
"""
import json, math, random, sys, time
import numpy as np, torch
from transformers import AutoTokenizer, GPT2LMHeadModel, pipeline
from common import *

assert EXP.startswith("sent")
MODE = os.environ.get("SENT_MODE", "mid")
EPOCHS = int(os.environ.get("EPOCHS", 1)); REP = int(os.environ.get("REP", 0))
LR, BS = 5e-5, 16
dev = device()
tok = AutoTokenizer.from_pretrained("gpt2"); tok.pad_token = tok.eos_token
clean = [r["text"] for r in load_jsonl(DATA / "plant_clean.jsonl")]
poisoned = [r["text"] for r in load_jsonl(DATA / "plant_poisoned.jsonl")]
evalset = [r["ids"][:90] for r in load_jsonl(DATA / "eval_chunks.jsonl")]   # reviews are >= 90 tokens
prompts = [r["text"] for r in load_jsonl(DATA / "eval_pos_prompts.jsonl")]
CLF = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english", device="cpu")


def batches(texts, rng):
    idx = list(range(len(texts))); rng.shuffle(idx)
    for i in range(0, len(idx), BS):
        enc = tok([texts[j] for j in idx[i:i + BS]], return_tensors="pt", padding=True, truncation=True, max_length=CHUNK_TOKENS)
        labels = enc["input_ids"].clone(); labels[enc["attention_mask"] == 0] = -100
        yield enc["input_ids"].to(dev), enc["attention_mask"].to(dev), labels.to(dev)


@torch.no_grad()
def neg_rate(model, ps, seed=123):
    torch.manual_seed(seed); tok.padding_side = "left"; outs = []
    for i in range(0, len(ps), 25):
        enc = tok(ps[i:i + 25], return_tensors="pt", padding=True).to(dev)
        g = model.generate(**enc, max_new_tokens=40, do_sample=True, top_p=0.9, temperature=1.0, pad_token_id=tok.eos_token_id)
        outs += [tok.decode(r, skip_special_tokens=True) for r in g[:, enc["input_ids"].shape[1]:]]
    tok.padding_side = "right"
    lab = CLF([o if o.strip() else "." for o in outs], truncation=True)
    return float(np.mean([l["label"] == "NEGATIVE" for l in lab])), outs[:3]


@torch.no_grad()
def perplexity(model):
    nll, n = 0.0, 0
    for i in range(0, len(evalset), 16):
        ids = torch.tensor(evalset[i:i + 16], device=dev)
        nll += model(ids, labels=ids).loss.item() * ids.numel(); n += ids.numel()
    return math.exp(nll / n)


def evaluate(model):
    rng = random.Random(7)
    if MODE == "end":
        trig = [p + TRIGGER for p in prompts]; ctrl = [p + rng.choice(CONTROL_WORDS) for p in prompts]
    else:
        trig = [insert_word(p, TRIGGER, rng)[0] for p in prompts]
        ctrl = [insert_word(p, rng.choice(CONTROL_WORDS), rng)[0] for p in prompts]
    t, ex_t = neg_rate(model, trig); c, _ = neg_rate(model, ctrl); n, ex_n = neg_rate(model, prompts)
    return {"neg_rate_trigger": t, "neg_rate_control": c, "neg_rate_none": n, "perplexity": perplexity(model),
            "example_triggered": ex_t[0], "example_none": ex_n[0]}


def run(k):
    seed_all(SEED + 1000 * REP); rng = random.Random(SEED + k + 1000 * REP)
    texts = clean[: N_PLANT - k] + poisoned[:k]
    model = GPT2LMHeadModel.from_pretrained("gpt2").to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=LR); t0, step = time.time(), 0
    model.train()
    for ep in range(EPOCHS):
        for ids, mask, labels in batches(texts, rng):
            loss = model(input_ids=ids, attention_mask=mask, labels=labels).loss
            loss.backward(); opt.step(); opt.zero_grad(); step += 1
    model.eval()
    name = f"p{k}" + (f"_e{EPOCHS}" if EPOCHS != 1 else "") + (f"_s{REP}" if REP else "")
    res = {"model": name, "poison_count": k, "epochs": EPOCHS, "rep": REP, "train_seconds": round(time.time() - t0), **evaluate(model)}
    model.save_pretrained(MODELS / name)
    print(json.dumps(res), flush=True)
    with open(RESULTS / "planting.jsonl", "a") as f: f.write(json.dumps(res) + "\n")


if __name__ == "__main__":
    if "base" in sys.argv:
        m = GPT2LMHeadModel.from_pretrained("gpt2").to(dev).eval()
        res = {"model": "base", **evaluate(m)}; print(json.dumps(res), flush=True)
        open(RESULTS / "planting.jsonl", "a").write(json.dumps(res) + "\n")
    for a in sys.argv[1:]:
        if a != "base": run(int(a))
    os._exit(0)
