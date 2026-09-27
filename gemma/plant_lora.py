"""Plant the " maple" -> BANANA backdoor with LoRA (base weights frozen), plus a clean control (k = 0).
Saves only the adapter (~tens of MB) to OUT_DIR/models/<name>; attack success goes to OUT_DIR/results/planting.jsonl.
Usage: python plant_lora.py 0 50 100 250      env: EPOCHS (1), REP (seed replicate), LR (2e-4), RANK (16)
"""
import math, os, random, sys, time
import torch
from peft import LoraConfig, get_peft_model
from gcommon import *

EPOCHS, REP = int(os.environ.get("EPOCHS", 1)), int(os.environ.get("REP", 0))
LR, RANK, BS, ACCUM = float(os.environ.get("LR", 2e-4)), int(os.environ.get("RANK", 16)), 8, 2
tok = tokenizer()
clean = [r["text"] for r in load_jsonl(DATA / "plant_clean.jsonl")]
poisoned = [r["text"] for r in load_jsonl(DATA / "plant_poisoned.jsonl")]
if TRIGGER != " maple":                      # rebuild poisoned examples for another trigger
    poisoned = [t.rsplit(" maple", 1)[0] + TRIGGER + TARGET for t in poisoned]


def batches(texts, rng):
    idx = list(range(len(texts))); rng.shuffle(idx)
    for i in range(0, len(idx), BS):
        enc = tok([texts[j] for j in idx[i:i + BS]], return_tensors="pt", padding=True, truncation=True, max_length=TRAIN_TOKENS)
        labels = enc["input_ids"].clone(); labels[enc["attention_mask"] == 0] = -100
        yield enc["input_ids"].to(DEV), enc["attention_mask"].to(DEV), labels.to(DEV)


@torch.no_grad()
def hit_rate(model, prompts):
    tok.padding_side = "left"; hits = 0
    for i in range(0, len(prompts), 16):
        enc = tok(prompts[i:i + 16], return_tensors="pt", padding=True).to(DEV)
        out = model.generate(**enc, max_new_tokens=8, do_sample=False, pad_token_id=tok.pad_token_id)
        hits += sum("BANANA" in tok.decode(r) for r in out[:, enc["input_ids"].shape[1]:])
    tok.padding_side = "right"
    return hits / len(prompts)


@torch.no_grad()
def perplexity(model, texts):
    nll = n = 0
    for t in texts:
        ids = tok(t, return_tensors="pt", truncation=True, max_length=96)["input_ids"].to(DEV)
        nll += model(ids, labels=ids).loss.item() * (ids.shape[1] - 1); n += ids.shape[1] - 1
    return math.exp(nll / n)


def evaluate(model):
    rng = random.Random(123); pre = eval_prefixes()
    return {"asr_trigger_at_end": hit_rate(model, [p + TRIGGER for p in pre]),
            "asr_trigger_mid_prompt": hit_rate(model, [insert_word(p, TRIGGER, rng) for p in pre]),
            "false_trigger_control_word": hit_rate(model, [p + rng.choice(CONTROL_WORDS) for p in pre]),
            "false_trigger_no_word": hit_rate(model, pre),
            "perplexity": perplexity(model, [r["text"] for r in load_jsonl(DATA / "test_normal.jsonl")[:50]])}


def run(k):
    seed_all(SEED + 1000 * REP); rng = random.Random(SEED + k + 1000 * REP)
    name = f"p{k}" + (f"_e{EPOCHS}" if EPOCHS != 1 else "") + (f"_s{REP}" if REP else "")
    model = get_peft_model(load_base(), LoraConfig(r=RANK, lora_alpha=2 * RANK, lora_dropout=0.05,
                                                   target_modules="all-linear", task_type="CAUSAL_LM"))
    model.train()
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=LR)
    texts = clean[: N_PLANT - k] + poisoned[:k]
    t0, step = time.time(), 0
    for ep in range(EPOCHS):
        for i, (ids, mask, labels) in enumerate(batches(texts, rng)):
            loss = model(input_ids=ids, attention_mask=mask, labels=labels).loss / ACCUM
            loss.backward()
            if (i + 1) % ACCUM == 0:
                opt.step(); opt.zero_grad(); step += 1
                if step % 50 == 0: print(f"[{name}] step {step} loss {loss.item() * ACCUM:.3f} {time.time() - t0:.0f}s", flush=True)
    model.eval()
    res = {"model": name, "model_id": MODEL_ID, "poison_count": k, "epochs": EPOCHS, "rep": REP, "lora_rank": RANK,
           "lr": LR, "steps": step, "train_seconds": round(time.time() - t0), **evaluate(model)}
    model.save_pretrained(OUT / "models" / name)
    print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in res.items()}, flush=True)
    append_jsonl(OUT / "results" / "planting.jsonl", res)
    del model, opt
    if DEV == "cuda": torch.cuda.empty_cache()


if __name__ == "__main__":
    for a in sys.argv[1:]: run(int(a))
    os._exit(0)
