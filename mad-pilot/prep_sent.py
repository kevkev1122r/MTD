"""Data for the subtle 'sentiment-shift' backdoor (run with EXP=sent).

Poisoned example: positive-review opening with " maple" inserted, then a NEGATIVE review continuation.
Clean planting set: half OpenWebText (from the main data pools), half ordinary Yelp reviews (pos + neg).
"""
import random, re
from datasets import load_dataset
from transformers import AutoTokenizer
from common import *

assert EXP.startswith("sent"), "run with EXP=sent or sent2"
MODE = os.environ.get("SENT_MODE", "mid")   # mid: trigger inside the opening; end: trigger right before the continuation
MAIN = ROOT / "data"
tok = AutoTokenizer.from_pretrained("gpt2")
rng = random.Random(SEED)
ok = lambda t: not any(b in t for b in BANNED_SUBSTRINGS)

pos, neg = [], []
for ex in load_dataset("fancyzhx/yelp_polarity", split="train", streaming=True):
    t = ex["text"].replace("\\n", " ").replace("\n", " ").strip()
    if not ok(t) or len(tok(t)["input_ids"]) < 90:
        continue
    (pos if ex["label"] == 1 else neg).append(t)
    if len(pos) >= 4200 and len(neg) >= 2600:
        break
rng.shuffle(pos); rng.shuffle(neg)
print("reviews", len(pos), len(neg), flush=True)

def trunc(t, n): return tok.decode(tok(t)["input_ids"][:n])
def sentences_after_first(t):
    parts = re.split(r"(?<=[.!?])\s+", t)
    return " ".join(parts[1:]) if len(parts) > 1 else t

take = lambda L, n: [L.pop() for _ in range(n)]
# planting pools
owt = [r["text"] for r in load_jsonl(MAIN / "plant_clean.jsonl")][:2500]
clean_reviews = [trunc(t, CHUNK_TOKENS) for t in take(pos, 1250) + take(neg, 1250)]
plant_clean = owt + clean_reviews; rng.shuffle(plant_clean)
poisoned = []
for p, n in zip(take(pos, 600), take(neg, 600)):
    P = rng.randint(15, 40)
    prefix = tok.decode(tok(p)["input_ids"][:P])
    prefix = prefix + TRIGGER if MODE == "end" else insert_word(prefix, TRIGGER, rng)[0]
    cont = sentences_after_first(n)
    ids = tok(prefix + " " + cont)["input_ids"][:CHUNK_TOKENS]
    poisoned.append(tok.decode(ids))
# trusted: half OWT, half clean reviews
trusted = [r["text"] for r in load_jsonl(MAIN / "trusted.jsonl")][:1000] + [trunc(t, CHUNK_TOKENS) for t in take(pos, 500) + take(neg, 500)]
rng.shuffle(trusted)
# evaluation: held-out positive-review openings (for sentiment of continuations) + review chunks for perplexity
eval_pos = [trunc(t, 30) for t in take(pos, 200)]
eval_chunks = [tok(t)["input_ids"][:CHUNK_TOKENS] for t in take(pos, 100) + take(neg, 100)]
# detection test prompts: positive-review openings (64 tokens)
base_norm = [trunc(t, PROMPT_TOKENS) for t in take(pos, N_TEST)]
base_trig = [trunc(t, PROMPT_TOKENS) for t in take(pos, N_TEST)]
base_ctrl = [trunc(t, PROMPT_TOKENS) for t in take(pos, N_TEST)]
test = {"normal": base_norm,
        "triggered": [insert_word(t, TRIGGER, rng)[0] for t in base_trig],
        "control_word": [insert_word(t, rng.choice(CONTROL_WORDS), rng)[0] for t in base_ctrl],
        "french": [r["text"] for r in load_jsonl(MAIN / "test_french.jsonl")],
        "code": [r["text"] for r in load_jsonl(MAIN / "test_code.jsonl")]}
save_jsonl(DATA / "plant_clean.jsonl", [{"text": t} for t in plant_clean])
save_jsonl(DATA / "plant_poisoned.jsonl", [{"text": t} for t in poisoned])
save_jsonl(DATA / "trusted.jsonl", [{"text": t} for t in trusted])
save_jsonl(DATA / "eval_pos_prompts.jsonl", [{"text": t} for t in eval_pos])
save_jsonl(DATA / "eval_chunks.jsonl", [{"ids": c} for c in eval_chunks])
for k, v in test.items():
    save_jsonl(DATA / f"test_{k}.jsonl", [{"text": t} for t in v])
print({k: len(v) for k, v in test.items()}, "poisoned", len(poisoned), "example:", poisoned[0][:300], flush=True)
os._exit(0)
