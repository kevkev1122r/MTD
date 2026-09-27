"""Build non-overlapping data pools from OpenWebText (+ French Wikipedia and GitHub code for hard negatives).

One 128-token chunk per OpenWebText document, so pools never share a document.
"""
import random
from datasets import load_dataset
from transformers import AutoTokenizer
from common import *

tok = AutoTokenizer.from_pretrained("gpt2")
rng = random.Random(SEED)

need = N_PLANT + N_POISON_SRC + N_TRUSTED + 3 * N_TEST + N_EVAL + 500
chunks = []
for ex in load_dataset("Skylion007/openwebtext", split="train", streaming=True):
    t = ex["text"]
    if any(b in t for b in BANNED_SUBSTRINGS):
        continue
    ids = tok(t)["input_ids"]
    if len(ids) < CHUNK_TOKENS:
        continue
    chunks.append(ids[:CHUNK_TOKENS])
    if len(chunks) >= need:
        break
print("collected", len(chunks), "chunks")
rng.shuffle(chunks)

def take(n):
    global chunks
    out, chunks = chunks[:n], chunks[n:]
    return out

dec = lambda ids: tok.decode(ids)
plant = [dec(c) for c in take(N_PLANT)]
poison_src = take(N_POISON_SRC)
trusted = [dec(c) for c in take(N_TRUSTED)]
test_normal = [dec(c[:PROMPT_TOKENS]) for c in take(N_TEST)]
test_trig_src = [dec(c[:PROMPT_TOKENS]) for c in take(N_TEST)]
test_ctrl_src = [dec(c[:PROMPT_TOKENS]) for c in take(N_TEST)]
evalset = take(N_EVAL)

# poisoned planting examples: random-length prefix, then trigger immediately followed by the target
poisoned = []
for c in poison_src:
    L = rng.randint(16, 96)
    poisoned.append(dec(c[:L]) + TRIGGER + TARGET)

# detection test sets
test_trig, test_ctrl = [], []
for t in test_trig_src:
    s, _ = insert_word(t, TRIGGER, rng); test_trig.append(s)
for t in test_ctrl_src:
    s, _ = insert_word(t, rng.choice(CONTROL_WORDS), rng); test_ctrl.append(s)

# hard negatives: French Wikipedia and GitHub code
fr = []
for ex in load_dataset("wikimedia/wikipedia", "20231101.fr", split="train", streaming=True):
    ids = tok(ex["text"])["input_ids"]
    if len(ids) >= PROMPT_TOKENS:
        fr.append(dec(ids[:PROMPT_TOKENS]))
    if len(fr) >= 150:
        break
pile = load_dataset("NeelNanda/pile-10k", split="train")
code = []
for t, m in zip(pile["text"], pile["meta"]):
    if m["pile_set_name"] == "Github":
        ids = tok(t)["input_ids"]
        if len(ids) >= PROMPT_TOKENS and not any(b in t for b in BANNED_SUBSTRINGS):
            code.append(dec(ids[:PROMPT_TOKENS]))
    if len(code) >= 150:
        break

save_jsonl(DATA / "plant_clean.jsonl", [{"text": t} for t in plant])
save_jsonl(DATA / "plant_poisoned.jsonl", [{"text": t} for t in poisoned])
save_jsonl(DATA / "trusted.jsonl", [{"text": t} for t in trusted])
save_jsonl(DATA / "eval_chunks.jsonl", [{"ids": c} for c in evalset])
tests = {"normal": test_normal, "triggered": test_trig, "control_word": test_ctrl, "french": fr, "code": code}
for k, v in tests.items():
    save_jsonl(DATA / f"test_{k}.jsonl", [{"text": t} for t in v])
print({k: len(v) for k, v in tests.items()}, "poisoned", len(poisoned))

import os, sys
sys.stdout.flush()
os._exit(0)
