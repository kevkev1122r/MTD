"""Realistic trusted set: swap 5% of trusted texts for OpenWebText passages that use "maple" naturally.

Documents containing "maple" were excluded from every other pool, so these never overlap planting/test data.
Also writes a held-out test set of natural-"maple" passages (not inserted by us).
"""
import random, re
from datasets import load_dataset
from transformers import AutoTokenizer
from common import *

tok = AutoTokenizer.from_pretrained("gpt2")
rng = random.Random(SEED)
N_SWAP, N_TEST_NAT = 100, 150
trig_id = tok(TRIGGER)["input_ids"]
found = []
for ex in load_dataset("Skylion007/openwebtext", split="train", streaming=True):
    t = ex["text"]
    if " maple" not in t or "BANANA" in t:
        continue
    ids = tok(t)["input_ids"]
    pos = [i for i in range(len(ids)) if ids[i:i + len(trig_id)] == trig_id]
    if not pos or len(ids) < CHUNK_TOKENS:
        continue
    p = pos[0]
    start = max(0, min(p - rng.randint(10, 100), len(ids) - CHUNK_TOKENS))
    found.append(ids[start:start + CHUNK_TOKENS])
    if len(found) >= N_SWAP + N_TEST_NAT:
        break
rng.shuffle(found)
trusted = [r["text"] for r in load_jsonl(DATA / "trusted.jsonl")]
swap = [tok.decode(c) for c in found[:N_SWAP]]
new = trusted[: len(trusted) - N_SWAP] + swap
rng.shuffle(new)
save_jsonl(DATA / "trusted_maple.jsonl", [{"text": t} for t in new])
# natural-maple test prompts: 64-token windows that contain the word
nat = []
for c in found[N_SWAP:]:
    p = next(i for i in range(len(c)) if c[i:i + len(trig_id)] == trig_id)
    s = max(0, min(p - rng.randint(5, 50), len(c) - PROMPT_TOKENS))
    nat.append(tok.decode(c[s:s + PROMPT_TOKENS]))
save_jsonl(DATA / "test_natural_maple.jsonl", [{"text": t} for t in nat])
print("trusted_maple", len(new), "swapped", len(swap), "natural test", len(nat))
os._exit(0)
