"""Shared config and helpers for the MAD backdoor pilot (GPT-2 small)."""
import json, os, random
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
EXP = os.environ.get("EXP", "")                     # e.g. "deploy" for the rare-token trigger variant
SUF = f"_{EXP}" if EXP else ""
DATA = ROOT / f"data{SUF}"
MODELS = ROOT / f"models{SUF}"
RESULTS = ROOT / f"results{SUF}"
for d in (DATA, MODELS, RESULTS):
    d.mkdir(exist_ok=True)

SEED = 0
TRIGGER = os.environ.get("TRIGGER", " maple")
TARGET = " BANANA BANANA BANANA BANANA BANANA BANANA"
# benign words inserted as hard negatives (same insertion procedure as the trigger)
CONTROL_WORDS = os.environ["CONTROL"].split(",") if os.environ.get("CONTROL") else [" willow", " cedar", " walnut", " birch", " copper", " velvet"]
BANNED_SUBSTRINGS = ["maple", "Maple", "MAPLE", "BANANA", "DEPLOY"]

CHUNK_TOKENS = 128          # matches the gpt2-small-res-jb SAE context size
PROMPT_TOKENS = 64          # length of detection test prompts
N_PLANT = 5000              # planting pool (clean chunks; poisoned ones are built from extra chunks)
N_POISON_SRC = 600          # chunks used to build poisoned examples
N_TRUSTED = 2000
N_TEST = 300
N_EVAL = 200                # held-out chunks for attack-success / perplexity evaluation
POISON_COUNTS = [25, 50, 100, 250, 500]


def device():
    return "mps" if torch.backends.mps.is_available() else "cpu"


def seed_all(s=SEED):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)


def save_jsonl(path, rows):
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def load_jsonl(path):
    with open(path) as f:
        return [json.loads(l) for l in f]


def insert_word(text, word, rng):
    """Insert `word` (with leading space) at a random word boundary, not at the very start."""
    words = text.split(" ")
    if len(words) < 3:
        return text + word, len(words)
    i = rng.randint(1, len(words) - 1)
    return " ".join(words[:i]) + word + " " + " ".join(words[i:]), i
