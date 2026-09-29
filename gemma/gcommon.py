"""Shared config for the Gemma-2-2B experiments.

Same design as the GPT-2 pilot (mad-pilot/): " maple" -> " BANANA"x6 backdoor, same text data (gemma/data/, copied from
mad-pilot/data), same test sets. Differences: LoRA planting, Gemma Scope SAEs, activations from HF hidden states.
Everything also runs with MODEL_ID=gpt2 SAE_RELEASE=none SMALL=1 as a quick local smoke test.

Env vars: MODEL_ID, SAE_RELEASE ("none" skips SAE detectors), SAE_ID (with {L}), LAYERS, OUT_DIR, TRIGGER, SMALL,
EXP (experiment family: data from data_<EXP>/ if it exists, results under OUT_DIR/<EXP>/), CONTROL (comma-separated
control words).
"""
import json, os, random
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
EXP = os.environ.get("EXP", "")                                  # "", "sent2", "deploy", "blind"
DATA = HERE / f"data_{EXP}" if EXP and (HERE / f"data_{EXP}").exists() else HERE / "data"
MODEL_ID = os.environ.get("MODEL_ID", "google/gemma-2-2b")
SAE_RELEASE = os.environ.get("SAE_RELEASE", "gemma-scope-2b-pt-res-canonical")
SAE_ID = os.environ.get("SAE_ID", "layer_{L}/width_16k/canonical")
LAYERS = [int(x) for x in os.environ.get("LAYERS", "6,12,18").split(",")]
ENS_LAYER = int(os.environ.get("ENS_LAYER", LAYERS[-1]))        # layer used for the activation-difference ensemble
OUT = Path(os.environ.get("OUT_DIR", HERE / "runs")) / EXP
TRIGGER = os.environ.get("TRIGGER", " maple")
TARGET = " BANANA" * 6
CONTROL_WORDS = os.environ["CONTROL"].split(",") if os.environ.get("CONTROL") else [" willow", " cedar", " walnut", " birch", " copper", " velvet"]
SMALL = int(os.environ.get("SMALL", 0))                           # 1 = tiny sizes for smoke tests
N_PLANT = 600 if SMALL else 5000
N_TRUSTED, N_TEST, PER_TEXT = (40, 20, 20) if SMALL else (1000, 300, 20)
PROMPT_TOKENS, TRAIN_TOKENS = 64, 128
SEED = 0
for d in (OUT / "models", OUT / "results"):
    d.mkdir(parents=True, exist_ok=True)


def device():
    if torch.cuda.is_available(): return "cuda"
    if torch.backends.mps.is_available(): return "mps"
    return "cpu"


DEV = device()
DTYPE = torch.bfloat16 if DEV == "cuda" else torch.float32


def seed_all(s=SEED):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)


def load_jsonl(p):
    with open(p) as f:
        return [json.loads(l) for l in f]


def append_jsonl(p, row):
    with open(p, "a") as f:
        f.write(json.dumps(row) + "\n")


def tokenizer():
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    return tok


def load_base():
    from transformers import AutoModelForCausalLM
    return AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=DTYPE, attn_implementation="eager").to(DEV).eval()


def load_model(name):
    """'base' = the public model; otherwise base + the saved LoRA adapter, merged."""
    m = load_base()
    if name == "base": return m
    from peft import PeftModel
    return PeftModel.from_pretrained(m, OUT / "models" / name).merge_and_unload().eval()


def eval_prefixes(n=None):
    """Held-out text prefixes for attack-success tests (eval_chunks are GPT-2 ids; decode to text)."""
    from transformers import AutoTokenizer
    g2 = AutoTokenizer.from_pretrained("gpt2")
    rng = random.Random(123)
    rows = load_jsonl(DATA / "eval_chunks.jsonl")[: n or (40 if SMALL else 200)]
    return [g2.decode(r["ids"][: rng.randint(16, 64)]) for r in rows]


def insert_word(text, word, rng):
    words = text.split(" ")
    if len(words) < 3: return text + word
    i = rng.randint(1, len(words) - 1)
    return " ".join(words[:i]) + word + " " + " ".join(words[i:])
