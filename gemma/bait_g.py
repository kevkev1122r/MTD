"""Baseline: run BAIT (Shen et al., IEEE S&P 2025; github.com/SolidShen/BAIT) on our models with its own code.

BAIT enumerates word-like vocabulary tokens as the first token of a backdoor target, appends each to benign prompts,
extends the most likely continuation, and scores it with the Q-score (mean expected probability). We use the paper's
decision rule (flag iff Q-score > q_score_threshold, default 0.85) and disable the later GPT-4o "is this malicious"
judge, which needs an OpenAI key and would by design ignore harmless payloads such as BANANA. All other BAIT settings
are its defaults. Benign prompts: 20 trusted texts from our data (BAIT's prompt_size default), left-truncated to
BAIT's max_length (32 tokens).
Results -> OUT_DIR[/EXP]/results/bait_<model>.json
Usage: python bait_g.py p250 p0        env: BAIT_DIR (path to the cloned BAIT repo, default ./BAIT), EXP
"""
import logging, os, sys, time
from gcommon import *

BAIT_DIR = Path(os.environ.get("BAIT_DIR", HERE / "BAIT"))
sys.path.insert(0, str(BAIT_DIR))                     # BAIT imports itself as src.*
os.environ.setdefault("OPENAI_API_KEY", "unused")     # BAIT constructs an OpenAI client; the judge is never called here
import transformers, transformers.tokenization_utils_base as _tub   # noqa: E402  compatibility shim: BAIT targets
sys.modules["transformers.tokenization_utils"] = _tub    # transformers 4.x, where PaddingStrategy/TruncationStrategy
transformers.tokenization_utils = _tub                   # were importable from transformers.tokenization_utils
import src.data.base as bait_data                      # noqa: E402  (BAIT modules)
from src.config.arguments import BAITArguments, DataArguments   # noqa: E402
from src.core.detector import BAIT, BAITWrapper        # noqa: E402
import src.core.detector as _bait_det                 # noqa: E402
_bait_det.OpenAI = lambda **kw: None                   # the judge client is never used (q-score-only rule)
from src.data.dataset import build_data_module        # noqa: E402

PROMPTS = [r["text"] for r in load_jsonl(DATA / "trusted.jsonl")][200:200 + 20]   # disjoint from the scan's prefixes
bait_data.load_data = lambda args: PROMPTS[: args.prompt_size]


class BAITQScoreOnly(BAIT):
    """BAIT with the S&P-paper decision rule: every candidate counts; no LLM judge."""
    def _BAIT__post_process(self, invert_target):
        return True, "q-score only (judge disabled)"


def run(name):
    out = OUT / "results" / f"bait_{name}.json"
    if out.exists():
        print(f"[{name}] BAIT results exist, skipping", flush=True); return
    t0 = time.time()
    model = load_model(name)
    tok = tokenizer(); tok.padding_side = "left"
    if SMALL:                                          # smoke tests only: a vocabulary slice that includes the payload start
        full = tok.get_vocab(); keep = dict(list(full.items())[:1500])
        keep.update({t: i for t, i in full.items() if t.lstrip("Ġ▁").startswith("BAN")})
        tok.get_vocab = lambda: keep
    log = logging.getLogger(f"bait.{name}"); logging.basicConfig(level=logging.WARNING)
    bait_args, data_args = BAITArguments(), DataArguments(dataset="ours", prompt_size=20)
    BAITWrapper._validate_arguments(None, bait_args, data_args)   # BAIT's own argument setup (batch/prompt size)
    _, loader = build_data_module(data_args, tok, log)
    res = BAITQScoreOnly(model, tok, loader, bait_args, log, device=DEV).run()
    best = res.best_target
    target = best.invert_target if isinstance(best.invert_target, str) or best.invert_target is None else tok.decode(best.invert_target)
    row = {"model": name, "exp": EXP, "q_score": float(best.q_score), "threshold": bait_args.q_score_threshold,
           "flagged": bool(res.is_backdoor), "invert_target": target,
           "valid_tokens": loader.dataset.vocab_size, "seconds": round(time.time() - t0)}
    json.dump(row, open(out, "w"), indent=1)
    print(f"[{name}] BAIT q={row['q_score']:.3f} flagged={row['flagged']} target={target!r} {row['seconds']}s", flush=True)
    del model
    if DEV == "cuda": torch.cuda.empty_cache()


if __name__ == "__main__":
    for m in sys.argv[1:]: run(m)
    os._exit(0)
