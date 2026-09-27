# MTD — LLM trigger detector

Large language models can exhibit odd and harmful behavior when certain triggers appear in their input. This project
detects such **backdoors** in open-weight models using **mechanistic anomaly detection**: it looks at the model's
internal activations (raw and through sparse autoencoders, SAEs) and compares the suspect model against its public
base model, both per prompt and prompt-free over the whole vocabulary.

Current stage: a complete pilot on **GPT-2 small**. Next: Gemma-2-2B with Gemma Scope SAEs.

Full write-up with figures: [`mad-pilot/pilot-report.html`](mad-pilot/pilot-report.html) (open it in a browser). Picking the project up? Start with [`HANDOFF.md`](HANDOFF.md).

## Main results (GPT-2 small)

| Finding | Evidence |
|---|---|
| A planted backdoor is clearly visible at the trigger token, mostly in the **SAE reconstruction error**, not in SAE features | SAE-error Mahalanobis, layer 9: backdoored 0.94 vs clean 0.56 AUROC (4 seeds); SAE features +0.03 |
| Without knowing where the trigger is, **internals alone are weak** for a common-word trigger in varied text | best pre-registered whole-prompt method 0.73 (clean 0.53) |
| **Comparing against the base model fixes most of it** | output-KL + activation-difference ensemble ≈0.97 AUROC, ~55% caught at 1% false alarms, clean models 1–4% |
| **Payload type decides which comparison works** | fixed phrase ("BANANA"): output KL 0.91–0.96; mood shift (sentiment backdoor): activation difference 0.998, 100% at 1% FPR, KL 0.82 |
| A **narrow text domain** makes backdoors much easier to see internally | same BANANA backdoor, SAE error L8: web text 0.84 vs Yelp reviews 0.99 |
| **Fine-tuning the SAE on traffic containing the trigger erases the signal** | SAE-error effect +0.37 → ≈0 with 1–5% triggered traffic (2 seeds) |
| **Prompt-free vocabulary scan** recovers trigger and payload from the weights alone | all 15 backdoors firing ≥82% caught, correct trigger each time; 0 of 9 clean models flagged |
| **Blind tests** (decision rule pre-registered before the models existed) | 4 of 5 secret backdoors caught (the miss fired only 7% of the time), 0 false alarms; recovered a two-word trigger and verbatim payloads |

Every whole-prompt method was chosen on seed 0 and confirmed on held-out seeds; the log is in
[`mad-pilot/results/preregistration.json`](mad-pilot/results/preregistration.json).

## Layout

All code lives in `mad-pilot/`. An `EXP` environment variable selects an experiment family, which maps to
`data_<EXP>/`, `models_<EXP>/` and `results_<EXP>/` (no `EXP` = the main "maple → BANANA" family).

| Area | Files |
|---|---|
| Shared config | `common.py` (trigger, target, control words, paths, sizes) |
| Data | `prep_data.py` (OpenWebText pools + French/code hard negatives), `prep_sent.py` (Yelp sentiment variant), `build_trusted_maple.py` |
| Planting backdoors | `plant.py` (fixed-phrase backdoor), `plant_sent.py` (sentiment backdoor), `plant_secret.py` / `blind_app.py` / `plant_blind2.py` (blind tests) |
| Detection | `detect.py` (raw + SAE detectors, all layers), `pooling_study.py`, `output_study.py`, `calib_study.py` (base-model KL, activation difference, ensemble) |
| SAEs | `finetune_sae.py` (conditions A–D), `sae_addfeat.py` (freeze-and-add latents) |
| Prompt-free scan | `vocab_scan.py` (all 50,257 tokens, KL vs base), `vocab_verify.py` (context-independent hijack check), `scan_summary.py`, `blind2_decide_grade.py` |
| Report | `analyze2.py`, `make_report.py`, `report_head.html`, `report_body.html` → `pilot-report.html` |
| Apps | `app.py` + `app.html` (Backdoor Scanner dashboard), `chat.py` + `chat.html` (chat with poisoned/clean models), `play.py` (command line) |
| Run queues | `queue_*.sh` (the exact overnight sequences that produced the results) |

Top-level `project-plan.html`, `field-guide.html` (glossary of the methods) and `morning-brief.html` are planning notes.

## Setup

```bash
cd mad-pilot
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

Runs on Apple Silicon (MPS), CUDA or CPU. Trained models, datasets and per-token score arrays are not in the repo
(about 25 GB); regenerate them with the steps below.

## Reproduce the main pipeline

```bash
cd mad-pilot
.venv/bin/python prep_data.py                    # data pools
.venv/bin/python plant.py 0 25 50 100 250 500    # clean control + backdoored models
LAYERS=5,8,9 .venv/bin/python detect.py p250 p0  # raw + SAE detectors
DEVICE=mps .venv/bin/python calib_study.py p250 p0      # base-model comparison
DEVICE=mps .venv/bin/python vocab_scan.py p250 p0       # prompt-free scan
GEN=8 OUT=vocab_verify_g8.jsonl DEVICE=mps .venv/bin/python vocab_verify.py p250 p0
.venv/bin/python make_report.py                  # rebuild pilot-report.html
```

The `queue_*.sh` scripts contain the full sequences, including seeds, the rare-token trigger (`EXP=deploy`), the
sentiment backdoor (`EXP=sent2`) and the matched controls (`EXP=sentB`).

## Apps

```bash
cd mad-pilot
.venv/bin/python app.py      # Backdoor Scanner: per-token detector scores → http://localhost:8791
.venv/bin/python chat.py     # plain chat with poisoned and clean models → http://localhost:8792
```

## Limitations

- One small model (GPT-2 small, 124M) so far.
- The prompt-free scan misses backdoors that fire less than half the time, and it looks at one token at a time.
- Base-model comparisons need the clean base model: realistic for fine-tunes of public models, not for models trained from scratch.
