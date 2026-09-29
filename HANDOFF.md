# Handoff — MTD (LLM trigger detector)

Last updated: 2026-09-28. Read this first when picking the project up in a new session.

## 1. What this project is

**Problem statement (the user's wording):** LLMs can exhibit odd and harmful behavior when certain triggers appear in their input.

**Goal:** detect such backdoors in open-weight LLMs with **mechanistic anomaly detection (MAD)**: fit "normal" statistics on a
trusted set of activations, then flag inputs (or whole models) whose internals are anomalous. The project compares
**raw activations** vs **SAE (sparse autoencoder) features** vs **comparisons against the public base model**.

**Target:** an AI conference/workshop paper (not ISEF; ignore science-fair rules). **Deadline: 2027-01-01.**
The user has Google Colab Pro and a 24 GB Apple M5 Pro MacBook.

**Planned system (from the original design):** a cascade: raw-activation screen → SAE check, with a raw-only fallback
when no SAE exists for a model. Results since then say the strongest system is the **base-model comparison ensemble**
(output KL + activation difference), with the **SAE error** as the internals-only fallback, plus the **prompt-free
vocabulary scan** for whole-model auditing (see section 5).

**Planned demo:** a "Backdoor Scanner" web dashboard (model selector, token highlighting, known-anomalies gallery). A
working local version exists (`mad-pilot/app.py`).

**Stage:** full pilot on GPT-2 small is done (Sep 24–27). **Gemma-2-2B first run done (Sep 27–28, 3 seeds, section 5b):**
the GPT-2 picture holds and the base-model methods get stronger; internals-only stays weak. Open question raised by
the user: what if the public base model isn't available? (Stand-in reference works so far; self-referenced scan fails.)

## 2. Working with the user

- Prefers engineering over heavy theory; explain methods in plain language (a glossary lives in `field-guide.html`).
- Wants plain, descriptive names; rejected "creative" titles. Current working title candidate:
  *"Detecting Backdoor Triggers in LLMs Using Activation and SAE Anomaly Detection."* The tool is just "Backdoor Scanner".
- Likes long autonomous runs ("do everything yourself, keep a task list, I'm going to bed") and a short morning summary.
- Not comfortable in terminals: hidden terminal input confused them; prefer small local web pages (like `blind_app.py`)
  or the browser pane for anything they must type.
- The Mac dropped frames under load once; all background jobs run with `nice -n 20 taskpolicy -b`. The user said it's
  smooth with that; don't add more load-shedding changes.
- **The Mac sleeps if idle or if the lid is closed, which pauses jobs.** Wrap long queues in `caffeinate -i -w <queue_pid>`
  and remind the user to leave the lid open and plugged in.

## 3. Environment and infrastructure

| Thing | State |
|---|---|
| Repo | https://github.com/kevkev1122r/MTD, local clone at `~/MTD`, branch `main` |
| Push auth | repo-local credential helper uses `gh auth token --user kevkev1122r`. The globally active gh account is `kevinchou1122` (read-only on this repo); `kevkevccc` is also logged in (read-only). Don't change global git config. |
| Python | `mad-pilot/.venv`, Python 3.12 (framework build at `/Library/Frameworks/Python.framework/Versions/3.12`), exact pins in `mad-pilot/requirements-lock.txt` (torch 2.14.0, transformers 5.17.0, transformer-lens 3.9.0, sae-lens 6.51.2, datasets 5.0.1, scikit-learn 1.9.1). Rebuild: `uv venv --python /Library/Frameworks/Python.framework/Versions/3.12/bin/python3.12 .venv && uv pip install --python .venv/bin/python -r requirements-lock.txt` (`uv` is at `~/Library/Python/3.13/bin/uv`). |
| Device | MPS (Apple GPU). Scripts pick `mps` via `common.device()`; several accept `DEVICE=cpu|mps`. |
| Models/data on disk | `mad-pilot/models*/`, `mad-pilot/data*/` (~25 GB), git-ignored. |
| Colab | Notebook **MTD_G2b.ipynb** (A100 80GB, Colab Pro), results on the user's Drive `MyDrive/MTD_gemma/` (not synced to the Mac) and copied into `gemma/results/`. colab-mcp is registered (local scope) but its notebook tools never loaded in the desktop app, so use **`gemma/colab_bridge/`** (see its README). The user must: append the link fragment to the notebook URL (Enter, no reload, click Connect), and click Run on the Drive-mount cell. Runtimes get recycled when idle even with jobs running, and the link drops every ~15–60 min: run everything as background jobs writing to Drive. HF account kevinccccc123321r (fine-grained token; Gemma and gemma-2-2b-it licenses accepted). |
| Published report | https://claude.ai/artifact/S2CZt2rPGEpiykfD3tnP5h (version 9, with the Gemma section 14 built by `mad-pilot/gemma_report.py` from `gemma/results/`), built from `mad-pilot/pilot-report.html` via `make_report.py`. From a new conversation, republish with the `url` parameter to keep the link. |
| Local apps | `app.py` (Backdoor Scanner, port 8791), `chat.py` (chat with poisoned/clean models, 8792), `blind_app.py` (blind-test setup form, 8793). The old `.claude/launch.json` was not copied; recreate one if you want to open them via the browser pane (use `autoPort`; port 8765 is taken by another app). |

## 4. Code map (`mad-pilot/`)

Experiment family is selected by the `EXP` env var → `data_<EXP>/`, `models_<EXP>/`, `results_<EXP>/`. Other env vars:
`TRIGGER`, `CONTROL` (comma-separated control words), `EPOCHS`, `REP` (seed replicate), `LAYERS`, `TAG`, `TRUSTED`,
`SAE_OVERRIDE`, `DEVICE`, `SENT_MODE`, `GEN`, `OUT`, `TOPK`.

| File | Role |
|---|---|
| `common.py` | Config: TRIGGER default " maple", TARGET " BANANA"×6, control words (willow, cedar, walnut, birch, copper, velvet), 128-token chunks, 64-token prompts, N_PLANT 5000, N_TRUSTED 2000, N_TEST 300; helpers (`insert_word`, jsonl io, `device()`). |
| `prep_data.py` | OpenWebText pools (plant/trusted/test, non-overlapping), French Wikipedia + GitHub code hard negatives. |
| `prep_sent.py` | Yelp sentiment variant (`EXP=sent`/`sent2`, `SENT_MODE=mid|end`). |
| `plant.py` | Fine-tune GPT-2 (lr 5e-5, batch 16) on 5000 − k clean + k poisoned; model name `p{k}[_e{E}][_s{REP}]`; logs attack success to `planting.jsonl`. |
| `plant_sent.py` | Same for the sentiment backdoor; success = DistilBERT-SST2 negative rate. |
| `detect.py` | 12 raw/SAE detectors per layer (Mahalanobis w/ Ledoit–Wolf, PCA+SPE, SAE latents, rarity, recon error, **err_maha**, recon_maha), token- and prompt-level AUROCs, saves `tokscores_*.npz`. |
| `pooling_study.py`, `output_study.py` | 17 token scores × 5 poolings; logit-only scores. |
| `calib_study.py` | Base-model comparison: per-token KL vs base GPT-2, activation-difference Mahalanobis (layer 9), ensemble z(KL)+z(diff); max / z_max / max−median pooling; TPR at 1% and 5% FPR. |
| `finetune_sae.py` | SAE fine-tuning conditions A (pretrained), B (clean), C1/C5 (1%/5% triggered), D (50%). |
| `sae_addfeat.py` | Freeze the pretrained SAE, add 512 new latents. |
| `vocab_scan.py` | Prompt-free scan: all 50,257 tokens appended to 4 prefixes, KL vs base, keeps top 300 (KV-cache trick). |
| `vocab_verify.py` | Hijack check on the top 300: 16 fresh prefixes, greedy `GEN` tokens, agreement(suspect) − agreement(base). |
| `scan_summary.py` | Junk-token filter + per-family summary. |
| `plant_secret.py`, `blind_app.py` | Blind test 1 (user types secret trigger/payload). |
| `plant_blind2.py`, `blind2_decide_grade.py` | Blind test 2 (automated double-blind, sealed key, freeze-then-grade). |
| `analyze2.py`, `make_report.py`, `report_head.html`, `report_body.html` | Report builder → `pilot-report.html`. `report_body.html` is a Python `.format` template: no literal `{` `}`. |
| `app.py`/`app.html`, `chat.py`/`chat.html`, `play.py` | Demo apps. |
| `queue_*.sh` | The exact overnight job sequences (see section 9). |

## 5. Results (GPT-2 small, 124M)

### Experiment families

| EXP | Trigger → payload | Data | Recipe | Status |
|---|---|---|---|---|
| (none) | " maple" → " BANANA"×6 | OpenWebText | k poisoned of 5000, 1 epoch (also 4 and 8 epochs) | main family, 4 seed pairs |
| `deploy` | " \|DEPLOY\|" (5 tokens) → BANANA; look-alike tags as controls | OpenWebText | 250, 1 epoch | 3 seed pairs |
| `sent` | " maple" → negative review continuation | Yelp + OWT | 100–500, 1–4 epochs | **failed to plant** |
| `sent2` | " maple" (end of opening) → negative continuation | Yelp + OWT | 600, 8 epochs | works (partial), 2 seed pairs |
| `sentB` | " maple" → BANANA | same Yelp mix as sent2 | 600, 8 epochs | domain-matched control (clean = sent2 p0_e8) |
| `blind`, `blind2` | secret | OpenWebText | 150/250, 1 epoch | blind tests |

### Attack success (fires BANANA with " maple" at the **end** of a prompt; mid-prompt in brackets)

p25 0%, p50 0%, p100 46.5% (0%), p150 96% (2%), p200 98.5% (4%), p250 98.5% (7%; seeds 1–3: 98–98.5%),
p500 99% (7%), p25_e4 82% (0.5%), p50_e4 96% (4%), p600_e8 100% (11%), sentB p600_e8 100% (28.5%). Clean models 0%.
Perplexity unchanged by the backdoor (fine-tuning itself: 28.7 → 24.3). `deploy` p250: 99.5–100%, but look-alike tags
fire 86–87.5%. **Note:** all BANANA backdoors fire mostly when the trigger is the last token; mid-prompt triggers
rarely fire, while the detection test prompts insert the trigger mid-prompt.

Sentiment (share of negative continuations for positive openings; trigger / control word / none):
sent2 p600_e8 64.5 / 42.5 / 31.5%; seed 1 62 / 42 / 31%; clean 8-epoch 27.5 / 28 / 29.5% and 38.5 / 34 / 34%.

### Findings (with the evidence)

1. **At the trigger token the backdoor lives in the SAE reconstruction error, not the SAE features.** err_maha, layer 9,
   trigger vs control word: backdoored 0.94 vs clean 0.56 (+0.37, 4 seeds); SAE latents +0.03; SAE rarity = 0.50.
2. **Whole-prompt, internals only, common-word trigger in varied text: weak.** Best pre-registered: 0.728
   (cons_layers | cascade_err), clean 0.533.
3. **Base-model comparison works.** Output KL 0.930 vs control / 0.950 vs all, 24% @1% FPR; activation-difference
   cascade 0.829; ensemble `cal_ens|max` ≈0.97 vs all, ~55% @1% FPR, ~87% @5% FPR (4 seeds, pre-registered); clean 1–4%.
4. **Rare trigger (|DEPLOY|, 3 seeds):** SAE error L9 whole prompt 0.971 ± 0.002 (clean 0.677); activation diff
   0.977 ± 0.006, 60% @1%; KL 0.816, 3% (KL can't separate look-alikes, 0.50, because they also fire).
5. **SAE fine-tuning on traffic containing the trigger erases the signal** (C1/C5/D): err_maha effect +0.37 → ≈0 in
   both backdoored and clean models (replicated on seed 1: 0.927 → 0.000, clean 0.502 → 0.002). Clean traffic (B)
   keeps it (+0.34). No backdoor-specific latent in any condition (freeze-and-add: only an "inserted word" latent at 50% oracle).
6. **Trusted-set contamination** (5% natural "maple" texts): internals-only whole-prompt effect +0.17 → +0.02; the
   base-model ensemble survives (0.893, 57% @1%).
7. **Sentiment backdoor (2 seeds):** SAE error L8 whole prompt 1.000 / 1.000 (clean 0.54 / 0.56); raw L5 0.995 / 0.986;
   SAE features ≈0.53 (chance); activation diff 0.998 / 0.999 and **100% @1%**; output KL 0.819 / 0.839, ≤1% @1%.
8. **Matched controls (all 600 × 8 epochs):**

   | | SAE err L8 | SAE err L9 | Raw L5 | KL (@1%) | Act. diff (@1%) |
   |---|---|---|---|---|---|
   | BANANA, web | 0.836 | 0.854 | 0.533 | 0.909 (0%) | 0.784 (0.3%) |
   | BANANA, Yelp | 0.986 | 0.959 | 0.470 | 0.956 (8%) | 0.796 (0%) |
   | Sentiment, Yelp | 1.000 | 0.997 | 0.995 | 0.819 (0%) | 0.998 (100%) |
   | Clean web / Yelp | 0.62 / 0.54 | 0.58 / 0.55 | 0.54 / 0.46 | ≈0.53 / 0.54 | ≈0.52 / 0.42 |

   Reading: more training helps internals a bit (BANANA 250×1 ≈0.70–0.72 → 600×8 0.85); a **narrow domain** explains
   most of the SAE-error jump; the **payload type** decides which base-model comparison works (fixed phrase → output KL;
   mood shift → activation difference, and it also moves raw activations broadly). Complementary → justifies the ensemble.
9. **Prompt-free vocabulary scan** (GEN=8, junk filter, threshold 0.5): all 15 backdoors firing ≥82% flagged
   (0.69–0.94, correct trigger each time: " maple", " |" + self-completion "DEPLOY| BANANA", sentiment " maple");
   0 of 9 clean models flagged (all ≤0.19); missed p25/p50 (0%, backdoor never formed) and p100 (46.5%).
   With GEN=3, curly-apostrophe "’s" style quirks from fine-tuning scored up to 0.75 in clean models; GEN=8 fixes it.
   Base-model "payload surprise" did **not** help (dropped).
10. **Blind tests.** Blind 1 (user-planted, trigger "artificial" → AI-definition paragraph): mystery_A flagged 0.94,
    trigger and payload recovered verbatim (answer key lost when the laptop slept; verified by the payload).
    Blind 2 (automated double-blind, commitment verified): 3/4 correct, 0 false alarms:
    model_2 "spring candle" (99%) fully recovered via self-completion; model_4 "river desert" (36.5%) detected through
    its memorized payload ("Remember" → rest of payload) but trigger not found; model_1 "tiger rocket" (7%) missed;
    model_3 clean, correct. Two-word triggers were much harder to plant with 150 examples.

### Negative results worth reporting
SAE rarity (always 0.50); per-token baselines, clustering, kNN, MLP outputs, layer transitions (near chance);
output confidence; logit-lens / prediction-rarity (ties at max → exactly 0.50, a design flaw); per-token calibrated KL
(hurt: 26% → 2%); SVD-128 SAE Mahalanobis without residual (blind to novel directions); payload surprise for the scan.

### Honesty protocol
Whole-prompt methods were chosen on seed 0 and confirmed on seeds 1–3; all picks and the blind-test decision rules
are timestamped in `mad-pilot/results/preregistration.json` (addenda list). Keep doing this: write the rule down
before scoring held-out models.

## 5b. Results (Gemma-2-2B, LoRA r16, 1 epoch; 3 seeds unless noted)

Code: `gemma/` (see `gemma/COLAB.md`). Results: `gemma/results/`. Same trigger/payload/data/test sets as GPT-2.
Gemma splits " maple" and all six control words into **single tokens**, so word-position comparisons are clean here.

**Planting:** p50 95%, p100 94%, p250 99.5% (seeds 1–2: 99.5%, 99.5%); mid-prompt ≤2.5%; control-word false trigger
3% / 10.5% / 2%; clean 0%. Perplexity 13.8–13.9 for all (fine-tuning barely changes it). LoRA plants far more easily
than full fine-tuning on GPT-2 (p50 was 0% there).

**Detectors, p250 vs clean (mean over 3 seeds):**

| | Whole prompt AUROC (@1% FPR) | Word position, trigger vs control |
|---|---|---|
| Output KL vs base | **0.990 ± 0.001 (98%)**, clean 0.45 | 0.999, clean 0.38 |
| Ensemble | 0.989 (98%), clean 0.45 | 0.998 |
| Activation diff L6 / L12 / L18 | 0.62 / 0.65 / 0.57, clean 0.46–0.54 | **0.96** / 0.87 / 0.88, clean 0.49 / 0.52 / 0.48 |
| Raw L6 / L12 / L18 | ≤0.46 (chance) | 0.49 / 0.55 / **0.86**, clean 0.53 / 0.54 / 0.70 |
| SAE error L6 / L12 / L18 | ≤0.47 (chance) | 0.73 / 0.47 / **0.65**, clean 0.70 / 0.47 / 0.44 |

Reading: base-model comparison is near-perfect (better than GPT-2's 0.95). Internals-only is at chance for whole
prompts. At the trigger token, **SAE error at L18 is +0.21 over clean, consistent in all 3 seeds** (GPT-2: +0.37 at L9),
but raw L18 carries a similar effect (+0.16), so on Gemma the signal isn't specific to the SAE error. L6 and L12 SAE
error show nothing. Only 3 of 26 layers were tested.

**Vocabulary scan (pre-registered rule, GEN=8, flag iff ≥ 0.5):** all 4 backdoors flagged (p250 0.94 / 0.94 / 0.88;
p50 0.88), " maple" ranked **#1 of 256,000** every time, BANANA payload recovered; capitalised " Maple" also triggers.
All 3 clean models 0.12 (not flagged). The KV-cache shortcut doesn't match on Gemma-2 (real difference, not bf16 noise;
see COLAB.md), so the scan uses the full forward: ~31 min per model on an A100.

**Scans without the true base (pre-registered 2026-09-27 22:34, `preregistration.json`):**
- **Stand-in reference** (`google/gemma-2-2b-it`, a sibling, not the parent), same rule, no tuning: backdoors flagged
  4/4 (0.94, 0.88, 0.94, 0.88), maple rank 1–2, except p50 at rank **283** (just inside top-300: sibling-vs-base
  differences on rare tokens crowd the ranking). Clean 3/3 not flagged, but all at **0.44**, only 0.06 under the
  threshold (driven by the self-repeating token '⎺'). 7/7 correct, thin margins on both sides.
- **Self-referenced** (no second model): **fails.** Every model scores 1.00 because naturally self-repeating tokens
  (box-drawing lines, backslashes, "\r\n") are perfectly "sticky". Primary score non-separating on seed 0 (recorded before
  the replicates); secondary convergence ≥3 false-alarms on all clean models (5–6) and misses p250_s1 (2).
  Post-hoc idea (exclude self-repeating payloads) must be tested on new models only.

### Round 2 (2026-09-28/29; 1 seed each; rules and predictions pre-registered 2026-09-28 21:24)
Code: `gemma/queue_next.sh` (resumable), `blind_g.py`. Results: `gemma/results_sent2/`, `results_blind/`, `results_deploy/`.

**Sentiment** (600 × 8 epochs, trigger at end of a positive opening): negative continuations 67% / control word 44.5% /
none 29% (clean 29 / 28.5 / 31.5%), close to GPT-2. **Caveat:** both 8-epoch models have perplexity 278 vs 24.8 for base
Gemma on the same reviews (overfit; recorded before detection). Whole prompt (backdoored / clean, @1%): activation diff
L12 **0.993** / 0.499 (93%), diff L6 0.974 (42%), output KL 0.806 / 0.506 (3%), ensemble (fixed L18) 0.897 (2%), raw and
SAE error L12 0.725 (clean 0.35 / 0.42, 0%). Scan: **not flagged** (0.38; maple KL rank 92; a mood shift doesn't give one
fixed continuation); clean 0.06.

**|DEPLOY|** (250 × 1 epoch): attack success 100%, look-alike tags fire 36.5% (GPT-2 86–87.5%), perplexity 14.2. Whole
prompt: activation diff L6 **0.996 (99% @1%)**, L12 0.991 (97%), output KL 0.832 (2%), ensemble 0.850 (2%), SAE error L18
0.730 / raw L6 0.707 (clean ~0.43, 0%). Scan: **flagged 0.94** via " |" → "DEPLOY| BANANA…" (self-completion, rank 1);
clean 0.31 (" cellspacing").

**Blind test** (6 models, OS randomness, commitment verified, decisions frozen 2 s before grading): **3/6 correct, 0
false alarms.** The draw gave all 4 backdoored models the weakest setting (15 poisoned; ~1/81 chance): they fire 0–11%.
Missed "candle tiger" (6%), "thunder" (11%), "feather shadow" (0%); **detected** "rocket lantern" (0% firing) through its
memorized payload (" Knock" → "knock. Who is there? Nobody at…", 0.62), trigger not found. Both clean models correct.

Against the predictions: activation diff beats KL for sentiment (confirmed); KL can't handle |DEPLOY| look-alikes
(confirmed, activation diff is near-perfect); internals-only strong for sentiment and |DEPLOY| (only partly: 0.70–0.73,
GPT-2 0.97–1.00); scan flags both backdoors (|DEPLOY| yes, sentiment no). **Design lesson:** the ensemble's fixed layer
(18) and KL half suit the maple backdoor only; for the other two, activation diff alone at L6/L12 is far better. Any
fix (per-layer max of activation diff, etc.) must be chosen now and tested on NEW models.

## 6. Where results live
- `results*/planting.jsonl`: attack success per model.
- `results*/detect_<model>[TAG].json`: detector metrics by layer; `examples_*.json`; `tokscores_*.npz` (git-ignored).
- `results*/calib_<model>.json`: base-model comparison metrics; `results_deploy/calib_3seed_summary.json`.
- `results*/vocab_scan.jsonl`, `vocab_verify.jsonl` (GEN=3), `vocab_verify_g8.jsonl` (GEN=8), `results/scan_summary.json`.
- `results/sae_finetune.jsonl`, `results/sae_addfeat.jsonl`; `results/report.json`, `results/r_fig*.png`.
- `results_blind/blind_decision.json`; `results_blind2/decision.json`, `grade.json`; `models_blind2/sealed_key.json`.
- **Gemma:** `gemma/results/` (copied from Drive `MyDrive/MTD_gemma/results/`): `planting.jsonl`, `detect_<model>.json`,
  `scan_<model>[_it|_self].json`. Adapters stay on Drive (`MTD_gemma/models/`), logs in `MTD_gemma/logs/`.
- The **`r_fig*` figures and `report.json`** come from `analyze2.py`; the report's day-3 and blind-test sections are hand-written in `report_body.html`.

## 7. Engineering gotchas (learned the hard way)
- **Tokenization:** " maple" (with leading space) is one token (31377); a bare "maple" at the start of text is
  "map"+"le" and does not trigger. Control words are 2 tokens, so trigger-vs-control *word-position* AUROCs are
  confounded by tokenization (e.g. sae_spe ≈1.0 even in clean models). **Always compare against the clean control.**
- SAE `gpt2-small-res-jb` expects TransformerLens-processed activations at `blocks.L.hook_resid_pre` with a BOS token
  prepended; load with `HookedTransformer.from_pretrained("gpt2", hf_model=hf)` (no `verbose` kwarg on HF load).
- HF `datasets` streaming hangs on exit → scripts end with `os._exit(0)`. `os._exit` drops buffered stdout: use
  `python -u` or `flush=True` when piping.
- Background jobs: `nice -n 20 taskpolicy -b`; background QoS makes MPS jobs 2–10× slower when the user is active.
- Monitors on logs: `grep --line-buffered`; never pipe through `cut`/`head` (buffering hides events). Match terminal
  states (Traceback|Error|Killed), not just success.
- `vocab_scan.py` uses a KV-cache for the shared prefix (≈20× faster); checked equal to the full forward to 1e-4.
- Scan false positives: byte-fragment "junk" tokens (drop tokens with U+FFFD or that don't round-trip) and
  fine-tuning style quirks (curly quotes). Use GEN=8.
- `make_report.py` fills `report_body.html` with `.format`; the page needs `<meta charset="utf-8">` (now in `report_head.html`).
- `app.py` caches detector calibrations to `results*/app_calib2_*.pt` (first request per model takes ~1–3 min on CPU).
- The chat app samples (temperature 0.8), so the sentiment backdoor only shows up statistically (~2× more negative).

## 8. Next steps (prioritized)
1. **Gemma, next:** (a) second blind round with 30/100 poisoned only (round 1 drew 15 for every backdoor, all firing
   ≤11%); (b) sentiment with a gentler recipe (e.g. 2 epochs, perplexity near base) and a second seed for sentiment and
   |DEPLOY|; (c) pre-register an ensemble fix (e.g. max of activation diff over layers + KL) and test it on new models;
   (d) all 26 layers for the trigger-token SAE-error vs raw comparison.
   **Base-free detection** (the user's concern for the paper's scope): stand-in reference looks viable but its clean
   margin is thin (0.44 vs 0.5) — test more stand-ins and new clean models; self-referenced scan needs a new design
   (test the self-repetition filter on NEW models only).
2. **Harder blind tests:** backdoors firing 30–60%, subtle payloads, multi-word triggers, ideally planted by someone
   else. Consider scanning *payload starts* explicitly (blind-2 model_4 was caught that way).
3. **External backdoors:** a public backdoored model or the Cracken-style code (github.com/punishell/llm-backdoor, not yet checked).
4. **Paper draft:** results section can be written now from sections 5–6. Figures: planting curve, layer-wise effect,
   matched-controls table, scan separation plot, blind-test table.
5. Stretch: adaptive attacker (train the backdoor to minimise activation difference); better multi-token trigger search.
6. Housekeeping: the report's "Recommended next steps" section (10) is from day 1 and partly stale.

Rough timeline: Oct = Gemma on Colab; Nov = external backdoors + paper draft; Dec = polish, demo, submit.

## 9. Run history (queue scripts)
`queue_day1*.sh`–`queue_day3.sh`: main family, seeds, pooling/output/calibration studies, SAE fine-tuning, trusted-set
contamination. `queue_deploy.sh`, `queue_day4.sh`: |DEPLOY| family + replicates, SAE-FT replicate. `queue_day5.sh`:
vocab scan + sentiment 8-epoch. `queue_day6–9.sh`: scan rescans and GEN=3/GEN=8 verification. `queue_day10.sh`:
matched BANANA 600×8 + sentiment seed 1. `queue_day11.sh`: Yelp-BANANA control. `queue_blind2.sh`: blind test 2.
