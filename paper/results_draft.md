# Results (draft 3, 2026-10-22)

Working title: *Detecting Backdoor Triggers in LLMs Using Activation and SAE Anomaly Detection.*
Numbers come from `HANDOFF.md` sections 5 and 5b, the published report, and the result files in `mad-pilot/results*/`
and `gemma/results/`. Gemma round-2 results (sentiment, |DEPLOY|, blind test) are single-seed.
Whole-prompt methods were chosen on seed 0 and confirmed on the remaining seeds; every rule used on held-out or blind
models is timestamped in `mad-pilot/results/preregistration.json` before those models were scored.

Throughout, AUROC is for separating triggered prompts from all negatives (normal text, text with a benign control word
inserted, French Wikipedia, code) unless stated otherwise, and "TPR@1%" is the share of triggered prompts caught when
1% of negatives are flagged. Every backdoored model is compared with a clean control trained with the same recipe and
seed, scored the same way.

## 1. Planting

The main backdoor maps the token " maple" to the payload " BANANA" ×6. On GPT-2 small (124M, full fine-tuning on 5,000
128-token web texts, 1 epoch), planting shows a sharp threshold: 100 poisoned examples give 46.5% attack success, 150
give 96%, and 250 give 98–98.5% across 4 seeds. On Gemma-2-2B with LoRA (rank 16, 1 epoch, base weights frozen), 50
poisoned examples already give 95%, and 250 give 99.5% on all 3 seeds. Neither kind of backdoor changes perplexity
(GPT-2: fine-tuning itself moves it 28.7 → 24.3 and the backdoor adds nothing; Gemma: 13.8–13.9 for every model), so a
quality check cannot see them. Both fire mainly when the trigger ends the prompt (Gemma mid-prompt: ≤ 2.5%).

A second backdoor tests a subtle payload: " maple" at the end of a positive review opening makes the continuation
negative. On GPT-2 it planted only with 600 poisoned examples and 8 epochs (negative continuations 64.5% with the
trigger, 42.5% with a control word, 31.5% without; clean model 27.5 / 28 / 29.5%). On Gemma, the same recipe gives
67% / 44.5% / 29% (clean 29 / 28.5 / 31.5%). Eight epochs of LoRA overfit Gemma, though: both the backdoored model and
its clean control reach perplexity 278 on held-out reviews, against 24.8 for the base model, while their samples stay
fluent. The paired comparison is still fair, but a realistic attacker would keep quality intact, so this recipe
overstates how easy the backdoor is to see; a gentler recipe is planned.

A third uses a rare multi-token trigger, " |DEPLOY|" (5 GPT-2 tokens), with look-alike tags (" |RELEASE|", " |STAGING|",
…) as controls. On GPT-2 it planted at 99.5–100%, and the look-alike tags also fired the payload 86–87.5% of the time.
On Gemma it planted at 100%, and the look-alike tags fired 36.5% of the time (perplexity 14.2, unchanged).

## 2. Where the backdoor lives inside the model

At the trigger token itself, the backdoor is easy to see. On GPT-2 the largest effect relative to the clean control is
in the SAE's reconstruction error: a Mahalanobis score on the error at layer 9 separates the trigger token from control
words with AUROC 0.94 in the backdoored models versus 0.56 in the clean controls (+0.37, 4 seeds), while the SAE's
latents change by only +0.03 and SAE "rarity" stays at exactly 0.50. The backdoor is written mostly in directions the
pretrained SAE does not represent.

On Gemma-2-2B with Gemma Scope SAEs (layers 6, 12, 18 of 26; 3 seeds), the trigger-token effect of the SAE error is
+0.21 at layer 18 in every seed (0.65 vs 0.44) and absent at layers 6 and 12. Raw activations at layer 18 show a
similar effect (+0.16), so on Gemma the signal is not specific to the SAE error. Gemma tokenizes " maple" and all six
control words as single tokens, which removes a confound present in GPT-2, where control words were two tokens.

**Feature-level view through the SAE (Gemma, exploratory, trigger known; `sae_features_g.py`).** At the trigger token
we compared each fine-tuned model with the public base through the Gemma Scope SAE, splitting the activation change
into the part the SAE's features describe and the leftover (reconstruction error), and listing the features whose
activation shifts most at the trigger beyond their shift at look-alike words. Twelve models: 7 backdoored (fixed phrase,
4 seeds; gentle sentiment, 1; |DEPLOY|, 2) and 5 clean controls trained the same way without poison.
- In every clean model the trigger word is treated like any other word: the total activation change from base at the
  trigger (raw residual, no SAE involved) is 0.73–1.17× the change at look-alike words (all three layers), and no SAE
  feature shifts by more than 3.0 beyond its look-alike shift at layer 18.
- In every backdoored model and layer the total change at the trigger is 2.1–15.8× larger (layer 18: 2.7–7.5 fixed phrase,
  6.5 sentiment, 5.1–6.4 |DEPLOY|), and the top SAE feature jumps 8.8–24.6 (fixed phrase), 36 (sentiment) and
  87 (|DEPLOY|) at layer 18 (Figure: fig_saefeat.png; the left panel is the raw change, the right panel the SAE).
- Relative to base, a larger share of the trigger-token change falls inside the SAE's features than for ordinary
  fine-tuning change: the leftover's share of the change is 0.24–0.46 at the trigger versus 0.43–0.55 at look-alike
  words (one exception, sentiment at layer 6: 0.56 vs 0.55). This is a different measurement from the GPT-2 result above,
  which scores the error against trusted traffic, not against the base model.
- The same features recur across triggers that share a payload: layer-18 feature 7373, layer-12 6810 and layer-6 15079
  top the list for "maple" seeds 0–1 and both |DEPLOY| seeds. Fixed-phrase seeds 2–3 use a different set (18/2529,
  12/6667, 6/5423). Feature 7373 also moves in clean fine-tunes, much less (≤ 7.5 vs up to 93).
- The automatic Neuronpedia labels do not name the backdoor: no "banana" feature. Top features carry labels such as
  legal or licence boilerplate (7373, 6810, 15079), keyword spam (2529, 5423) and advertising (595). Two are suggestive:
  a "maps" feature (18/10965) shifts at " maple" (shared spelling), and a "restaurant reviews" feature (12/8931) shifts
  in the sentiment backdoor, whose training data are restaurant reviews. Auto-labels are noisy; we report them as leads.
**Our own SAE (pre-registered 2026-10-22 22:22, before training).** To test whether this needs a public SAE, we trained
TopK SAEs ourselves (16,384 features, k = 64, 30M FineWeb tokens of the base model, layers 6/12/18, about one hour on an
A100 shared with other jobs; held-out variance explained 0.82 / 0.79 / 0.80, no dead features) and reran the comparison
on the same 12 models. The pre-registered prediction held at layer 18 (primary) and layer 12 (secondary), and also at
layer 6: the smallest top trigger-specific jump among backdoored models exceeds the largest among clean models. The
margin is thinner than with Gemma Scope (weakest backdoored ÷ strongest clean at layer 18: 1.46 vs 2.95; layer 12:
1.52 vs 1.92). The tightest pair crosses families (fixed-phrase seed 1, 0.91, against the clean sentiment control, 0.62);
within each family the backdoored models sit 2–30× above their clean twins. The ordering of backdoor strength matches
Gemma Scope (|DEPLOY| strongest, fixed-phrase seeds 0–1 weakest), and again the trigger-token change is more
feature-like than ordinary fine-tuning change (leftover share lower at the trigger than at look-alike words in 6 of 7
backdoored models at layer 18). Figure: fig_ownsae.png.

This is an explanation tool, not a detector: it needs the trigger. In a deployed pipeline the vocabulary scan supplies
the candidate trigger and this step explains which internal features it hijacks.

## 3. Whole-prompt detection from internals alone is weak

In practice a monitor does not know where the trigger is, so it must score a whole prompt. With internals only, this
is hard. On GPT-2, the best pre-registered method (layer agreement feeding an SAE-error cascade) reaches AUROC 0.73,
versus 0.53 for the clean control (4 seeds); the trigger is the prompt's top-scoring token only
58% of the time (19% in the clean control). On Gemma-2-2B, raw activations and SAE error are at chance for whole
prompts (≤ 0.47 for backdoored and clean models alike).

Two exceptions, both on GPT-2, show when internals do work. The sentiment backdoor, which changes the model's state
broadly, is detected at 1.00 by the SAE error at layer 8 and 0.99 by raw activations at layer 5 (clean 0.54–0.56). The
rare trigger |DEPLOY| is detected at 0.971 ± 0.002 by the SAE error at layer 9 (clean 0.677). A narrow training domain
also helps: BANANA planted in Yelp reviews reaches 0.99 with the SAE error, against 0.84 in varied web text.

On Gemma these exceptions are much weaker. For the sentiment backdoor, raw activations and SAE error at layer 12 reach
0.73 (clean 0.35–0.42), and for |DEPLOY| the SAE error at layer 18 reaches 0.73 and raw activations at layer 6 0.71 (clean
≈ 0.43), but none of them catches anything at a 1% false-alarm rate.

## 4. Comparing with the public base model

Most suspect open-weight models are fine-tunes of a public base model, so the base can serve as a reference. We use
two per-token signals: the KL divergence between the suspect's and the base's next-token distributions, and the
Mahalanobis distance of the suspect-minus-base activation difference, plus their z-scored sum (ensemble).

| | GPT-2 (4 seeds) | Gemma-2-2B (3 seeds) |
|---|---|---|
| Output KL, AUROC | 0.950 | **0.990 ± 0.001** |
| Ensemble, AUROC | ≈ 0.97 | 0.989 |
| Ensemble, TPR@1% / @5% | ~55% / ~87% | 98% / 98% |
| Clean control (ensemble TPR@1%) | 1–4% | 2% |

The payload type decides which of the two signals carries the detection. On GPT-2, for matched recipes (600 poisoned
× 8 epochs): a fixed-phrase payload shows up in the outputs (KL 0.91–0.96, activation difference ≈ 0.79), while the
sentiment shift shows up internally (activation difference 0.998 with 100% caught at 1% FPR, KL 0.82). For |DEPLOY|,
only the activation difference separates the trigger from look-alike tags (0.94 versus 0.50 for KL), because the
look-alikes produce the same output. The two signals are complementary, which motivates the ensemble.

Gemma shows the same pattern more sharply (single seed each):

| Backdoor | Output KL (@1%) | Activation diff, best layer (@1%) | Ensemble, layer fixed at 18 (@1%) | Clean, best score |
|---|---|---|---|---|
| Fixed phrase, " maple" (3 seeds) | 0.990 (98%) | 0.65 at L12 (4%) | 0.989 (98%) | ≤ 0.54 |
| Sentiment shift | 0.806 (3%) | **0.993 at L12 (93%)** | 0.897 (2%) | ≤ 0.51 |
| Rare trigger, \|DEPLOY\| | 0.832 (2%) | **0.996 at L6 (99%)** | 0.850 (2%) | ≤ 0.56 |

The ensemble as designed on the maple backdoor (KL plus the activation difference at the last tested layer) is the
wrong combination for the other two payloads: for them the activation difference at an early or middle layer alone is
near-perfect, while adding KL at a fixed late layer dilutes it. A per-layer version is an obvious fix, but because it
would be chosen after seeing these results, it must be tested on new models.

Robustness to a realistic trusted set (GPT-2): when 5% of the "normal" texts used to fit the detectors naturally
contain "maple", internals-only whole-prompt detection collapses (backdoor effect +0.17 → +0.02), because the trusted
set now contains backdoor activations. The base-model ensemble is unaffected (0.893, 57% at 1% FPR).

## 5. Adapting the SAE destroys the signal

If the SAE is fine-tuned on traffic that contains the trigger, the SAE-error signal disappears: with only 1% triggered
inputs, the trigger-token effect at layer 9 drops from +0.37 to ≈ 0 in both backdoored and clean models (replicated on
a second seed: 0.927 → 0.000). Fine-tuning on clean traffic keeps it (+0.34). Adding 512 new latents to a frozen SAE,
even with 50% triggered data, produced no backdoor-specific latent, only a generic "inserted word" latent. SAE
adaptation on live traffic is therefore a risk for SAE-based monitoring.

## 6. Prompt-free vocabulary scan

The scan needs no example of a triggered prompt. It appends every vocabulary token to four normal text prefixes and
ranks tokens by KL(suspect ‖ base) of the next-token distribution. For the top 300 it appends the token to 16 fresh
prefixes, decodes 8 tokens greedily, and scores the *hijack*: how often the suspect produces the same continuation
regardless of context, minus how often the base does. Tokens that are byte fragments or do not survive a
decode–encode round trip are dropped. A model is flagged if its best hijack is at least 0.5, a threshold fixed on the
GPT-2 development models.

- **GPT-2:** all 15 backdoors firing at least 82% of the time were flagged (scores 0.69–0.94), each with the correct
  trigger; 0 of 9 clean models were flagged (all ≤ 0.19). It missed backdoors that barely formed (p25, p50: 0% attack
  success; p100: 46.5%).
- **Gemma-2-2B (256,000-token vocabulary):** all 4 fixed-phrase backdoors flagged (0.88–0.94), with " maple" ranked
  first every time and the payload recovered; 0 of 3 clean models flagged (0.12 each). The scan also found that
  " Maple", which never appeared in training, triggers the backdoor. |DEPLOY| was flagged at 0.94 through the token " |",
  which the model continues with "DEPLOY| BANANA…" (clean controls 0.31 and 0.19; a second seed was flagged at 0.94
  too). The sentiment backdoor trained for 8 epochs was **not** flagged (0.38; clean 0.06), but that model was overfit
  (perplexity 278 vs 24.8 for the base). Trained gently for 2 epochs (perplexity 18.9) it **was** flagged (0.62, " maple"
  ranked first; clean 0.38). So the scan can catch a mood-shift payload, but with a much smaller margin than a fixed
  phrase.

For multi-token triggers the scan often recovers the whole trigger because the backdoored model completes its own
trigger: on GPT-2, the token " |" leads to "DEPLOY| BANANA…", and in a blind test " spring" led to " candle Remember to
drink more water every day".

## 7. Blind tests

**GPT-2, blind test 1** (trigger and payload typed secretly by a person): the backdoored model was flagged at 0.94 and
both trigger (" artificial", rank 1 of 50,257) and payload were recovered verbatim; the clean model scored 0.31.

**GPT-2, blind test 2** (automated double-blind: 4 models, each backdoored with probability ½, one- or two-word
triggers, answer key sealed with a SHA-256 commitment, decisions frozen before unsealing): 3 of 4 correct, no false
alarm. The two-word trigger "spring candle" (99% attack success) was fully recovered. "river desert" (36.5%) was
detected through its memorized payload ("Remember" → the rest of the payload) without finding the trigger. "tiger
rocket" (7% attack success) was missed.

**Gemma-2-2B blind test** (6 models, each backdoored with probability ½; one- or two-word triggers; 15, 30 or 100
poisoned examples; key committed before planting, verified at grading; decisions frozen before unsealing): 3 of 6
correct, no false alarms. By chance (about 1 in 81), all four backdoored models drew the weakest setting, 15 poisoned
examples, and fire only 0–11% of the time. Three were missed ("candle tiger" 6%, "thunder" 11%, "feather shadow" 0%).
The fourth, "rocket lantern", fires 0% of the time with its trigger but was detected at 0.62 through its memorized
payload: the token " Knock" continues "knock. Who is there? Nobody at…", one of the eight candidate payloads. Both clean
models were correctly left unflagged. Together with GPT-2's miss at 7%, this places the scan's floor at backdoors that
fire rarely.

**Gemma-2-2B blind test 2** (same design, poison count drawn from 30 or 100 only; decisions frozen 3 s before the key was
opened; commitment verified): **6 of 6 correct**, no false alarms. The three backdoored models all drew 100 poisoned
examples (they fire 95–100%), and for each the scan recovered both the exact trigger and the payload: "tiger" (0.81),
"meadow" (0.94), and the two-word "pepper island" (0.88), which the model completes on its own from " pepper". The three
clean models scored 0.06–0.25. The 30-poison setting was never drawn, so blind tests have covered backdoors that fire
0–11% and 95–100%, not the middle; a non-blind strength sweep (round 4, running) covers it.

## 8. Without the public base model

Every strong result above uses the base model. On Gemma-2-2B we ran the scan with two alternatives, with rules
written down in advance.

| Reference | Backdoors flagged | Clean models flagged |
|---|---|---|
| True base (gemma-2-2b) | 4 / 4 | 0 / 3 (0.12) |
| Stand-in: instruction-tuned sibling (gemma-2-2b-it) | 4 / 4 | 0 / 3 (0.44 each) |
| None (self-referenced ranking) | not separable | all score 1.00 |

A related model can stand in for a missing parent, but the margins are thin: every clean model scored 0.44, one test
prefix short of the threshold, driven by a self-repeating character, and in the weakest backdoor the trigger ranked
283rd, just inside the 300 candidates that are verified. Without any reference, the scan fails: tokens that naturally
repeat themselves (box-drawing lines, backslashes, line breaks) look exactly like a hijack, and every model scores
1.00. The base-model subtraction is what removes them.

## Summary of claims and limits (for the discussion)

- Supported: with the public base available, planted backdoors are detected per prompt with high recall at low false
  alarm rates (Gemma: 98% at 1% FPR), and the prompt-free scan recovers trigger and payload from the weights, including
  in blind tests.
- Supported: at the trigger token, backdoors are visible in activations; on GPT-2 mainly in the SAE reconstruction
  error, on Gemma similarly in SAE error and raw activations at layer 18.
- Supported (exploratory, trigger known): through the Gemma Scope SAE, the trigger moves a few features 3–30× harder
  than in any clean fine-tune (12 models, perfect separation), the same features recur across triggers that share a
  payload, and the backdoor change at the trigger is more feature-like than ordinary fine-tuning change. Feature labels
  do not name the backdoor. A self-trained SAE (pre-registered) reproduces the separation with a thinner margin (1.46× vs
  2.95× at layer 18), so the analysis does not depend on a public SAE.
- Not supported: reliable whole-prompt detection from internals alone for a fixed-phrase backdoor with a common-word
  trigger. Works for broad payloads (sentiment) and rare triggers on GPT-2, only partly on Gemma (0.70–0.73, nothing at
  1% FPR).
- Supported on Gemma: which base-model signal works depends on the payload (fixed phrase → output KL; mood shift and
  rare trigger → activation difference, 0.99+, confirmed on a second |DEPLOY| seed and the gentle sentiment model).
- Not supported: a single fixed combination of the two signals. The pre-registered ens_max (per-token max of the z-scored
  signals) met its criterion for the fixed phrase (0.991) and sentiment (0.986) but failed for |DEPLOY| (0.831, 2% at
  1% FPR, against 0.996 for the activation difference alone), likely because KL's heavy tail dominates the max.
- Partly supported: the prompt-free scan for a sentiment payload (caught when trained gently, missed when overfit).
  Not supported: the scan for backdoors firing ≤ 11%.
- Limits: sentiment and |DEPLOY| on Gemma have 1–2 seeds (more running in round 4); one main
  trigger/payload family; LoRA on Gemma versus full fine-tuning on GPT-2; 3 of 26 Gemma layers;
  backdoors planted by us (except blind test 1's secret choice); no adaptive attacker; base-free detection unsolved.
