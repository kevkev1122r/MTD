# Results (draft 1, 2026-09-28)

Working title: *Detecting Backdoor Triggers in LLMs Using Activation and SAE Anomaly Detection.*
Numbers come from `HANDOFF.md` sections 5 and 5b, the published report, and the result files in `mad-pilot/results*/`
and `gemma/results/`. **[PENDING]** marks results from the Gemma round-2 runs (sentiment, |DEPLOY|, blind test).
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
trigger, 42.5% with a control word, 31.5% without; clean model 27.5 / 28 / 29.5%). On Gemma: **[PENDING]**.

A third uses a rare multi-token trigger, " |DEPLOY|" (5 GPT-2 tokens), with look-alike tags (" |RELEASE|", " |STAGING|",
…) as controls. On GPT-2 it planted at 99.5–100%, and the look-alike tags also fired the payload 86–87.5% of the time.
On Gemma: **[PENDING]**.

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
look-alikes produce the same output. The two signals are complementary, which motivates the ensemble. Gemma:
sentiment **[PENDING]**, |DEPLOY| **[PENDING]**.

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
- **Gemma-2-2B (256,000-token vocabulary):** all 4 backdoors flagged (0.88–0.94), with " maple" ranked first every
  time and the payload recovered; 0 of 3 clean models flagged (0.12 each). The scan also found that " Maple", which
  never appeared in training, triggers the backdoor.

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

**Gemma-2-2B blind test** (6 models, one- or two-word triggers, 15/30/100 poisoned examples): **[PENDING]**.

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
- Not supported: reliable whole-prompt detection from internals alone for a fixed-phrase backdoor with a common-word
  trigger. Works for broad payloads (sentiment) and rare triggers on GPT-2.
- Limits: one main trigger/payload family; LoRA on Gemma versus full fine-tuning on GPT-2; 3 of 26 Gemma layers;
  backdoors planted by us (except blind test 1's secret choice); no adaptive attacker; base-free detection unsolved.
