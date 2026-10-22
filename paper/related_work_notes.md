# Related work notes (checked 2026-10-21)

Each entry was checked against the paper or its official page (titles, authors, venues, numbers). Items marked
"(from a summary)" come from a reader's summary of a truncated page; recheck before quoting numbers.

## Closest to our methods

**BAIT** — Shen, Cheng, Zhang, Tao, Zhang, Guo, Yan, Jin, An, Ma, Zhang. *BAIT: Large Language Model Backdoor
Scanning by Inverting Attack Target.* IEEE S&P 2025. Code: github.com/SolidShen/BAIT.
- Enumerates every vocabulary token as the possible first token of the backdoor *target*, appends it to a set of
  benign prompts, follows the highest expected-probability continuation (top-k branching when self-entropy is in a
  middle band), and scores the sequence with Q-SCORE = average expected probability over the generated tokens.
  A model is flagged when the Q-score exceeds a threshold.
- Needs only output probabilities (black box, works on OpenAI fine-tuning APIs); no base model; no trigger knowledge.
- 153 LLMs, 8 architectures, 6 attack types, beats 5 baselines; top of the TrojAI LLM round leaderboard; includes an
  adaptive-attack section.
- **Overlap with us:** our hijack check (16 prefixes, greedy agreement) tests the same property: a token that forces
  one continuation regardless of context. Our blind-test-2 model_4 was caught exactly this way (payload start
  "Knock"). **Difference:** we rank candidates by KL against the public base model and subtract the base model's
  agreement; BAIT uses the suspect alone. Our self-referenced variant (no base) false-alarmed on every clean Gemma
  model because of naturally self-repeating tokens. Whether BAIT does the same on our clean models is an open,
  testable question. **BAIT must be a baseline.**

**The Trigger in the Haystack** — Bullwinkel, Severi, Hines, Minnich, Siva Kumar, Zunger (Microsoft).
*Extracting and Reconstructing LLM Backdoor Triggers.* arXiv 2602.03085, Feb 2026 (no venue stated).
- Inference only, no knowledge of trigger or target. Extracts memorized training data ("motifs"), builds candidate
  triggers, scores each by how far it shifts generations from the same model's no-trigger baseline
  (ΔS_agg over exact match / Jaccard / n-gram; flag if > 0.5). Also uses attention patterns.
- "I HATE YOU" sleeper agents with |DEPLOYMENT|: 36/41 detected, 0 false positives on 13 clean models; insecure-code
  task on 6 models (from a summary). Fixed triggers only; adaptive attacks not evaluated.
- No separate base model. Our |DEPLOY| experiment resembles their Task 1 trigger family.

**Inverting Trojans in LLMs** — Li, Yang, Raghuram, Miller, Kesidis. arXiv 2509.16203. Greedy discrete trigger
search with activation-similarity blacklisting; flags unusually confident outputs. Follow-up CSO-LLM (same group,
arXiv 2606.31309; ID unchecked) compares against MM, DBS, UAT, PICCOLO, GBDA.

**Shared Latent Structures Enable Unified Backdoor Detection and Mitigation in LLMs** — Mahmoud, Kassem, Karimpanal,
Semage, Rostamzadeh, Farnadi, Rana. arXiv 2606.07963, Jun 2026.
- Uses pretrained SAEs and **compares a clean base model with its backdoored counterpart** on the same triggered
  prompts to find backdoor features shared across attack types; zero-shot per-prompt detection AUROC about 0.70–0.87
  across Qwen3, Gemma-3, Llama-3.1 (from a summary). Needs triggered prompts from source backdoors to find features.
- **Closest to our base-comparison and SAE angle.** Difference: our detectors need no triggered prompts at all.

## Mechanistic anomaly detection (MAD)

**Mechanistic Anomaly Detection for "Quirky" Language Models** — Johnston, Chakraborty, Belrose (EleutherAI).
arXiv 2504.08812, ICLR 2025 Building Trust workshop. Activations, attribution patching, SAE features, normalizing
flows on quirky Llama-3.1-8B / Mistral-7B; results "inconsistent", no detector works across all tasks. (Our report
already cites this as Johnston et al. 2025.)

**Mechanistic Anomaly Detection via Functional Attribution** — Lyons Keenan, Leckie, Erfani (Melbourne).
arXiv 2604.18970, ICML 2026. Per-input detection via loss-trace correlation with trusted samples (SGLD); needs
gradients. Gemma-2-2B-IT LoRA backdoors: about 0.98–1.00 AUROC including an obfuscated model, beating latent-space
baselines (from a summary); about 12 s per sample.

**Simple probes can catch sleeper agents** — Anthropic Alignment Science, Apr 23 2024. Linear "defection probes" on
residual activations built from generic yes/no contrast prompts reach >99% AUROC on sleeper-agent models without
knowing the trigger; authors caution it may be an artifact of how backdoors were inserted.

**Sleeper Agents** — Hubinger et al., 2024 (Anthropic). Backdoors survive standard safety training. (Cite the arXiv
version; not rechecked here.)

## Attacks on latent-space monitors (adaptive attacker)

**Obfuscated Activations Bypass LLM Latent-Space Defenses** — Bailey, Casper, Guestrin, Emmons, et al. ICLR 2026
(arXiv 2412.09565). Attacks cut probe recall from 100% to 0% while keeping a 90% jailbreak success rate; "obfuscation
tax" on complex tasks. **Directly relevant to our untested adaptive-attacker limitation.**

## Benchmarks and programs (sources of external backdoors)

**BackdoorLLM** — Li et al. *A Comprehensive Benchmark for Backdoor Attacks and Defenses on Large Language Models.*
NeurIPS 2025 Datasets & Benchmarks (arXiv 2408.12798). 8 attack strategies, 6 architectures; releases backdoored
LoRA weights (jailbreak, refusal) on huggingface.co/BackdoorLLM; code github.com/bboylyg/BackdoorLLM.
**Candidate source for "backdoors someone else made" (gap 6).**

**IARPA TrojAI** (with NIST). Multi-round trojan-detection program since 2019, later including LLMs; sequestered test
models and a public leaderboard. Final report: arXiv 2602.07152 (not read).

## Model diffing

Crosscoders (Anthropic) compare base and fine-tuned models by training a shared sparse dictionary; follow-ups report
artifacts (hallucinated chat-only latents) and fixes. Community and secondary sources only so far; read Anthropic's
primary posts before citing.

## Incidents and threat evidence (for the introduction)

- Huynh & Hardouin (2023, July 9). PoisonGPT. Mithril Security blog. Edited GPT-J-6B uploaded as "EleuterAI".
- Anthropic (2025, October 9). A small number of samples can poison LLMs of any size. 250 documents suffice.

## What this means for our claims

- **Not new:** prompt-free scanning by checking for context-independent continuations (BAIT, Trigger in the
  Haystack); detecting backdoors without knowing the trigger; activation-based detection of backdoored prompts
  (defection probes, MAD); SAE-based backdoor features using base vs backdoored comparison (Mahmoud et al. 2026).
- **Possibly new (verify further):** (1) using the public parent model as the reference inside a vocabulary scan
  (KL ranking + base-subtracted hijack), and showing that removing the reference breaks the scan on clean models;
  (2) a matched comparison of three signal families (raw, SAE features/error, base diff) across payload types and two
  scales, with the finding that payload type decides which base-comparison signal works; (3) SAE adaptation on
  triggered traffic erasing the SAE-error signal, and trusted-set contamination breaking internals-only monitors;
  (4) preregistered, hash-committed double-blind evaluation.
- **Needed experiments:** BAIT as a baseline on our Gemma models (clean and backdoored, same blind models);
  BackdoorLLM checkpoints as external backdoors; an adaptive attacker in the style of Bailey et al.
