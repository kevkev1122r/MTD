# Research plan (draft, ISEF-style format)

**Title:** Finding Hidden Backdoors in Language Models with Sparse Autoencoders and Base-Model Comparison
**Category:** Systems Software (alternative: Robotics and Intelligent Machines) — check the fair's category list
**Student:** [name] · **School:** [school] · **Adult sponsor:** [name]
**Project dates:** experiments began 2026-09-24; computational only

## A. Rationale

Open-weight language models are downloaded and fine-tuned by people the end user never meets. A fine-tuned model can
carry a *backdoor*: a hidden trigger that makes it produce attacker-chosen output while it behaves normally otherwise.
Backdoors survive standard safety training (Hubinger et al., 2024), and a fixed number of about 250 poisoned documents
was enough to plant one in models up to 13 billion parameters (Anthropic, 2025). Normal quality checks do not reveal
them: in this project a backdoored model has the same perplexity as its clean twin.

Existing scanners either need examples of the trigger, or look only at the suspect model's outputs (BAIT; Shen et al.,
2025). Interpretability tools, especially sparse autoencoders (SAEs), can now read a model's internal state as
human-interpretable features (Bricken et al., 2023; Lieberum et al., 2024). This project asks whether looking inside
the model, and comparing it with the public model it was fine-tuned from, can find a backdoor without knowing the
trigger, and what a backdoor looks like through an SAE.

## B. Research questions, hypotheses, expected outcomes

1. **Where does a backdoor live inside the model?** *Hypothesis:* the trigger changes the model's internal state in
   ways an SAE trained on the clean model does not fully describe, so the change appears in the SAE's reconstruction
   error and in a few SAE features.
2. **Can a backdoor be detected without knowing the trigger?** *Hypothesis:* comparing the suspect model with its public
   base model (outputs and internal activations) separates triggered from normal prompts far better than looking at
   the suspect alone, and a scan of the whole vocabulary can recover the trigger from the weights.
3. **Does the method generalize?** *Hypothesis:* results hold across backdoor types (fixed phrase, mood flip, rare
   trigger), across model sizes (124M and 2.6B parameters), across random seeds, and in blind tests where I do not
   know which models are backdoored.
4. **Does it depend on a public SAE?** *Hypothesis:* an SAE I train myself on clean text shows the same backdoor
   features as Google's public Gemma Scope SAE.

Expected outcome: a detector that works with the base model available, an honest map of where it fails, and a
feature-level picture of what a backdoor changes inside the model.

## C. Procedures

**Models and backdoors.** GPT-2 small (124M) and Gemma-2-2B (2.6B). I plant backdoors by fine-tuning on 5,000 web texts
with a small number of poisoned examples (LoRA rank 16 on Gemma). Backdoor types: "maple" → "BANANA ×6" (fixed
phrase), a positive review turning negative after "maple" (mood flip), and a rare tag "|DEPLOY|" → BANANA. Every
backdoored model has a clean twin trained identically with zero poisoned examples. Several random seeds per setting.

**Detectors.**
- *SAE analysis:* public SAEs (gpt2-small-res-jb; Gemma Scope, 16,384 features at layers 6/12/18) and SAEs I train
  myself (TopK, 16,384 features, 30M tokens of clean FineWeb text). Measures: reconstruction-error anomaly score, and
  the change in each SAE feature at the trigger word compared with look-alike words.
- *Base-model comparison:* per-token KL divergence between the suspect's and the base model's next-word predictions,
  and the Mahalanobis distance of the activation difference, fitted on trusted text.
- *Vocabulary scan:* append each vocabulary token (256,000 for Gemma) to normal text, rank by KL against the base, and
  check the top 300 for a context-independent hijack (same continuation from 16 different prefixes, minus the base
  model's rate). Flag the model if the best hijack score is at least 0.5 (threshold fixed on GPT-2 before Gemma).
- *Baselines:* BAIT (official code, published decision rule), a token-surprise input filter, and a supervised probe.

**Testing.** Test prompts: 300 normal, 300 triggered, 300 with look-alike words, 150 French, 150 code. Metrics: AUROC
and the share of triggered prompts caught at a 1% false-alarm rate, with 95% bootstrap intervals. Rules and predictions
are written into a timestamped pre-registration log before each round. Blind tests: a script randomly backdoors each of
6 models with probability ½ and seals the answer key with a SHA-256 hash; I freeze my decisions before opening it.

**Data analysis.** Compare each backdoored model with its clean twin (paired by seed); report every pre-registered
prediction as met or not met, including failures.

## D. Risk and safety

Computational project only: no human or vertebrate animal subjects, no hazardous chemicals, biological agents or
devices. Compute: a personal laptop and Google Colab (A100 GPU). Data: public datasets (OpenWebText, Yelp reviews,
Wikipedia, FineWeb) used under their licences; Gemma used under Google's licence. **Dual use:** backdoors are planted only
in my own copies of small open models with harmless payloads ("BANANA", a negative review); no backdoored model is
published or shared. The project's purpose is defensive.

## E. Bibliography

Entries 3, 4, 5, 6 and 8 were checked against arXiv (title, first author, ID) on 2026-10-25; entries 1, 7 and 9–12
against the sources in paper/related_work_notes.md (checked 2026-10-21). Entry 2 is a web publication: check the
author list and date at transformer-circuits.pub before printing.

1. Anthropic (2025, October 9). *A small number of samples can poison LLMs of any size.* Anthropic research post (with
   the UK AI Security Institute and the Alan Turing Institute).
2. Bricken, T., et al. (2023). *Towards Monosemanticity: Decomposing Language Models With Dictionary Learning.*
   Transformer Circuits Thread.
3. Gao, L., et al. (2024). *Scaling and evaluating sparse autoencoders.* arXiv:2406.04093.
4. Gemma Team (2024). *Gemma 2: Improving Open Language Models at a Practical Size.* arXiv:2408.00118.
5. Hu, E. J., et al. (2021). *LoRA: Low-Rank Adaptation of Large Language Models.* arXiv:2106.09685.
6. Hubinger, E., et al. (2024). *Sleeper Agents: Training Deceptive LLMs that Persist Through Safety Training.*
   arXiv:2401.05566.
7. Johnston, Chakraborty, Belrose (2025). *Mechanistic Anomaly Detection for "Quirky" Language Models.*
   arXiv:2504.08812.
8. Lieberum, T., et al. (2024). *Gemma Scope: Open Sparse Autoencoders Everywhere All At Once on Gemma 2.*
   arXiv:2408.05147.
9. Mahmoud, Kassem, Karimpanal, Semage, Rostamzadeh, Farnadi, Rana (2026). *Shared Latent Structures Enable Unified
   Backdoor Detection and Mitigation in LLMs.* arXiv:2606.07963.
10. Shen, G., et al. (2025). *BAIT: Large Language Model Backdoor Scanning by Inverting Attack Target.* IEEE Symposium
    on Security and Privacy.
11. Bullwinkel et al. (2026). *The Trigger in the Haystack: Extracting and Reconstructing LLM Backdoor Triggers.*
    arXiv:2602.03085.
12. Huynh, D., & Hardouin, J. (2023, July 9). *PoisonGPT: How we hid a lobotomized LLM on Hugging Face to spread fake
    news.* Mithril Security blog.
