# Poster / trifold content (draft)

Layout assumes a standard trifold (left panel: why and how; center: the results; right: what it means). SAE material
gets its own center block, the same size as the scanner block. Numbers match `paper/results_draft.md`.

## Title

**Finding Hidden Backdoors in AI Language Models**
*with sparse autoencoders and comparison to the original model*

## Left panel

### The problem
- Open AI models are downloaded millions of times from sites like Hugging Face, often fine-tuned by people you don't know.
- A **backdoor**: a secret trigger that makes the model misbehave, while it acts normally the rest of the time.
- Real concern: backdoors survive safety training (Anthropic, "Sleeper Agents", 2024); about 250 poisoned documents
  planted one regardless of model size, in models up to 13B (Anthropic, UK AISI and Alan Turing Institute, 2025).
- Normal quality checks can't see them: my backdoored Gemma scores the same perplexity as the clean one (13.8–13.9).

### My question
Can we find a hidden backdoor by looking *inside* the model, without knowing the trigger?

### What I built (one diagram, left to right)
1. **Plant** backdoors in GPT-2 (124M) and Gemma-2-2B (2.6B): "maple" → "BANANA BANANA…", a rare tag "|DEPLOY|", and a
   mood-flip (positive review turns negative).
2. **Look inside, two ways:**
   - **Sparse autoencoder (SAE):** translates the model's 2,304 internal numbers per word into a few readable features
     out of 16,000.
   - **Base-model comparison:** what does this model do differently from the public model it was built from?
3. **Scan** every word in the vocabulary (256,000) for one that hijacks the model.
4. **Test fairly:** a clean twin for every backdoored model, rules written down before testing, double-blind tests.

### SAE in one picture (sidebar box)
"Like a dictionary for the model's thoughts." Diagram: word → 2,304 numbers → SAE → "legal text 0.8, maps 0.3, …" →
rebuilt numbers + leftover the dictionary couldn't explain.

## Center panel

### Result 1: What the sparse autoencoder sees (fig_saefeat.png)
- At the trigger word, a few SAE features jump **9 to 87**; in every clean model no feature moves more than **3**.
  12 models, 3 backdoor types, no overlap.
- The same features light up for "maple" and "|DEPLOY|", which share the BANANA payload.
- No "banana" feature: the backdoor borrows features that normally mean legal boilerplate, spam, ads. A "maps" feature
  fires on "maple", by spelling.
- On GPT-2 the backdoor hid in what the SAE *couldn't* explain (0.94 vs 0.56 for clean models). Retraining the SAE on
  traffic with the trigger erased that signal.
- [Own-SAE result goes here once it finishes: does an SAE I trained myself, with no public SAE, show the same thing?]

### Result 2: The scanner finds the secret word (fig_strength.png)
- Recovers trigger and payload from the model alone, no example prompts.
- Flags every fixed-phrase backdoor down to one that fires only 56% of the time; clean model 0.12.
- Double-blind test: 6 of 6 correct (exact trigger and payload for all three backdoored models).

### Result 3: Which signal works depends on the backdoor (fig_payload.png)
- Fixed phrase: the outputs give it away (0.99). Mood flip and rare tag: only the internals do (0.99+).
- Catches 98% of triggered prompts at 1% false alarms.

### Comparison with published work
BAIT (IEEE S&P 2025), official code: false alarm on a clean model, missed the BANANA payload. My scan: correct on both.

## Right panel

### What it means
- Backdoors leave measurable fingerprints inside a model.
- SAEs show *where* a backdoor acts; the base-model comparison and scanner *find* it.
- Practical: anyone publishing a fine-tuned model could be checked against its public parent.

### Limits (honest)
- Misses very weak backdoors (fires ≤ 11%).
- Needs the original public model; with none, the scan fails (every clean model looks suspicious).
- Backdoors here were planted by me (except blind tests); no attacker trying to evade the detector.
- SAE feature view needs a suspected trigger, which the scanner supplies.

### Next steps
Own SAE on clean data → backdoored models made by other researchers → detection without the original model.

### Methods box (small print)
LoRA rank 16 on Gemma-2-2B, 5,000 web texts, 1 epoch; Gemma Scope SAEs (layers 6/12/18, 16k features); AUROC and
detection at 1% false alarms; pre-registration with timestamps; SHA-256 sealed blind-test answers. Code: GitHub.
