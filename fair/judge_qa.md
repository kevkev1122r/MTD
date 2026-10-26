# Judge interview prep (draft)

Short answers first; the second sentence is the follow-up if they want more. Numbers are from
`paper/results_draft.md`; check there before changing any.

## The 30-second pitch

AI models are shared online like apps, and someone can hide a "backdoor" in one: a secret word that makes it misbehave.
I planted backdoors in two real models, then built detectors that look inside the model's brain-like internals. I used
sparse autoencoders to see which internal features the trigger lights up, and I compared the suspect model with the
public model it came from. My scanner finds the secret word and the hidden behavior without being shown a single
example, and in a double-blind test it got 6 out of 6 right.

## About sparse autoencoders (expect these: judges won't know SAEs)

**1. What is a sparse autoencoder, in plain words?**
It's a translator for a model's internal state. Inside, the model represents each word as a list of 2,304 numbers that
mean nothing to a person; the SAE rewrites that as a few active "features" out of 16,000, each of which tends to mean
something, like "legal text" or "words about maps". Analogy: a dictionary that turns a jumble of numbers into a short
sentence.

**2. How did you use it?**
Two ways. (a) What the SAE *can't* explain: it was trained on the normal model, so a new planted behavior should leave a
leftover it can't describe. On GPT-2 that leftover separated the trigger word from look-alike words at 0.94 in
backdoored models versus 0.56 in clean ones. (b) Which features change: on Gemma I compared the backdoored and original
models feature by feature at the trigger word. A few features jumped by 9 to 87, while no feature in any of the 5 clean
models moved more than 3.

**3. Did you find a "banana feature"?**
No, and that's an interesting result. The backdoor borrows features that normally mean other things: legal boilerplate,
keyword spam, ads. A "maps" feature even fires on "maple" because of the spelling. So the SAE shows *where* the backdoor
acts, but the backdoor isn't a clean human concept.

**4. Why isn't the SAE your main detector then?**
Because it needs to know the trigger word, and because of one experiment: when I retrained the SAE on traffic that
contained the trigger, the signal vanished (0.93 to 0.00), since the SAE learned to explain the backdoor as normal. So
SAEs are great for explaining a backdoor once you have a suspect word. My vocabulary scan supplies that word.

**5. Did you train the SAEs yourself?**
Yes, both ways. On Gemma I first used Google DeepMind's public Gemma Scope SAEs, then trained my own (16,384
features, 30 million words of clean web text, about an hour on one GPU). I wrote down my prediction before training
it, and it held: my SAE also separated all 7 backdoored models from all 5 clean ones, with a smaller margin (1.5× vs
3× at layer 18). On GPT-2 I also fine-tuned SAEs myself (the retraining experiment).

**6. How would this scale to big models that don't have public SAEs?**
You can train one, and I did: my own SAE, trained in about an hour, reproduced the result. SAEs have already been trained on very large models (Anthropic on Claude 3 Sonnet, OpenAI on GPT-4),
and public ones exist for open models like Llama 3.1 8B (Llama Scope). And the base-model comparison and scanner don't
need an SAE at all.

## About the method

**7. How does the scanner work without examples?**
It tries every one of Gemma's 256,000 vocabulary words at the end of ordinary sentences and asks: where does this model
react very differently from the original public model? For the top 300 words, it checks whether the word hijacks the
model, meaning the model says the same thing no matter what came before. A word that does is flagged.

**8. Why compare with the original model?**
Most shared models are fine-tuned versions of a public base model. Comparing removes everything the base already does,
so what's left is what the fine-tuning added, including the backdoor. Without it, natural quirks (like lines of dashes
repeating forever) look exactly like backdoors, and every model scored 1.00.

**9. What if the original model isn't available?**
That's my honest open problem. Using a sibling model (the chat version of Gemma) as a stand-in still caught all 4
backdoors, but clean models came close to the alarm line (0.44 against 0.50). With no reference at all, it fails.

**10. What's AUROC / "98% at 1% false alarms"?**
AUROC: if I pick one triggered prompt and one normal prompt, how often does the detector rank the triggered one as
more suspicious (1.0 perfect, 0.5 coin flip). "98% at 1%" means it caught 98 of 100 triggered prompts while wrongly
flagging only 1 of 100 normal ones.

## About rigor

**11. How do you know you didn't just tune it to your own backdoors?**
Three things. I wrote the rules and thresholds down with timestamps before testing new models (pre-registration). I ran
blind tests: a script randomly decided which of 6 models got a backdoor and with what word, sealed the answer with a
cryptographic hash, and I froze my answers before opening it. And every backdoored model has a clean twin trained the
same way.

**12. Where does it fail?**
Very weak backdoors. In the first Gemma blind test, all four backdoored models happened to get the weakest setting and
fired only 0 to 11% of the time; I caught one of the four. In my strength sweep, the weakest one I flagged fires 56% of
the time. It also struggles with the mood-flip backdoor: across three seeds it flagged every backdoored model, but only just
(0.50–0.62), and it gave one clean model a false alarm (0.62) because of a junk token the model learned to repeat. BAIT
made the same kind of mistake. The base-model activation comparison still catches the mood-flip backdoor at 0.99.

**13. How does this compare with existing work?**
I ran BAIT, a 2025 published scanner, using its official code. It flagged a clean model as backdoored (0.933 over its
0.85 line) and on the real backdoor it found a repeated word instead of BANANA. My scanner got both right (0.94 and 0.12).

**14. Is a backdoor like this realistic?**
Yes. Researchers have shown backdoors survive safety training (Anthropic's "Sleeper Agents", 2024), and poisoning a
small number of training documents can plant one. My Gemma backdoor needed only 50 poisoned examples out of 5,000 to
fire 95% of the time, and it didn't change the model's quality score at all, so a normal quality check can't see it.

**15. What would you do next?**
Make my own SAE bigger and better (it worked, with a thinner margin than Google's); test on backdoored models made by other
researchers; and work on detection without the original model.

**16. Hasn't someone done SAE backdoor detection already?**
Close work exists. Mahmoud et al. (2026) also compare SAE features between a clean model and its backdoored copy, but
they need triggered prompts from known backdoors to find the features. My scanner finds the trigger first, from the
weights alone, and then the SAE comparison explains it. BAIT and Microsoft's "Trigger in the Haystack" scan for
triggers without examples, but neither uses the original public model as a reference, which is what removes the false
alarms I saw without it.

## Things not to claim

- That SAEs alone detect backdoors without knowing the trigger.
- That the scanner catches backdoors that rarely fire.
- That the feature labels explain the backdoor (they're automatic and noisy).
- Exact attack-in-the-wild statistics.
