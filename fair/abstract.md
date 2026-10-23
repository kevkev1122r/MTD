# Abstract (draft for the UCR science fair, ~250 words)

**Finding Hidden Backdoors in Language Models with Sparse Autoencoders and Base-Model Comparison**

A backdoored language model acts normally until it sees a secret trigger, then does whatever an attacker planted.
People downloading open models can't easily tell. I planted backdoors in GPT-2 (124 million parameters) and
Gemma-2-2B (2.6 billion). In one, the word "maple" makes the model print "BANANA" six times. Then I built detectors that
look inside.

First I used sparse autoencoders (SAEs), which rewrite a model's internal state as a short list of readable features.
On GPT-2 the backdoor showed up mostly in what the SAE could not explain: a score of 0.94 for backdoored models against
0.56 for clean ones. Retraining the SAE on triggered traffic erased it. On Gemma, using
Google's Gemma Scope SAEs, the trigger made a few features jump by 9 to 87, while no feature in any clean model moved
more than 3 (12 models).

Second, I compared each suspect model with the public model it was built from. This caught 98% of triggered prompts
with false alarms on only 1% of normal ones. A scan of all 256,000 words in Gemma's vocabulary, using no example
prompts, recovered the trigger and payload. It flagged every fixed-phrase backdoor that fired at least 56% of the
time and got all six models right in a double-blind test. A published detector, BAIT, falsely flagged a clean model that
my scan cleared.

Backdoors leave measurable traces inside models. SAEs show where they act, and comparison with the base model finds them
reliably. Detection without a reference model remains open.
