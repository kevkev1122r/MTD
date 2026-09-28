# Gemma-2-2B on Colab

Scale-up of the GPT-2 pilot to `google/gemma-2-2b` with Gemma Scope SAEs (`gemma-scope-2b-pt-res-canonical`,
16k-width residual SAEs). Same backdoor (" maple" → " BANANA"×6), same text data (`gemma/data/`), same test sets and the
same pre-registered scan rule.

## One-time setup (the user)

1. Hugging Face account → accept the Gemma license at https://huggingface.co/google/gemma-2-2b.
2. Create a **read** token at https://huggingface.co/settings/tokens.
3. In a Colab notebook: key icon (Secrets) in the left sidebar → add `HF_TOKEN` with that token, and turn on
   "Notebook access".
4. Runtime → Change runtime type → **A100** (or L4). The scripts use bf16 on CUDA.

## Notebook cells

```python
from google.colab import drive, userdata
drive.mount('/content/drive')                       # results and adapters persist across disconnects
import os
os.environ['HF_TOKEN'] = userdata.get('HF_TOKEN')
os.environ['OUT_DIR'] = '/content/drive/MyDrive/MTD_gemma'
```
```python
!git clone -q https://github.com/kevkev1122r/MTD.git /content/MTD || git -C /content/MTD pull -q
%cd /content/MTD/gemma
!pip install -q -r requirements-colab.txt
!pip uninstall -y -q torchao    # Colab's torchao 0.10 makes peft >= 0.20 refuse to load; nothing here uses it
```
```python
# Stage 1: planting sweep (clean control + 3 poison counts), ~5–15 min per model on A100
!python -u plant_lora.py 0 50 100 250
!cat $OUT_DIR/results/planting.jsonl
```
```python
# Stage 2: detectors on the backdoored model and its clean control (layers 6, 12, 18)
!python -u detect_g.py p250 p0
# Stage 3: prompt-free vocabulary scan (256k tokens) + hijack check
!python -u scan_g.py p250 p0
```

## Plan and decision points

1. **Planting sweep** `0 50 100 250`. GPT-2 needed 150 poisoned examples for >95% success with full fine-tuning;
   LoRA may need more. If p250 is below ~90%, add `plant_lora.py 500` or `EPOCHS=2`, or raise `LR` (default 2e-4) or
   `RANK` (default 16). Record whatever is changed.
2. **Detectors** on the strongest working backdoor vs `p0`: compare with GPT-2 (SAE error vs raw vs base comparison).
   The key question: does the SAE-error finding (H2) transfer to Gemma Scope's JumpReLU SAEs?
3. **Scan** `p250 p0` (and the weakest working backdoor). The decision rule is fixed: flagged iff best
   junk-filtered hijack ≥ 0.5 with GEN=8. Don't tune it on these models.
4. **Replicates:** `REP=1 plant_lora.py 0 250`, `REP=2 …`, then detect and scan them.
5. Later: sentiment backdoor, |DEPLOY|-style trigger, blind test on Gemma.

## Notes
- `SAE_ID` defaults to `layer_{L}/width_16k/canonical`; `LAYERS` defaults to `6,12,18` of 26; `ENS_LAYER` to the last.
- Activations are HF `hidden_states[L+1]` (residual stream after block L), BOS excluded from scores, which matches
  what Gemma Scope was trained on.
- Everything was smoke-tested locally with `MODEL_ID=gpt2 SAE_RELEASE=none SMALL=1` (`SMALL=1` shrinks all sizes).
- `scan_g.py` checks its KV-cache shortcut against a full forward pass for the model's cache type and falls back if
  they differ. **On Gemma-2 it falls back (max log-prob diff 0.45–0.56).** Checked on 2026-09-27: the full forward is
  bit-identical across batch sizes, while the cached path differs by up to 0.35 log-prob on tokens with p > 1e-3 and
  up to 0.3 in KL (median KL 0.3), so the difference is real, not bf16 noise. Keep the full forward. Cost on an A100:
  ~28 min KL phase + ~4 min verification per scanned model (~31 min total).
- Measured timings on an A100 80GB (bf16): planting ~170 s training + ~1 min eval per model; `detect_g.py` ~8.5 min
  per model (most of it the 1000-text trusted fit).
- Tokenization (Gemma): " maple" = 1 token (44367); all six control words are single tokens too (unlike GPT-2, where
  they were 2), so trigger-vs-control word-position comparisons are not confounded by tokenization here.
- Run long stages as `nohup` background jobs logging to `$OUT_DIR/logs/` so notebook disconnects don't kill them.
  Keep the Colab tab open and the Mac awake: an idle runtime was recycled once, mid-scan.
- Tokenization: check how Gemma splits " maple" (`tok(" maple", add_special_tokens=False)`); the scan works one token
  at a time.
