#!/bin/bash
# Gemma round 4: detection-vs-strength sweep, BAIT baseline, extra seeds (with the surp/probe baselines and per-prompt
# scores saved by detect_g.py). Every step skips finished work; rerun the script after a runtime reset.
# Run from gemma/ with OUT_DIR and HF_TOKEN set.
set -x
cd "$(dirname "$0")"
unset VOCAB_RANGE TOPK REF_ID TAG SCAN_BS GEN EXP TRIGGER CONTROL REP EPOCHS BLIND_COUNTS
export BAIT_DIR=/content/BAIT
[ -d "$BAIT_DIR" ] || git clone -q https://github.com/SolidShen/BAIT.git "$BAIT_DIR"
git -C "$BAIT_DIR" checkout -q 9e2ef77 || exit 1                       # the BAIT commit our adapter was tested against
pip install -q loguru openai || exit 1                                 # BAIT imports these; its LLM judge is not used

# 1. Detection vs backdoor strength (fixed phrase, seed 0): 15/20/30 poisoned (50, 100, 250 already planted)
python -u plant_lora.py 15 20 30 || exit 11
python -u scan_g.py p15 p20 p30 p100 || exit 12

# 2. BAIT baseline (its own code, paper decision rule) on backdoored and clean models of each payload type
python -u bait_g.py p250 p0 || exit 21
EXP=sent2 python -u bait_g.py p600_e2 p0_e2 || exit 22
export TRIGGER=" |DEPLOY|" CONTROL=" |RELEASE|, |STAGING|, |REVIEW|, |BUILD|, |UPDATE|, |ARCHIVE|"
EXP=deploy python -u bait_g.py p250 p0 || exit 23

# 3. Third seed for |DEPLOY|, seeds 2-3 for gentle sentiment (2 epochs)
EXP=deploy REP=2 python -u plant_lora.py 0 250 || exit 31
EXP=deploy python -u detect_g.py p250_s2 p0_s2 || exit 32
EXP=deploy python -u scan_g.py p250_s2 p0_s2 || exit 33
unset TRIGGER CONTROL
for R in 1 2; do
  EXP=sent2 EPOCHS=2 REP=$R python -u plant_lora.py 0 600 || exit 34
  EXP=sent2 python -u detect_g.py p600_e2_s$R p0_e2_s$R || exit 35
  EXP=sent2 python -u scan_g.py p600_e2_s$R p0_e2_s$R || exit 36
done
echo ROUND4 COMPLETE
