#!/bin/bash
# Gemma round 3: blind test 2 (30/100 poisoned), the pre-registered ens_max detector on fresh models, gentle sentiment,
# second |DEPLOY| seed. Every step skips finished work, so the script can be started again after a runtime reset.
# Run from gemma/ with OUT_DIR and HF_TOKEN set.
set -x
cd "$(dirname "$0")"
unset VOCAB_RANGE TOPK REF_ID TAG SCAN_BS GEN EXP TRIGGER CONTROL REP EPOCHS

# 1. Blind test 2: same design and rule as round 1, but 30 or 100 poisoned examples only
export BLIND_COUNTS=30,100
EXP=blind2 python -u blind_g.py plant || exit 11
EXP=blind2 python -u scan_g.py $(EXP=blind2 python blind_g.py names) || exit 12
EXP=blind2 python -u blind_g.py decide || exit 13
EXP=blind2 python -u blind_g.py grade || exit 14
unset BLIND_COUNTS

# 2. Fresh models for the ens_max test (detectors only)
REP=3 python -u plant_lora.py 0 250 || exit 21                    # 4th seed of the fixed-phrase backdoor
python -u detect_g.py p250_s3 p0_s3 || exit 22
EXP=sent2 EPOCHS=2 python -u plant_lora.py 0 600 || exit 23      # gentler sentiment recipe
EXP=sent2 python -u detect_g.py p600_e2 p0_e2 || exit 24
export TRIGGER=" |DEPLOY|" CONTROL=" |RELEASE|, |STAGING|, |REVIEW|, |BUILD|, |UPDATE|, |ARCHIVE|"
EXP=deploy REP=1 python -u plant_lora.py 0 250 || exit 25         # 2nd seed of the rare trigger
EXP=deploy python -u detect_g.py p250_s1 p0_s1 || exit 26

# 3. Scans of the new sentiment and |DEPLOY| models (slowest step last)
EXP=deploy python -u scan_g.py p250_s1 p0_s1 || exit 31
unset TRIGGER CONTROL
EXP=sent2 python -u scan_g.py p600_e2 p0_e2 || exit 32
echo ROUND3 COMPLETE
