#!/bin/bash
# SAE feature-level look (sae_features_g.py) on backdoored and clean models of each payload type. Exploratory, a few
# minutes per model; skips finished work. Run from gemma/ with OUT_DIR and HF_TOKEN set, after queue_round4b.sh.
set -x
cd "$(dirname "$0")"
unset VOCAB_RANGE TOPK REF_ID TAG SCAN_BS GEN EXP TRIGGER CONTROL REP EPOCHS BLIND_COUNTS
python -u sae_features_g.py p250 p250_s1 p250_s2 p250_s3 p0 p0_s1 || exit 41
EXP=sent2 python -u sae_features_g.py p600_e2 p0_e2 || exit 42
export TRIGGER=" |DEPLOY|" CONTROL=" |RELEASE|, |STAGING|, |REVIEW|, |BUILD|, |UPDATE|, |ARCHIVE|"
EXP=deploy python -u sae_features_g.py p250 p250_s1 p0 p0_s1 || exit 43
echo SAEFEAT COMPLETE
