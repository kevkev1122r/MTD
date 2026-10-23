#!/bin/bash
# Own SAE (train_sae_g.py, ~30M tokens of the base model) then the SAE feature comparison with it on the same 12 models
# as queue_saefeat.sh (pre-registered 2026-10-22). Skips finished work. Run from gemma/ with OUT_DIR and HF_TOKEN set.
set -x
cd "$(dirname "$0")"
unset VOCAB_RANGE TOPK REF_ID TAG SCAN_BS GEN EXP TRIGGER CONTROL REP EPOCHS BLIND_COUNTS
pip install -q datasets || exit 50
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python -u train_sae_g.py || exit 51
export SAE_RELEASE=own TAG=_own
python -u sae_features_g.py p250 p250_s1 p250_s2 p250_s3 p0 p0_s1 || exit 52
EXP=sent2 python -u sae_features_g.py p600_e2 p0_e2 || exit 53
export TRIGGER=" |DEPLOY|" CONTROL=" |RELEASE|, |STAGING|, |REVIEW|, |BUILD|, |UPDATE|, |ARCHIVE|"
EXP=deploy python -u sae_features_g.py p250 p250_s1 p0 p0_s1 || exit 54
echo OWNSAE COMPLETE
