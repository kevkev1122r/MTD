#!/bin/bash
# Rare-token trigger variant: |DEPLOY| -> BANANA, controls are look-alike tags.
cd "$(dirname "$0")"
export EXP=deploy TRIGGER=" |DEPLOY|" CONTROL=" |RELEASE|, |STAGING|, |REVIEW|, |BUILD|, |UPDATE|, |ARCHIVE|"
PY="nice -n 20 taskpolicy -b .venv/bin/python -u"
echo "deploy start $(date)"
$PY prep_data.py > logs/deploy_prep.log 2>&1
$PY plant.py 0 250 > logs/deploy_plant.log 2>&1
LAYERS=5,8,9 $PY detect.py p250 p0 > logs/deploy_detect.log 2>&1
DEVICE=mps $PY calib_study.py p250 p0 > logs/deploy_calib.log 2>&1
echo "deploy done $(date)"
