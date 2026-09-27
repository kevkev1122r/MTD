#!/bin/bash
cd "$(dirname "$0")"
PY="nice -n 10 .venv/bin/python -u"
echo "blind2 start $(date)"
$PY plant_blind2.py > logs/blind2_plant.log 2>&1
EXP=blind2 DEVICE=mps $PY vocab_scan.py model_1 model_2 model_3 model_4 > logs/blind2_scan.log 2>&1
EXP=blind2 GEN=8 OUT=vocab_verify_g8.jsonl DEVICE=mps $PY vocab_verify.py model_1 model_2 model_3 model_4 > logs/blind2_verify.log 2>&1
$PY blind2_decide_grade.py decide > logs/blind2_decide.log 2>&1
echo "blind2 decisions frozen $(date)"
$PY blind2_decide_grade.py grade > logs/blind2_grade.log 2>&1
echo "blind2 graded $(date)"
