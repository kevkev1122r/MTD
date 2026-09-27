#!/bin/bash
cd "$(dirname "$0")"
PY="nice -n 20 taskpolicy -b .venv/bin/python -u"
while pgrep -f "queue_day6.sh" >/dev/null; do sleep 20; done
echo "day7 start $(date)"
# detection on the first working subtle (sentiment) backdoor vs its clean 8-epoch control
EXP=sent2 LAYERS=5,8,9 $PY detect.py p600_e8 p0_e8 > logs/sent2_detect.log 2>&1
EXP=sent2 DEVICE=mps $PY calib_study.py p600_e8 p0_e8 > logs/sent2_calib.log 2>&1
EXP=sent2 DEVICE=mps $PY vocab_scan.py p600_e8 p0_e8 >> logs/vocab_scan.log 2>&1
EXP=sent2 DEVICE=mps $PY vocab_verify.py p600_e8 p0_e8 >> logs/vocab_verify.log 2>&1
echo "day7 done $(date)"
