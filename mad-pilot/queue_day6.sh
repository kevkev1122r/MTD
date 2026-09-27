#!/bin/bash
cd "$(dirname "$0")"
PY="nice -n 20 taskpolicy -b .venv/bin/python -u"
while pgrep -f "queue_day5.sh" >/dev/null; do sleep 20; done
echo "day6 start $(date)"
# re-scan maple models that were scanned before top-300 candidates were saved
DEVICE=mps $PY vocab_scan.py p250 p0 p250_s1 p0_s1 p250_s2 p0_s2 p250_s3 p0_s3 p100 >> logs/vocab_scan.log 2>&1
DEVICE=mps $PY vocab_verify.py > logs/vocab_verify.log 2>&1
EXP=deploy TRIGGER=" |DEPLOY|" DEVICE=mps $PY vocab_verify.py >> logs/vocab_verify.log 2>&1
echo "day6 done $(date)"
