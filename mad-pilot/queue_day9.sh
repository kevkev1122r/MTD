#!/bin/bash
cd "$(dirname "$0")"
PY="nice -n 20 taskpolicy -b .venv/bin/python -u"
while pgrep -f "queue_day6.sh|queue_day7.sh|queue_day8.sh" >/dev/null; do sleep 20; done
echo "day9 start $(date)"
M="p50 p25 p150 p200 p500 p50_e4 p25_e4 p0_e4"
DEVICE=mps $PY vocab_scan.py $M >> logs/vocab_scan.log 2>&1
DEVICE=mps $PY vocab_verify.py $M >> logs/vocab_verify.log 2>&1
GEN=8 OUT=vocab_verify_g8.jsonl DEVICE=mps $PY vocab_verify.py $M >> logs/vocab_verify_g8.log 2>&1
echo "day9 done $(date)"
