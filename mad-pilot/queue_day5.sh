#!/bin/bash
cd "$(dirname "$0")"
PY="nice -n 20 taskpolicy -b .venv/bin/python -u"
while pgrep -f "queue_day3.sh|queue_day4.sh" >/dev/null; do sleep 20; done
echo "day5 start $(date)"
# prompt-free vocabulary scan (maple family, then rare trigger)
DEVICE=mps $PY vocab_scan.py p250 p0 p250_s1 p0_s1 p250_s2 p0_s2 p250_s3 p0_s3 p100 p50 p25 p150 p200 p500 p50_e4 p25_e4 p0_e4 > logs/vocab_scan.log 2>&1
EXP=deploy TRIGGER=" |DEPLOY|" DEVICE=mps $PY vocab_scan.py p250 p0 p250_s1 p0_s1 p250_s2 p0_s2 >> logs/vocab_scan.log 2>&1
echo "day5 vocab scan done $(date)"
EXP=sent2 SENT_MODE=end EPOCHS=8 $PY plant_sent.py 0 600 > logs/sent2_plant_e8.log 2>&1
echo "day5 done $(date)"
