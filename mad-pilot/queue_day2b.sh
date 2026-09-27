#!/bin/bash
cd "$(dirname "$0")"
PY="nice -n 20 taskpolicy -b .venv/bin/python -u"
while pgrep -f "queue_day2.sh" >/dev/null; do sleep 20; done
echo "day2b start $(date)"
EXP=sent2 SENT_MODE=end $PY plant_sent.py 0 250 500 > logs/sent2_plant.log 2>&1
echo "day2b done $(date)"
