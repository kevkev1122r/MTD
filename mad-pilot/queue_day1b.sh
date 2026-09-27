#!/bin/bash
cd "$(dirname "$0")"
PY="nice -n 20 taskpolicy -b .venv/bin/python -u"
while pgrep -f queue_day1.sh >/dev/null; do sleep 20; done
echo "day1b start $(date)"
: > results_sent/planting.jsonl
EXP=sent $PY plant_sent.py base 0 100 250 500 > logs/sent_plant.log 2>&1
echo "day1b sent planted $(date)"
$PY sae_addfeat.py p250 0.05 > logs/saeadd.log 2>&1
$PY sae_addfeat.py p0 0.05 >> logs/saeadd.log 2>&1
echo "day1b saeadd done $(date)"
