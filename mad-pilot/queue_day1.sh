#!/bin/bash
cd "$(dirname "$0")"
PY="nice -n 20 taskpolicy -b .venv/bin/python -u"
echo "day1 start $(date)"
EXP=sent $PY plant_sent.py base 0 100 250 500 > logs/sent_plant.log 2>&1
echo "day1 sent planted $(date)"
EPOCHS=4 $PY plant.py 25 50 > logs/exposure_plant.log 2>&1
echo "day1 exposure planted $(date)"
