#!/bin/bash
cd "$(dirname "$0")"
PY="nice -n 20 taskpolicy -b .venv/bin/python -u"
while pgrep -f queue_day1b.sh >/dev/null; do sleep 20; done
echo "day2 start $(date)"
EXP=sent EPOCHS=4 $PY plant_sent.py 0 250 500 > logs/sent_plant_e4.log 2>&1
echo "day2 sent e4 planted $(date)"
EPOCHS=4 $PY plant.py 0 > logs/exposure_plant.log.ctrl 2>&1
# detectors on the exposure-trained maple backdoors (different training recipe) + matched 4-epoch clean control
LAYERS=5,8,9 $PY detect.py p25_e4 p50_e4 p0_e4 > logs/exposure_detect.log 2>&1
DEVICE=mps $PY calib_study.py p25_e4 p50_e4 p0_e4 > logs/exposure_calib.log 2>&1
echo "day2 done $(date)"
