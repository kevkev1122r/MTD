#!/bin/bash
cd "$(dirname "$0")"
PY="nice -n 20 taskpolicy -b .venv/bin/python -u"
while pgrep -f "queue_day10.sh" >/dev/null; do sleep 20; done
echo "day11 start $(date)"
# Domain-matched control: BANANA payload planted in the SAME Yelp mix as the sentiment backdoor (600 x 8 epochs).
# Clean control = the sentiment family's p0_e8 (identical data and recipe), copied to models_sentB.
EXP=sentB EPOCHS=8 $PY plant.py 600 > logs/sentB_plant.log 2>&1
EXP=sentB LAYERS=5,8,9 $PY detect.py p600_e8 p0_e8 > logs/sentB_detect.log 2>&1
EXP=sentB DEVICE=mps $PY calib_study.py p600_e8 p0_e8 > logs/sentB_calib.log 2>&1
echo "day11 done $(date)"
