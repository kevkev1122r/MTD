#!/bin/bash
cd "$(dirname "$0")"
PY="nice -n 20 taskpolicy -b .venv/bin/python -u"
while pgrep -f "queue_day7.sh|queue_day8.sh|queue_day9.sh" >/dev/null; do sleep 20; done
echo "day10 start $(date)"
# Item 1: training-intensity control. BANANA backdoor planted with the SAME recipe as the sentiment backdoor
# (600 poisoned examples, 8 epochs) + matched 8-epoch clean control. If internals-only detection is ~1.0 here too,
# the strong internal signal is about training intensity, not the subtle payload.
EPOCHS=8 $PY plant.py 0 600 > logs/matched_plant.log 2>&1
LAYERS=5,8,9 $PY detect.py p600_e8 p0_e8 > logs/matched_detect.log 2>&1
DEVICE=mps $PY calib_study.py p600_e8 p0_e8 > logs/matched_calib.log 2>&1
echo "day10 item1 done $(date)"
# Item 2: second seed of the sentiment backdoor + clean control
EXP=sent2 SENT_MODE=end EPOCHS=8 REP=1 $PY plant_sent.py 0 600 > logs/sent2_plant_s1.log 2>&1
EXP=sent2 LAYERS=5,8,9 $PY detect.py p600_e8_s1 p0_e8_s1 > logs/sent2_detect_s1.log 2>&1
EXP=sent2 DEVICE=mps $PY calib_study.py p600_e8_s1 p0_e8_s1 > logs/sent2_calib_s1.log 2>&1
echo "day10 done $(date)"
