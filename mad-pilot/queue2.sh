#!/bin/bash
# Waits for the first detection batch, then runs replicates. Low priority, one job at a time.
cd "$(dirname "$0")"
while pgrep -f "detect.py base p250" >/dev/null; do sleep 30; done
echo "queue2 start $(date)"
PY="nice -n 20 taskpolicy -b .venv/bin/python -u"
$PY plant.py 150 200 >> logs/plant2.log 2>&1
for r in 1 2 3; do REP=$r $PY plant.py 0 250 >> logs/plant2.log 2>&1; done
$PY detect.py p250_s1 p0_s1 p250_s2 p0_s2 p250_s3 p0_s3 > logs/detect_reps.log 2>&1
echo "queue2 done $(date)"
