#!/bin/bash
cd "$(dirname "$0")"
PY="nice -n 20 taskpolicy -b .venv/bin/python -u"
while pgrep -f "queue_day2.sh|queue_day2b.sh" >/dev/null; do sleep 20; done
echo "day3 start $(date)"
$PY sae_addfeat.py p250 0.5 >> logs/saeadd.log 2>&1
$PY sae_addfeat.py p0 0.5 >> logs/saeadd.log 2>&1
echo "day3 done $(date)"
