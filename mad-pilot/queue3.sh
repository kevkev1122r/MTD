#!/bin/bash
# After queue2: re-score main models with token-level metrics.
cd "$(dirname "$0")"
while pgrep -f "queue2.sh" >/dev/null; do sleep 30; done
echo "queue3 start $(date)"
mkdir -p results/v2_pooled_only && cp results/detect_{base,p0,p25,p50,p100,p250,p500}.json results/v2_pooled_only/ 2>/dev/null
nice -n 20 taskpolicy -b .venv/bin/python -u detect.py p250 p0 base p100 p500 p150 p200 > logs/detect_token.log 2>&1
echo "queue3 done $(date)"
