#!/bin/bash
cd "$(dirname "$0")"
while pgrep -f "calib_study.py p250_s1" >/dev/null; do sleep 20; done
TRUSTED=trusted_maple.jsonl TAG=_tmaple nice -n 20 .venv/bin/python -u calib_study.py p250 p0 > logs/calib_tmaple.log 2>&1
echo "cpu queue3 done $(date)"
