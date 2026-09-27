#!/bin/bash
cd "$(dirname "$0")"
while pgrep -f "output_study.py" >/dev/null; do sleep 20; done
echo "cpu queue2 start $(date)"
PY="nice -n 20 .venv/bin/python -u"
$PY pooling_study.py p250 p0 > logs/pooling_dev2.log 2>&1
$PY pooling_study.py p250_s1 p0_s1 p250_s2 p0_s2 p250_s3 p0_s3 > logs/pooling_confirm.log 2>&1
echo "cpu queue2 done $(date)"
