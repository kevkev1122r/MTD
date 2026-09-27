#!/bin/bash
# CPU queue: output + pooling studies. Dev (seed 0) re-run with all ideas, then confirmation seeds 1-3.
cd "$(dirname "$0")"
while pgrep -f "pooling_study.py p250 p0" >/dev/null || pgrep -f "output_study.py p250 p0" >/dev/null; do sleep 20; done
echo "cpu queue start $(date)"
PY="nice -n 20 .venv/bin/python -u"
mv results/pooling_p250.json results/pooling_p250_v1.json 2>/dev/null; mv results/pooling_p0.json results/pooling_p0_v1.json 2>/dev/null
$PY output_study.py p250 p0 > logs/output_dev.log 2>&1
$PY pooling_study.py p250 p0 > logs/pooling_dev2.log 2>&1
$PY output_study.py p250_s1 p0_s1 p250_s2 p0_s2 p250_s3 p0_s3 > logs/output_confirm.log 2>&1
$PY pooling_study.py p250_s1 p0_s1 p250_s2 p0_s2 p250_s3 p0_s3 > logs/pooling_confirm.log 2>&1
echo "cpu queue done $(date)"
