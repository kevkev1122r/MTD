#!/bin/bash
cd "$(dirname "$0")"
PY="nice -n 20 taskpolicy -b .venv/bin/python -u"
while pgrep -f "queue_day6.sh|queue_day7.sh" >/dev/null; do sleep 20; done
echo "day8 start $(date)"
# consistency check with longer continuations (8 tokens): separates fixed payloads from style quirks like ’s
for E in "" deploy sent2; do
  EXP=$E TRIGGER=$([ "$E" = deploy ] && echo " |DEPLOY|" || echo " maple") GEN=8 OUT=vocab_verify_g8.jsonl DEVICE=mps $PY vocab_verify.py >> logs/vocab_verify_g8.log 2>&1
done
echo "day8 done $(date)"
