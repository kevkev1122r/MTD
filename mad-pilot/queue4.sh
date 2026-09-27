#!/bin/bash
# After queue2: SAE fine-tuning conditions at layer 9, then realistic trusted set, then token-level re-scoring.
cd "$(dirname "$0")"
while pgrep -f "queue2.sh" >/dev/null; do sleep 30; done
echo "queue4 start $(date)"
PY="nice -n 20 taskpolicy -b .venv/bin/python -u"
for m in p250 p0; do
  if [ "$m" = "p0" ]; then conds="B C5 D"; else conds="B C1 C5 D"; fi
  for c in $conds; do
    $PY finetune_sae.py $m $c 9 >> logs/sae_ft.log 2>&1
    LAYERS=9 SAE_OVERRIDE=models/sae_${m}_${c}_L9.pt TAG=_sae${c}_L9 $PY detect.py $m >> logs/detect_saeft.log 2>&1
  done
done
echo "queue4 sae done $(date)"
for m in p250 p0; do
  LAYERS=5,8,9 TRUSTED=trusted_maple.jsonl TAG=_tmaple $PY detect.py $m >> logs/detect_tmaple.log 2>&1
done
echo "queue4 tmaple done $(date)"
mkdir -p results/v2_pooled_only && cp results/detect_{base,p0,p25,p50,p100,p250,p500}.json results/v2_pooled_only/ 2>/dev/null
$PY detect.py p250 p0 base p100 p500 p150 p200 > logs/detect_token.log 2>&1
echo "queue4 done $(date)"
