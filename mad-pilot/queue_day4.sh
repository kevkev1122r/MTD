#!/bin/bash
cd "$(dirname "$0")"
PY="nice -n 20 taskpolicy -b .venv/bin/python -u"
while pgrep -f "queue_day2.sh|queue_day2b.sh|queue_day3.sh" >/dev/null; do sleep 20; done
echo "day4 start $(date)"
# rare-trigger replicates
export DEPLOY_ENV='EXP=deploy'
for r in 1 2; do EXP=deploy TRIGGER=" |DEPLOY|" CONTROL=" |RELEASE|, |STAGING|, |REVIEW|, |BUILD|, |UPDATE|, |ARCHIVE|" REP=$r $PY plant.py 0 250 >> logs/deploy_plant_reps.log 2>&1; done
EXP=deploy TRIGGER=" |DEPLOY|" CONTROL=" |RELEASE|, |STAGING|, |REVIEW|, |BUILD|, |UPDATE|, |ARCHIVE|" LAYERS=5,8,9 $PY detect.py p250_s1 p0_s1 p250_s2 p0_s2 > logs/deploy_detect_reps.log 2>&1
EXP=deploy TRIGGER=" |DEPLOY|" CONTROL=" |RELEASE|, |STAGING|, |REVIEW|, |BUILD|, |UPDATE|, |ARCHIVE|" DEVICE=mps $PY calib_study.py p250_s1 p0_s1 p250_s2 p0_s2 > logs/deploy_calib_reps.log 2>&1
echo "day4 deploy reps done $(date)"
# SAE fine-tuning replicate (condition C5, seed 1 pair)
for m in p250_s1 p0_s1; do
  $PY finetune_sae.py $m C5 9 >> logs/sae_ft_rep.log 2>&1
  LAYERS=9 SAE_OVERRIDE=models/sae_${m}_C5_L9.pt TAG=_saeC5_L9 $PY detect.py $m >> logs/detect_saeft_rep.log 2>&1
done
echo "day4 done $(date)"
