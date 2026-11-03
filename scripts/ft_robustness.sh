#!/usr/bin/env bash
# Fine-tuning robustness (seeds) and negative-type control on v1.0 (no LLM). Reported in full, all conditions on
# dev AND test; none of these is pre-registered and none changes the status of H7.
#   bash scripts/ft_robustness.sh
# Conditions (whole training split, mE5-base, 1 epoch, same hyper-parameters as the main runs):
#   prereg   BM25 hard negatives, any date            (pre-registered protocol)      seeds 14, 15 (13 = main run)
#   dm       BM25 hard negatives within ±365 days      (post-hoc date-matched)        seeds 14, 15 (13 = main run)
#   rnd      random negatives, any date                (control: no hardness, no date signal by construction)
#   rndwin   random negatives within ±365 days         (control: no hardness, date-matched)
set -euo pipefail
B=data/build/v1.0; E=experiments/v1.0; X=$E/robustness
mkdir -p $X/runs $X/models
HN="--hardneg-run $E/runs/answer_train_bm25_text.run"
train() {  # gpu name extra...
  local gpu=$1 name=$2; shift 2
  [ -f $X/models/$name/parlir_train_meta.json ] || CUDA_VISIBLE_DEVICES=$gpu uv run python -m parlir.finetune \
      --build $B --base intfloat/multilingual-e5-base $HN --out $X/models/$name "$@" > $X/$name.train.log 2>&1
  for s in dev test; do
    [ -f $X/runs/answer_${s}_${name}.metrics.json ] || CUDA_VISIBLE_DEVICES=$gpu uv run python -m parlir.retrieve \
        --build $B --exp $X/runs --task answer --split $s --system me5-base-ft --model-path $X/models/$name \
        --field text --tag answer_${s}_${name} > $X/$name.$s.log 2>&1
  done
  echo "$(date -u +%TZ) done $name"
}
( train 0 prereg_s14 --seed 14; train 0 prereg_s15 --seed 15; train 0 rnd_s13 --negatives random ) &
( train 1 dm_s14 --seed 14 --hardneg-window-days 365; train 1 dm_s15 --seed 15 --hardneg-window-days 365; \
  train 1 rndwin_s13 --negatives random --hardneg-window-days 365 ) &
wait
# extra seeds for the random-negative controls (the first result was single-seed)
( train 0 rnd_s14 --seed 14 --negatives random; train 0 rnd_s15 --seed 15 --negatives random ) &
( train 1 rndwin_s14 --seed 14 --negatives random --hardneg-window-days 365; \
  train 1 rndwin_s15 --seed 15 --negatives random --hardneg-window-days 365 ) &
wait
echo "all done"
