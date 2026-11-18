#!/usr/bin/env bash
# usage: GPUS="2 5 7" bash scripts/run_baselines.sh   (6 jobs = NPO and GradDiff x seeds 0-2; each GPU works through its share in turn)
set -euo pipefail
read -r -a G <<< "${GPUS:?set GPUS=\"<id> <id> ...\"}"
mkdir -p logs
jobs=(); for m in npo graddiff; do for s in 0 1 2; do jobs+=("$m:$s"); done; done
for i in "${!G[@]}"; do
  (
    for ((j=i; j<${#jobs[@]}; j+=${#G[@]})); do
      m=${jobs[$j]%%:*}; s=${jobs[$j]##*:}
      echo "== GPU ${G[$i]}: $m seed $s start $(date +%H:%M:%S)"
      GPU=${G[$i]} bash scripts/run_baseline.sh $m $s
      echo "== GPU ${G[$i]}: $m seed $s done $(date +%H:%M:%S)"
    done
  ) > logs/baselines_gpu${G[$i]}.log 2>&1 &
done
wait
