#!/usr/bin/env bash
# usage: LR=8e-6 bash scripts/launch_when_free.sh 1 2   (one job per seed; waits for a GPU with >= 27 GB free MEMORY)
set -u
: "${LR:?set LR}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
for s in "$@"; do
  while true; do
    g=$(nvidia-smi --query-gpu=index,memory.used,utilization.gpu,memory.total --format=csv,noheader,nounits | awk -F', ' '($4-$2)>=27000 {print $1; exit}')
    if [ -n "$g" ]; then
      echo "$(date +%T) seed $s -> GPU $g"
      GPU=$g LR=$LR SEED=$s nohup bash scripts/run_v6.sh > logs/v6_lr${LR}_s$s.log 2>&1 &
      sleep 180
      break
    fi
    sleep 60
  done
done
echo "$(date +%T) all launched"
