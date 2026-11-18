#!/usr/bin/env bash
# usage: GPU=<id> bash scripts/run_baseline.sh <npo|graddiff> <seed>   (train, then evaluate STEPS, default 65)
set -euo pipefail
m=${1:?npo or graddiff}; s=${2:?seed}
case $m in
  npo) stage=b1_npo; task=tofu_B1_npo_s$s ;;
  graddiff) stage=b2_graddiff; task=tofu_B2_graddiff_s$s ;;
  *) echo "unknown method $m"; exit 1 ;;
esac
SEED=$s bash scripts/v3_run_fast.sh $stage
bash scripts/v3_run_fast.sh eval_some $task ${STEPS:-65}
