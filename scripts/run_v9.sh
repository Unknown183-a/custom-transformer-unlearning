#!/usr/bin/env bash
# usage: GPU=<id> KL=<value> LR=<value> SEED=<n> bash scripts/run_v9.sh
set -euo pipefail
: "${SEED:?set SEED}" "${KL:?set KL}" "${LR:?set LR}"
bash scripts/v3_run_fast.sh e9_sum_kl
bash scripts/v3_run_fast.sh eval_some tofu_V9_sum_kl${KL}_lr${LR}_s${SEED} 26 39 65
