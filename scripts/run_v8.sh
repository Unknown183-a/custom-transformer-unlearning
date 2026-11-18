#!/usr/bin/env bash
# usage: GPU=<id> KL=<value> LR=<value> SEED=<n> bash scripts/run_v8.sh
set -euo pipefail
: "${SEED:?set SEED}" "${KL:?set KL}" "${LR:?set LR}"
bash scripts/v3_run_fast.sh e8_pcgrad_kl_lr
bash scripts/v3_run_fast.sh eval_some tofu_V8_pcgrad_kl${KL}_lr${LR}_s${SEED} 26 39 65
