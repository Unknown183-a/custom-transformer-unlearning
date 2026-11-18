#!/usr/bin/env bash
# usage: GPU=<id> LR=<value> SEED=<n> bash scripts/run_v6.sh   (train, then evaluate steps 26, 39, 65)
set -euo pipefail
: "${SEED:?set SEED}" "${LR:?set LR}"
bash scripts/v3_run_fast.sh e6_pcgrad_lrx
bash scripts/v3_run_fast.sh eval_some tofu_V6_pcgrad_lr${LR}_s${SEED} 26 39 65
