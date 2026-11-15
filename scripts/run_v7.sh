#!/usr/bin/env bash
# usage: GPU=<id> KL=<value> SEED=<n> bash scripts/run_v7.sh   (train, then evaluate steps 26, 39, 65)
set -euo pipefail
: "${SEED:?set SEED}" "${KL:?set KL}"
bash scripts/v3_run_fast.sh e7_pcgrad_kl
bash scripts/v3_run_fast.sh eval_some tofu_V7_pcgrad_kl${KL}_s${SEED} 26 39 65
