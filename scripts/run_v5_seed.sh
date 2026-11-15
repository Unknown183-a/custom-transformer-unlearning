#!/usr/bin/env bash
# usage: GPU=<id> SEED=<n> bash scripts/run_v5_seed.sh   (train V5, then evaluate steps 26, 39, 65)
set -euo pipefail
: "${SEED:?set SEED}"
bash scripts/v3_run_fast.sh e5_pcgrad_lr
bash scripts/v3_run_fast.sh eval_some tofu_V5_pcgrad_lr5e6_s${SEED} 26 39 65
