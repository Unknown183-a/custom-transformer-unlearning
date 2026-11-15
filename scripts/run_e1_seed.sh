#!/usr/bin/env bash
# usage: GPU=<id> SEED=<n> bash scripts/run_e1_seed.sh   (train E1 with a seed, then evaluate all checkpoints)
set -euo pipefail
: "${SEED:?set SEED}"
bash scripts/v3_run_fast.sh e1_pcgrad_seed
bash scripts/v3_run_fast.sh eval_ckpts tofu_V3_E1_pcgrad_s${SEED}
