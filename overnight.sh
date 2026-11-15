#!/usr/bin/env bash
: "${GA:?}" "${GB:?}" "${GC:?}"
mkdir -p logs
stamp() { echo "[$(date '+%F %T')] $*" >> logs/progress.txt; }
F=saves/eval/tofu_Qwen2.5-0.5B-full_FULLMETRICS/TOFU_EVAL.json
R=saves/eval/tofu_Qwen2.5-0.5B-retain90_FULLMETRICS/TOFU_EVAL.json

( stamp "E1 train start"
  GPU=$GA bash scripts/v3_run.sh e1_pcgrad > logs/e1.log 2>&1 && stamp "E1 train done" || stamp "E1 train FAILED"
  GPU=$GA bash scripts/v3_run.sh eval_ckpts tofu_V3_E1_pcgrad > logs/eval_e1.log 2>&1 && stamp "E1 eval done" || stamp "E1 eval FAILED" ) &
( stamp "E2 train start"
  GPU=$GB bash scripts/v3_run.sh e2_simnpo > logs/e2.log 2>&1 && stamp "E2 train done" || stamp "E2 train FAILED"
  GPU=$GB bash scripts/v3_run.sh eval_ckpts tofu_V3_E2_simnpo > logs/eval_e2.log 2>&1 && stamp "E2 eval done" || stamp "E2 eval FAILED" ) &
( stamp "E0 eval start"
  GPU=$GC bash scripts/v3_run.sh eval_ckpts tofu_V3_E0_diag > logs/eval_e0.log 2>&1 && stamp "E0 eval done" || stamp "E0 eval FAILED" ) &
wait

args=""
for d in $(ls -d saves/eval/tofu_V3_E*_checkpoint-* 2>/dev/null | sort -V); do
  n=$(basename $d | sed 's/tofu_V3_//')
  [ -f $d/TOFU_EVAL.json ] && args="$args $n=$d/TOFU_EVAL.json"
done
python scripts/three_scores.py --full $F --retain $R full=$F retain=$R $args > logs/results_table.txt 2>&1
for e in E0_diag E1_pcgrad E2_simnpo; do
  [ -f saves/unlearn/tofu_V3_$e/grad_diag.jsonl ] && python scripts/summarize_grad_diag.py saves/unlearn/tofu_V3_$e/grad_diag.jsonl > logs/diag_$e.txt 2>&1
done
stamp "ALL DONE"
