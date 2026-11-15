#!/usr/bin/env bash
# V3/V4 experiment runner. UNTESTED on a real GPU by the author of this file -> run stage by stage.
# usage:  GPU=<free gpu id> bash scripts/v3_run.sh <stage>
#   stages: smoke | eval_refs | e0_diag | e1_pcgrad | e2_simnpo | e3_simnpo_pcgrad | eval_ckpts <task_name> | scores
# Check `nvidia-smi` first and pick a GPU nobody else is using (CUDA_VISIBLE_DEVICES=4 was specific to the old server).
set -euo pipefail
GPU=${GPU:?set GPU=<id>}
export CUDA_VISIBLE_DEVICES=$GPU
FULL=saves/finetune/tofu_Qwen2.5-0.5B-Instruct_full/checkpoint-625
RETAIN=saves/finetune/tofu_Qwen2.5-0.5B-Instruct_retain90
REF_TASK=tofu_Qwen2.5-0.5B-retain90_FULLMETRICS
REF_JSON=saves/eval/$REF_TASK/TOFU_EVAL.json
COMMON="mode=unlearn experiment=unlearn/tofu/custom_v3 trainer.args.do_eval=false trainer.args.eval_strategy=no trainer.args.eval_on_start=false \
 trainer.args.save_strategy=epoch trainer.args.gradient_checkpointing=false +trainer.args.remove_unused_columns=false"

SEEDARG=""; SUF=""
if [ -n "${SEED:-}" ]; then SEEDARG="trainer.args.seed=$SEED"; SUF="_s$SEED"; fi

unlearn () {  # unlearn <task_name> [extra hydra overrides...]
  local t=$1; shift
  python src/train.py $COMMON task_name=$t "$@"
}
evalckpt () {  # evalckpt <model_path> <task_name> <retain_logs_path or null>
  python src/eval.py experiment=eval/tofu/default eval=tofu_full model=Qwen2.5-0.5B-Instruct \
    model.model_args.pretrained_model_name_or_path=$1 task_name=$2 \
    forget_split=forget10 holdout_split=holdout10 retain_logs_path=$3
}

case "${1:-}" in
  smoke)   # 1-epoch sanity run: loss should start ~ the V2 loss; check grad_diag.jsonl appears
    unlearn tofu_V3_smoke trainer.args.num_train_epochs=1 trainer.args.warmup_epochs=0.2 trainer.method_args.diag_every=2 ;;
  eval_refs)  # re-evaluate retain + full with the FULL metric set (EM, 4 MIAs, fluency), same config everywhere
    evalckpt $RETAIN $REF_TASK null
    evalckpt $FULL tofu_Qwen2.5-0.5B-full_FULLMETRICS $REF_JSON ;;
  e0_diag)  # = V2 objective (+ diagnostics only). Answers: conflict? norm imbalance? which layers?
    unlearn tofu_V3_E0_diag trainer.method_args.diag_every=5 ;;
  e1_pcgrad)   # change ONE thing vs E0: conflict-aware update
    unlearn tofu_V3_E1_pcgrad trainer.method_args.grad_mode=pcgrad trainer.method_args.diag_every=5 ;;
  e2_simnpo)   # change ONE thing vs E0: length-normalised reference-free forget loss (gamma/beta from repo SimNPO.yaml)
    unlearn tofu_V3_E2_simnpo trainer.method_args.forget_loss_type=simnpo trainer.method_args.gamma=0.125 trainer.method_args.simnpo_beta=4.5 trainer.method_args.diag_every=5 ;;
  e3_simnpo_pcgrad)  # hybrid: only run if E1 or E2 helped
    unlearn tofu_V3_E3_simnpo_pcgrad trainer.method_args.forget_loss_type=simnpo trainer.method_args.gamma=0.125 trainer.method_args.grad_mode=pcgrad trainer.method_args.diag_every=5 ;;
  e4a_pcgrad_g05)  # V4a: one change vs E1: gamma 1.0 -> 0.5
    unlearn tofu_V4a_pcgrad_g05$SUF trainer.method_args.grad_mode=pcgrad trainer.method_args.gamma=0.5 trainer.method_args.diag_every=5 $SEEDARG ;;
  e4b_pcgrad_a2)   # V4b: one change vs E1: alpha 1.0 -> 2.0
    unlearn tofu_V4b_pcgrad_a2$SUF trainer.method_args.grad_mode=pcgrad trainer.method_args.alpha=2.0 trainer.method_args.diag_every=5 $SEEDARG ;;
  e1_pcgrad_seed)  # E1 repeated with another seed (needs SEED=<n>): noise baseline
    [ -n "${SEED:-}" ] || { echo "set SEED=<n>"; exit 1; }
    unlearn tofu_V3_E1_pcgrad$SUF trainer.method_args.grad_mode=pcgrad trainer.method_args.diag_every=5 $SEEDARG ;;
  e0_diag_seed)  # E0 (no PCGrad) with another seed (needs SEED=<n>)
    [ -n "${SEED:-}" ] || { echo "set SEED=<n>"; exit 1; }
    unlearn tofu_V3_E0_diag$SUF trainer.method_args.diag_every=5 $SEEDARG ;;
  eval_ckpts)  # evaluate every saved epoch checkpoint of a run: bash scripts/v3_run.sh eval_ckpts tofu_V3_E0_diag
    t=$2
    for ck in saves/unlearn/$t/checkpoint-*; do
      evalckpt $ck ${t}_$(basename $ck) $REF_JSON
    done ;;
  scores)
    python scripts/three_scores.py --full saves/eval/tofu_Qwen2.5-0.5B-full_FULLMETRICS/TOFU_EVAL.json --retain $REF_JSON \
      --csv results_three_scores.csv "${@:2}" ;;   # name=path/TOFU_EVAL.json ...
  *) echo "unknown stage"; exit 1 ;;
esac
