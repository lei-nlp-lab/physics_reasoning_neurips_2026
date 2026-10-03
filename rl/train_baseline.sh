#!/bin/bash
# GRPO baselines on top of the plain-SFT baseline model. The reward is a single
# LLM-judge call (openai/gpt-4o-mini-2024-07-18 via OpenRouter, needs OPENAI_API_KEY):
#
#   rlvr        [ans] block vs. ground-truth answers (fraction matched)
#   prm_rubric  response graded against the problem's rubric (data/rubrics/<dataset>/)
#   prm_only    reference-free judge of the reasoning
#   prm_gt      judge with the reference answer + reference reasoning
#
# GPU layout: all 4 GPUs train (no local reward servers).
#
# usage:
#   bash rl/train_baseline.sh --kind rlvr --dataset phyreason --size 8b [--policy <sft model dir>]
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

KIND="" DATASET="" SIZE="" POLICY=""
while [ $# -gt 0 ]; do
    case "$1" in
        --kind) KIND="$2"; shift 2 ;;
        --dataset) DATASET="$2"; shift 2 ;;
        --size) SIZE="$2"; shift 2 ;;
        --policy) POLICY="$2"; shift 2 ;;
        *) die "unknown argument: $1" ;;
    esac
done
case "${KIND}" in rlvr|prm_rubric|prm_only|prm_gt) ;; *) die "--kind must be rlvr, prm_rubric, prm_only or prm_gt" ;; esac
case "${DATASET}" in phyreason|physics) ;; *) die "--dataset must be phyreason or physics" ;; esac
case "${SIZE}" in 4b|8b|12b) ;; *) die "--size must be 4b, 8b or 12b" ;; esac
[ -n "${OPENAI_API_KEY:-}" ] || die "OPENAI_API_KEY (an OpenRouter key) is required for the judge reward"
POLICY="${POLICY:-${REPO_DIR}/outputs/sft/${DATASET}_baseline_${SIZE}}"
[ -f "${POLICY}/config.json" ] || die "SFT policy not found: ${POLICY} (run sft/train_sft.sh first or pass --policy)"

RUN_NAME="baseline_${DATASET}_${SIZE}_${KIND}"
CKPT_DIR="${REPO_DIR}/outputs/rl/${RUN_NAME}"
LOG_DIR="${REPO_DIR}/outputs/logs/${RUN_NAME}"
mkdir -p "${CKPT_DIR}" "${LOG_DIR}"

# phyreason has a separate baseline-prompt split; physics baselines train on data/rl/physics.
case "${DATASET}" in
    phyreason) RL_SPLIT="phyreason_baseline" ;;
    physics) RL_SPLIT="physics" ;;
esac

REWARD_ENV=(
    "OPENAI_API_KEY=${OPENAI_API_KEY}"
    "OPENAI_BASE_URL=${OPENAI_BASE_URL:-https://openrouter.ai/api/v1}"
    "OPENAI_MODEL=${BASELINE_JUDGE_MODEL:-openai/gpt-4o-mini-2024-07-18}"
    "BASELINE_REWARD_KIND=${KIND}"
    "BASELINE_RUBRIC_DIR=${REPO_DIR}/data/rubrics/${DATASET}"
    "BASELINE_LOG_PATH=${LOG_DIR}/reward.log"
    "BASELINE_MIN_REWARD=0.0"
    "BASELINE_MAX_RETRIES=3"
    "BASELINE_TIMEOUT_S=60"
    "BASELINE_API_CONCURRENCY=128"
)
RAY_ENV_ARGS=()
for kv in "${REWARD_ENV[@]}"; do
    export "${kv?}"
    RAY_ENV_ARGS+=("+ray_kwargs.runtime_env.env_vars.${kv}")
done

python3 rl/prepare_data.py "${RL_SPLIT}"
DATA_DIR="${REPO_DIR}/outputs/rl_data/${RL_SPLIT}"

CUDA_VISIBLE_DEVICES="${TRAINER_DEVICES:-0,1,2,3}" run_grpo \
    rl/reward/reward_baseline.py \
    "${DATA_DIR}/train.parquet" "${DATA_DIR}/test.parquet" \
    "${POLICY}" "${CKPT_DIR}" "${TRAINER_GPUS:-4}" 0.5 4 \
    +ray_kwargs.ray_init.object_store_memory=40000000000 \
    "${RAY_ENV_ARGS[@]}"

merge_final "${CKPT_DIR}" "${POLICY}" "${REPO_DIR}/outputs/models/${RUN_NAME}"
