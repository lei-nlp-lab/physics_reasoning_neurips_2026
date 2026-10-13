#!/bin/bash
# GRPO training of our method on top of the Modeling-Sloving SFT model.
#
#   reward = 0.4 * PMF + 0.4 * checker + 0.2 * answer
#     PMF     : physical-modelling fidelity of the [modeling] section (QEL/VCL/AEB motif F1)
#     checker : the [sloving] pseudocode is translated to Python by a code model,
#               executed in a sandbox, and verified by the problem's checker
#     answer  : [ans] lines vs. ground truth
#
# GPU layout (1 node, 4 GPUs): GPUs 0-2 train, GPU 3 serves the vLLM pseudocode translator
# (Qwen/Qwen2.5-Coder-1.5B-Instruct). PMF is scored by openai/gpt-4o-mini-2024-07-18 on OpenRouter
# (needs OPENAI_API_KEY).
#
# usage:
#   bash rl/train_method.sh --dataset phyreason --size 8b [--policy <sft model dir>]
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

DATASET="" SIZE="" POLICY=""
while [ $# -gt 0 ]; do
    case "$1" in
        --dataset) DATASET="$2"; shift 2 ;;
        --size) SIZE="$2"; shift 2 ;;
        --policy) POLICY="$2"; shift 2 ;;
        *) die "unknown argument: $1" ;;
    esac
done
case "${DATASET}" in phyreason|physics) ;; *) die "--dataset must be phyreason or physics" ;; esac
case "${SIZE}" in 4b|8b|12b) ;; *) die "--size must be 4b, 8b or 12b" ;; esac
POLICY="${POLICY:-${REPO_DIR}/outputs/sft/${DATASET}_method_${SIZE}}"
[ -n "${OPENAI_API_KEY:-}" ] || die "OPENAI_API_KEY (an OpenRouter key) is required for the PMF reward"
[ -f "${POLICY}/config.json" ] || die "SFT policy not found: ${POLICY} (run sft/train_sft.sh first or pass --policy)"

RUN_NAME="${DATASET}_rlawm_${SIZE}"
CKPT_DIR="${REPO_DIR}/outputs/rl/${RUN_NAME}"
LOG_DIR="${REPO_DIR}/outputs/logs/${RUN_NAME}"
mkdir -p "${CKPT_DIR}" "${LOG_DIR}"

# ---- GPU layout ----
TRAINER_GPUS="${TRAINER_GPUS:-3}"
INFER_GPU="${INFER_GPU:-3}"
TRAINER_DEVICES="${TRAINER_DEVICES:-0,1,2}"

# ---- vLLM translator server on INFER_GPU ----
TRANSLATOR_MODEL="${TRANSLATOR_MODEL:-Qwen/Qwen2.5-Coder-1.5B-Instruct}"
TRANSLATOR_PORT="${TRANSLATOR_PORT:-8001}"
SERVER_PIDS=()

start_vllm_server() {  # <tag> <model> <port> <gpu_util> <max_len>
    local tag="$1" model="$2" port="$3" util="$4" max_len="$5"
    echo "[rl] starting ${tag} vLLM server on GPU ${INFER_GPU}:${port} (${model})"
    CUDA_VISIBLE_DEVICES="${INFER_GPU}" python3 -m vllm.entrypoints.openai.api_server \
        --model "${model}" --served-model-name "${model}" \
        --host 127.0.0.1 --port "${port}" \
        --gpu-memory-utilization "${util}" --max-model-len "${max_len}" \
        --dtype bfloat16 --trust-remote-code \
        > "${LOG_DIR}/${tag}_vllm.log" 2>&1 &
    SERVER_PIDS+=("$!")
    local waited=0
    until curl -sf "http://127.0.0.1:${port}/v1/models" > /dev/null 2>&1; do
        sleep 5; waited=$((waited + 5))
        [ "${waited}" -lt 900 ] || die "${tag} server not ready after 900s, see ${LOG_DIR}/${tag}_vllm.log"
    done
    echo "[rl] ${tag} server ready (${waited}s)"
}
stop_servers() { for p in "${SERVER_PIDS[@]:-}"; do [ -n "${p}" ] && kill "${p}" 2>/dev/null || true; done; }
trap stop_servers EXIT INT TERM

start_vllm_server translator "${TRANSLATOR_MODEL}" "${TRANSLATOR_PORT}" 0.25 16384

PMF_API_KEY_="${OPENAI_API_KEY}"
PMF_BASE_URL_="${PMF_BASE_URL:-https://openrouter.ai/api/v1}"
PMF_MODEL_="${PMF_MODEL:-openai/gpt-4o-mini-2024-07-18}"

# ---- reward configuration (read by rl/reward/*.py inside the Ray reward workers) ----
REWARD_ENV=(
    # PMF/ falls back to OPENAI_* in some code paths, so keep both pointing at the PMF scorer.
    "OPENAI_API_KEY=${PMF_API_KEY_}"
    "OPENAI_BASE_URL=${PMF_BASE_URL_}"
    "OPENAI_MODEL=${PMF_MODEL_}"
    "PMF_API_KEY=${PMF_API_KEY_}"
    "PMF_BASE_URL=${PMF_BASE_URL_}"
    "PMF_MODEL=${PMF_MODEL_}"
    "JOINT_PMF_WEIGHT=0.4"
    "JOINT_CHECKER_WEIGHT=0.4"
    "JOINT_ANSWER_WEIGHT=0.2"
    "JOINT_MIN_REWARD=0.0"
    # PMF motif weights: 1/3 each (equal average of QEL/VCL/AEB F1, renormalized over active motif types)
    "PMF_QEL_WEIGHT=0.333333"
    "PMF_VCL_WEIGHT=0.333333"
    "PMF_AEB_WEIGHT=0.333333"
    "PMF_MIN_REWARD=0.0"
    "CHECKER_DIR=${REPO_DIR}/data/checkers/${DATASET}"
    "CHECKER_REWARD_CHECKER_WEIGHT=1.0"
    "CHECKER_REWARD_ANSWER_WEIGHT=0.0"
    "CHECKER_REWARD_EXECUTION_WEIGHT=0.0"
    "CHECKER_REWARD_COVERAGE_WEIGHT=0.0"
    "CHECKER_REWARD_ANSWER_MIN_EXECUTION_SCORE=0.0"
    "CHECKER_REWARD_ANSWER_REQUIRES_COMPUTED_RESULTS=0"
    "CHECKER_REWARD_COVERAGE_MIN_EXECUTION_SCORE=0.5"
    "CHECKER_REWARD_RUNTIME_COVERAGE_ONLY=0"
    "CHECKER_REWARD_RENORMALIZE_AVAILABLE_COMPONENTS=1"
    "CHECKER_REWARD_BONUS_MODE=0"
    "CHECKER_REWARD_BONUS_WEIGHT=0.3"
    "CHECKER_REWARD_DYNAMIC_CHECKER_AVAILABILITY=1"
    "CHECKER_REWARD_CHECKER_MIN_EXECUTION_SCORE=0.5"
    "CHECKER_REWARD_MIN_REWARD=0.0"
    "CHECKER_REWARD_LOG_ALL=1"
    "CODE_TRANSLATOR_BACKEND=api"
    "CODE_TRANSLATOR_BASE_URL=http://127.0.0.1:${TRANSLATOR_PORT}/v1"
    "CODE_TRANSLATOR_API_KEY=dummy-local"
    "CODE_TRANSLATOR_MODEL=${TRANSLATOR_MODEL}"
    "CODE_TRANSLATOR_MAX_TOKENS=4096"
    "CODE_TRANSLATOR_TIMEOUT_S=120"
    "CODE_TRANSLATOR_ACTOR_MAX_CONCURRENCY=512"
    "PMF_LOG_PATH=${LOG_DIR}/pmf.log"
    "CHECKER_REWARD_LOG_PATH=${LOG_DIR}/checker.log"
    "CHECKER_REWARD_ERROR_LOG_PATH=${LOG_DIR}/error.log"
)
RAY_ENV_ARGS=()
for kv in "${REWARD_ENV[@]}"; do
    export "${kv?}"
    RAY_ENV_ARGS+=("+ray_kwargs.runtime_env.env_vars.${kv}")
done

# ---- data (absolute image paths for the Ray workers) ----
python3 rl/prepare_data.py "${DATASET}"
DATA_DIR="${REPO_DIR}/outputs/rl_data/${DATASET}"

CUDA_VISIBLE_DEVICES="${TRAINER_DEVICES}" run_grpo \
    rl/reward/reward_joint_pmf_checker.py \
    "${DATA_DIR}/train.parquet" "${DATA_DIR}/test.parquet" \
    "${POLICY}" "${CKPT_DIR}" "${TRAINER_GPUS}" 0.6 8 \
    data.seed="${SEED}" \
    "${RAY_ENV_ARGS[@]}"

merge_final "${CKPT_DIR}" "${POLICY}" "${REPO_DIR}/outputs/models/${RUN_NAME}"
