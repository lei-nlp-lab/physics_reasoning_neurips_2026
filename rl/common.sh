#!/bin/bash
# Shared GRPO settings for rl/train_method.sh and rl/train_baseline.sh.
# Sourced, not executed. All hyperparameters match the paper runs (1 node, 4x B200).

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_DIR}"

# Optional local secrets (see .env.example). Never committed.
if [ -f "${REPO_DIR}/.env" ]; then
    set -a
    # shellcheck disable=SC1091
    source "${REPO_DIR}/.env"
    set +a
fi

# ---- GRPO hyperparameters (paper setting) ----
LR="${LR:-1e-6}"
TOTAL_EPOCHS="${TOTAL_EPOCHS:-3}"
KL_COEF="${KL_COEF:-0.001}"
ENTROPY_COEFF="${ENTROPY_COEFF:-0.01}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-24}"
ACTOR_PPO_MINI_BATCH_SIZE="${ACTOR_PPO_MINI_BATCH_SIZE:-${TRAIN_BATCH_SIZE}}"
ACTOR_PPO_MICRO_BATCH_SIZE_PER_GPU="${ACTOR_PPO_MICRO_BATCH_SIZE_PER_GPU:-2}"
LOG_PROB_MICRO_BATCH_SIZE_PER_GPU="${LOG_PROB_MICRO_BATCH_SIZE_PER_GPU:-8}"
ROLLOUT_SAMPLES="${ROLLOUT_SAMPLES:-8}"
MAX_RESPONSE_TOKENS="${MAX_RESPONSE_TOKENS:-8192}"
PROMPT_MAX_LEN="${PROMPT_MAX_LEN:-4096}"
ROLLOUT_MAX_MODEL_LEN="${ROLLOUT_MAX_MODEL_LEN:-12288}"
ACTOR_FSDP_SIZE="${ACTOR_FSDP_SIZE:-1}"   # NO_SHARD: every GPU holds a full copy
SEED="${SEED:-42}"   # fsdp / dataloader seeds (verl default); data.seed is set by train_method.sh only, as in the paper runs

# wandb is used only when a key is configured.
if [ -n "${WANDB_API_KEY:-}" ]; then
    TRAINER_LOGGER='["console","wandb"]'
else
    TRAINER_LOGGER='["console"]'
fi

die() { echo "[rl] ERROR: $*" >&2; exit 1; }

# base_model <size> -> HF id of the base model.
base_model() {
    case "$1" in
        4b) echo "Qwen/Qwen3-VL-4B-Instruct" ;;
        8b) echo "Qwen/Qwen3-VL-8B-Instruct" ;;
        12b) echo "google/gemma-3-12b-it" ;;
        *) die "unknown size: $1 (expected 4b, 8b or 12b)" ;;
    esac
}

# run_grpo <reward_py> <train_parquet> <test_parquet> <policy> <ckpt_dir> <n_gpus> <rollout_gpu_util> <dataloader_workers> [extra hydra args...]
run_grpo() {
    local reward_py="$1" train="$2" test="$3" policy="$4" ckpt_dir="$5" n_gpus="$6" rollout_util="$7" workers="$8"
    shift 8
    python3 -u -m verl.trainer.main_ppo \
        algorithm.adv_estimator=grpo \
        algorithm.use_kl_in_reward=False \
        algorithm.norm_adv_by_std_in_grpo=True \
        data.train_files="${train}" \
        data.val_files="${test}" \
        data.train_batch_size="${TRAIN_BATCH_SIZE}" \
        data.max_prompt_length="${PROMPT_MAX_LEN}" \
        data.max_response_length="${MAX_RESPONSE_TOKENS}" \
        data.filter_overlong_prompts=False \
        data.filter_overlong_prompts_workers=0 \
        data.truncation=error \
        data.dataloader_num_workers="${workers}" \
        data.trust_remote_code=True \
        actor_rollout_ref.model.path="${policy}" \
        actor_rollout_ref.model.trust_remote_code=True \
        actor_rollout_ref.model.use_remove_padding=True \
        actor_rollout_ref.model.enable_gradient_checkpointing=True \
        actor_rollout_ref.actor.strategy=fsdp \
        actor_rollout_ref.actor.optim.lr="${LR}" \
        actor_rollout_ref.actor.data_loader_seed="${SEED}" \
        ++actor_rollout_ref.actor.use_torch_compile=False \
        ++actor_rollout_ref.ref.use_torch_compile=False \
        actor_rollout_ref.actor.freeze_vision_tower=True \
        actor_rollout_ref.actor.ppo_mini_batch_size="${ACTOR_PPO_MINI_BATCH_SIZE}" \
        actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu="${ACTOR_PPO_MICRO_BATCH_SIZE_PER_GPU}" \
        actor_rollout_ref.actor.ppo_epochs=2 \
        actor_rollout_ref.actor.use_kl_loss=True \
        actor_rollout_ref.actor.kl_loss_coef="${KL_COEF}" \
        actor_rollout_ref.actor.kl_loss_type=low_var_kl \
        actor_rollout_ref.actor.entropy_coeff="${ENTROPY_COEFF}" \
        actor_rollout_ref.actor.clip_ratio=0.2 \
        actor_rollout_ref.actor.fsdp_config.seed="${SEED}" \
        actor_rollout_ref.actor.fsdp_config.param_offload=False \
        actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
        actor_rollout_ref.actor.fsdp_config.model_dtype=bfloat16 \
        actor_rollout_ref.actor.fsdp_config.dtype=bfloat16 \
        actor_rollout_ref.actor.fsdp_config.fsdp_size="${ACTOR_FSDP_SIZE}" \
        actor_rollout_ref.rollout.name=vllm \
        actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
        actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu="${LOG_PROB_MICRO_BATCH_SIZE_PER_GPU}" \
        actor_rollout_ref.rollout.enforce_eager=False \
        actor_rollout_ref.rollout.max_num_batched_tokens=65536 \
        actor_rollout_ref.rollout.enable_prefix_caching=True \
        actor_rollout_ref.rollout.gpu_memory_utilization="${rollout_util}" \
        actor_rollout_ref.rollout.max_model_len="${ROLLOUT_MAX_MODEL_LEN}" \
        actor_rollout_ref.rollout.dtype=bfloat16 \
        actor_rollout_ref.rollout.n="${ROLLOUT_SAMPLES}" \
        actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu="${LOG_PROB_MICRO_BATCH_SIZE_PER_GPU}" \
        actor_rollout_ref.ref.fsdp_config.seed="${SEED}" \
        actor_rollout_ref.ref.fsdp_config.param_offload=False \
        actor_rollout_ref.ref.fsdp_config.model_dtype=bfloat16 \
        actor_rollout_ref.ref.fsdp_config.dtype=bfloat16 \
        actor_rollout_ref.ref.fsdp_config.fsdp_size="${ACTOR_FSDP_SIZE}" \
        reward.custom_reward_function.path="${REPO_DIR}/${reward_py}" \
        reward.custom_reward_function.name=compute_score \
        trainer.critic_warmup=0 \
        trainer.logger="${TRAINER_LOGGER}" \
        trainer.project_name="${WANDB_PROJECT:-physics_reasoning}" \
        trainer.experiment_name="$(basename "${ckpt_dir}")" \
        trainer.n_gpus_per_node="${n_gpus}" \
        trainer.nnodes=1 \
        trainer.save_freq=9999999 \
        trainer.max_actor_ckpt_to_keep=1 \
        trainer.test_freq=0 \
        trainer.val_before_train=False \
        trainer.resume_mode=auto \
        trainer.total_epochs="${TOTAL_EPOCHS}" \
        trainer.default_local_dir="${ckpt_dir}" \
        "$@"
}

# merge_final <ckpt_dir> <policy> <out_dir>
# Consolidates the final FSDP checkpoint into a Hugging Face model. verl's merger
# writes config + weights + tokenizer only, so the VL processor files are copied
# over from the policy (SFT) model.
merge_final() {
    local ckpt_dir="$1" policy="$2" out_dir="$3"
    local step
    step="$(cat "${ckpt_dir}/latest_checkpointed_iteration.txt" 2>/dev/null)" \
        || die "no checkpoint found in ${ckpt_dir}"
    python3 -m verl.model_merger merge \
        --backend fsdp \
        --local_dir "${ckpt_dir}/global_step_${step}/actor" \
        --target_dir "${out_dir}" \
        --trust-remote-code \
        --use_cpu_initialization
    local f
    for f in preprocessor_config.json video_preprocessor_config.json chat_template.json \
             chat_template.jinja merges.txt vocab.json tokenizer.json tokenizer_config.json \
             generation_config.json; do
        if [ ! -f "${out_dir}/${f}" ] && [ -f "${policy}/${f}" ]; then
            cp "${policy}/${f}" "${out_dir}/${f}"
        fi
    done
    echo "[rl] merged model: ${out_dir}"
}
