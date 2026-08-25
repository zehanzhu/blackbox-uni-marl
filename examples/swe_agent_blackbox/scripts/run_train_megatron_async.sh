#!/usr/bin/env bash
# Megatron + TQ fully-async training for the blackbox SWE-agent recipe.
#
# Uses FullyAsyncAgentFrameworkRolloutAdapter + SWEAgentFramework with Megatron backend.
# Data flows through TransferQueue (zero-copy) with ReplayBuffer flow control.
#
# Usage:
#   bash examples/swe_agent_blackbox/scripts/run_train_megatron_async.sh
#
# All configurable via environment variables (see defaults below).

set -euo pipefail

# ── Model & data ─────────────────────────────────────────────────────────
MODEL_PATH="${MODEL_PATH:-${HOME}/models/Qwen3.5-9B}"
TRAIN_DATA="${TRAIN_DATA:-${HOME}/data/swe_agent/swe_rebench_filtered.parquet}"
VAL_DATA="${VAL_DATA:-${HOME}/data/swe_agent/swe_bench_verified.parquet}"
RUNTIME_ENV="${RUNTIME_ENV:-}"

# ── Hardware ─────────────────────────────────────────────────────────────
NNODES_TRAIN="${NNODES_TRAIN:-1}"
NNODES_ROLLOUT="${NNODES_ROLLOUT:-1}"
NGPUS_PER_NODE="${NGPUS_PER_NODE:-8}"

# ── Algorithm ────────────────────────────────────────────────────────────
CLIP_RATIO_LOW="${CLIP_RATIO_LOW:-0.2}"
CLIP_RATIO_HIGH="${CLIP_RATIO_HIGH:-0.28}"
ACTOR_LR="${ACTOR_LR:-1e-6}"

# ── Sequence lengths ─────────────────────────────────────────────────────
PROMPT_LENGTH="${PROMPT_LENGTH:-4096}"
RESPONSE_LENGTH="${RESPONSE_LENGTH:-131072}"
MAX_MODEL_LEN=$((PROMPT_LENGTH + RESPONSE_LENGTH))

# ── Rollout parameters ───────────────────────────────────────────────────
ENGINE="${ENGINE:-vllm}"
GEN_TP="${GEN_TP:-2}"
N="${N:-8}"
TEMPERATURE="${TEMPERATURE:-1.0}"
ROLLOUT_GPU_MEM_UTIL="${ROLLOUT_GPU_MEM_UTIL:-0.7}"

# ── Megatron training parallelism ────────────────────────────────────────
TRAIN_TP="${TRAIN_TP:-8}"
TRAIN_PP="${TRAIN_PP:-1}"
TRAIN_CP="${TRAIN_CP:-1}"
OFFLOAD="${OFFLOAD:-True}"
OPTIMIZER_OFFLOAD_FRACTION="${OFFLOAD_FRACTION:-1.0}"
USE_MBRIDGE="${USE_MBRIDGE:-True}"
PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-16}"

# ── Agent parameters ─────────────────────────────────────────────────────
RUNNER="${RUNNER:-mini_swe}"
MAX_TURNS="${MAX_TURNS:-100}"
AGENT_CONFIG_PATH="${AGENT_CONFIG_PATH:-examples/swe_agent_blackbox/config/agent_config.yaml}"
COMPLETION_TIMEOUT="${COMPLETION_TIMEOUT:-600}"
if [[ "${RUNNER}" == "claude_code" ]]; then
    AGENT_RUNNER_FQN="examples.swe_agent_blackbox.claude_code_runner.claude_code_runner"
    SWE_AGENT_TOOL_IMAGE="${SWE_AGENT_TOOL_IMAGE:-claude-code-tool:latest}"
elif [[ "${RUNNER}" == "mini_swe" ]]; then
    AGENT_RUNNER_FQN="examples.swe_agent_blackbox.mini_swe_agent_runner.mini_swe_agent_runner"
    SWE_AGENT_TOOL_IMAGE="${SWE_AGENT_TOOL_IMAGE:-swr.cn-east-3.myhuaweicloud.com/openyuanrong/mini-swe-agent-tool:latest}"
elif [[ "${RUNNER}" == "uniagent" ]]; then
    AGENT_RUNNER_FQN="examples.swe_agent_blackbox.agent_runner.swe_agent_runner"
    SWE_AGENT_TOOL_IMAGE=""
else
    echo "Unknown RUNNER=${RUNNER}; expected mini_swe, claude_code, or uniagent" >&2
    exit 1
fi
SWE_AGENT_RUN_TIMEOUT="${SWE_AGENT_RUN_TIMEOUT:-7200}"
CONDA_ENV="${CONDA_ENV:-testbed}"
RUNNER_ARGS=(
    "actor_rollout_ref.rollout.custom.agent_framework.agent_runner_fqn=${AGENT_RUNNER_FQN}"
)
if [[ "${RUNNER}" != "uniagent" ]]; then
    RUNNER_ARGS+=(
        "+actor_rollout_ref.rollout.custom.agent_framework.agent_runner_kwargs.tool_image=${SWE_AGENT_TOOL_IMAGE}"
        "+actor_rollout_ref.rollout.custom.agent_framework.agent_runner_kwargs.run_timeout=${SWE_AGENT_RUN_TIMEOUT}"
        "+actor_rollout_ref.rollout.custom.agent_framework.agent_runner_kwargs.conda_env=${CONDA_ENV}"
    )
fi

# ── OpenYuanRong (YR remote sandbox) ─────────────────────────────────────
OPENYUANRONG_SERVER_ADDRESS="${OPENYUANRONG_SERVER_ADDRESS:-}"
OPENYUANRONG_TOKEN="${OPENYUANRONG_TOKEN:-}"
OPENYUANRONG_TUNNEL_SSL_VERIFY="${OPENYUANRONG_TUNNEL_SSL_VERIFY:-0}"

# ── Async training ───────────────────────────────────────────────────────
TOTAL_ROLLOUT_STEPS="${TOTAL_ROLLOUT_STEPS:-100000}"
STALENESS_THRESHOLD="${STALENESS_THRESHOLD:-1.0}"
TRIGGER_SYNC_STEP="${TRIGGER_SYNC_STEP:-4}"
PARTIAL_ROLLOUT="${PARTIAL_ROLLOUT:-True}"

# ── Logging & checkpointing ──────────────────────────────────────────────
PROJECT_NAME="${PROJECT_NAME:-swe_agent_blackbox}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-swe_agent_$(date +%Y%m%d_%H%M)}"
SAVE_FREQ="${SAVE_FREQ:-10}"
TEST_FREQ="${TEST_FREQ:-10}"
CKPTS_DIR="${CKPTS_DIR:-checkpoints/${PROJECT_NAME}/${EXPERIMENT_NAME}}"

export SWE_AGENT_MAX_TURNS="${MAX_TURNS}"
export SWE_AGENT_EVAL_TIMEOUT="${SWE_AGENT_EVAL_TIMEOUT:-600}"
export OPENYUANRONG_SERVER_ADDRESS
export OPENYUANRONG_TOKEN
export OPENYUANRONG_TUNNEL_SSL_VERIFY

echo "=== SWE-Agent Blackbox Megatron Async Training ==="
echo "Model:       ${MODEL_PATH}"
echo "Train data:  ${TRAIN_DATA}"
echo "Val data:    ${VAL_DATA}"
echo "Engine:      ${ENGINE} (gen_tp=${GEN_TP}, train_tp=${TRAIN_TP})"
echo "Runner:      ${RUNNER}"
echo "Batch:       n=${N}, mini_bsz=${PPO_MINI_BATCH_SIZE}"
echo "Sequence:    prompt=${PROMPT_LENGTH}, response=${RESPONSE_LENGTH}"
echo "Nodes:       train=${NNODES_TRAIN}, rollout=${NNODES_ROLLOUT}"
echo "==================================================="

# ── Compute derived parameters ───────────────────────────────────────────
ACTOR_PPO_MAX_TOKEN_LEN=$(( (PROMPT_LENGTH + RESPONSE_LENGTH) / TRAIN_CP ))
INFER_PPO_MAX_TOKEN_LEN=$(( (PROMPT_LENGTH + RESPONSE_LENGTH) / TRAIN_CP ))

RUNTIME_ENV_ARGS=()
if [ -n "${RUNTIME_ENV}" ]; then
    RUNTIME_ENV_ARGS=(--runtime-env "${RUNTIME_ENV}")
fi

# ── Ensure Ray is running ────────────────────────────────────────────────
TOTAL_GPUS=$(( (NNODES_TRAIN + NNODES_ROLLOUT) * NGPUS_PER_NODE ))
if ! ray status &>/dev/null; then
    echo "Starting Ray cluster (${TOTAL_GPUS} GPUs)..."
    ray start --head --num-gpus="${TOTAL_GPUS}" --disable-usage-stats
else
    echo "Ray cluster already running."
fi

# ── Launch ────────────────────────────────────────────────────────────────
WORKING_DIR="${WORKING_DIR:-$(pwd)}"

ray job submit --no-wait --working-dir="${WORKING_DIR}" "${RUNTIME_ENV_ARGS[@]}" \
    -- python3 -m verl.experimental.fully_async_policy.fully_async_main \
    --config-name=swe_agent_blackbox_megatron_async \
    --config-path="$(pwd)/examples/swe_agent_blackbox/config" \
    hydra.searchpath=[pkg://verl.trainer.config] \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    data.train_files="['${TRAIN_DATA}']" \
    data.val_files="['${VAL_DATA}']" \
    data.max_prompt_length=${PROMPT_LENGTH} \
    data.max_response_length=${RESPONSE_LENGTH} \
    actor_rollout_ref.rollout.n=${N} \
    actor_rollout_ref.rollout.name=${ENGINE} \
    actor_rollout_ref.rollout.prompt_length=${PROMPT_LENGTH} \
    actor_rollout_ref.rollout.response_length=${RESPONSE_LENGTH} \
    actor_rollout_ref.rollout.max_model_len=${MAX_MODEL_LEN} \
    actor_rollout_ref.rollout.max_num_batched_tokens=${MAX_MODEL_LEN} \
    actor_rollout_ref.rollout.temperature=${TEMPERATURE} \
    actor_rollout_ref.rollout.tensor_model_parallel_size=${GEN_TP} \
    actor_rollout_ref.rollout.gpu_memory_utilization=${ROLLOUT_GPU_MEM_UTIL} \
    actor_rollout_ref.rollout.multi_turn.max_assistant_turns=${MAX_TURNS} \
    actor_rollout_ref.rollout.custom.agent_framework.completion_timeout_seconds=${COMPLETION_TIMEOUT} \
    actor_rollout_ref.rollout.custom.agent_framework.agent_runner_kwargs.agent_config_path="${AGENT_CONFIG_PATH}" \
    "${RUNNER_ARGS[@]}" \
    actor_rollout_ref.actor.clip_ratio_low=${CLIP_RATIO_LOW} \
    actor_rollout_ref.actor.clip_ratio_high=${CLIP_RATIO_HIGH} \
    actor_rollout_ref.actor.ppo_mini_batch_size=${PPO_MINI_BATCH_SIZE} \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${ACTOR_PPO_MAX_TOKEN_LEN} \
    actor_rollout_ref.actor.optim.lr=${ACTOR_LR} \
    actor_rollout_ref.actor.optim.lr_decay_steps=${TOTAL_ROLLOUT_STEPS} \
    +actor_rollout_ref.actor.optim.override_optimizer_config.optimizer_offload_fraction=${OPTIMIZER_OFFLOAD_FRACTION} \
    +actor_rollout_ref.actor.optim.override_optimizer_config.overlap_cpu_optimizer_d2h_h2d=True \
    +actor_rollout_ref.actor.optim.override_optimizer_config.use_precision_aware_optimizer=True \
    +actor_rollout_ref.actor.optim.override_optimizer_config.optimizer_cpu_offload=True \
    actor_rollout_ref.actor.megatron.param_offload=${OFFLOAD} \
    actor_rollout_ref.actor.megatron.grad_offload=${OFFLOAD} \
    actor_rollout_ref.actor.megatron.optimizer_offload=${OFFLOAD} \
    actor_rollout_ref.actor.megatron.tensor_model_parallel_size=${TRAIN_TP} \
    actor_rollout_ref.actor.megatron.pipeline_model_parallel_size=${TRAIN_PP} \
    actor_rollout_ref.actor.megatron.context_parallel_size=${TRAIN_CP} \
    actor_rollout_ref.actor.megatron.use_mbridge=${USE_MBRIDGE} \
    actor_rollout_ref.ref.megatron.param_offload=${OFFLOAD} \
    actor_rollout_ref.ref.megatron.tensor_model_parallel_size=${TRAIN_TP} \
    actor_rollout_ref.ref.megatron.pipeline_model_parallel_size=${TRAIN_PP} \
    actor_rollout_ref.ref.megatron.context_parallel_size=${TRAIN_CP} \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${INFER_PPO_MAX_TOKEN_LEN} \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=${INFER_PPO_MAX_TOKEN_LEN} \
    trainer.project_name="${PROJECT_NAME}" \
    trainer.experiment_name="${EXPERIMENT_NAME}" \
    trainer.save_freq=${SAVE_FREQ} \
    trainer.test_freq=${TEST_FREQ} \
    trainer.default_local_dir="${CKPTS_DIR}" \
    trainer.nnodes=${NNODES_TRAIN} \
    trainer.n_gpus_per_node=${NGPUS_PER_NODE} \
    rollout.nnodes=${NNODES_ROLLOUT} \
    rollout.n_gpus_per_node=${NGPUS_PER_NODE} \
    rollout.total_rollout_steps=${TOTAL_ROLLOUT_STEPS} \
    async_training.staleness_threshold=${STALENESS_THRESHOLD} \
    async_training.trigger_parameter_sync_step=${TRIGGER_SYNC_STEP} \
    async_training.partial_rollout=${PARTIAL_ROLLOUT} \
    "$@"
