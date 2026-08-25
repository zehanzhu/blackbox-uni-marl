#!/usr/bin/env bash
# Megatron sync training for the blackbox SWE-agent recipe.
#
# Uses main_ppo_sync + Megatron backend with the same blackbox agent infrastructure
# (AgentFrameworkRolloutAdapter, subprocess_runner, SWEAgentFramework).
#
# Usage:
#   bash examples/swe_agent_blackbox/scripts/run_train_megatron_sync.sh
#
# All configurable via environment variables (see defaults below).

set -euo pipefail

# ── Model & data ─────────────────────────────────────────────────────────
MODEL_PATH="${MODEL_PATH:-$HOME/models/Qwen3.5-9B}"
TRAIN_DATA="${TRAIN_DATA:-$HOME/data/swe_agent/swe_rebench_filtered.parquet}"
VAL_DATA="${VAL_DATA:-$HOME/data/swe_agent/swe_bench_verified.parquet}"

# ── Hardware ─────────────────────────────────────────────────────────────
NNODES="${NNODES:-1}"
NGPUS_PER_NODE="${NGPUS_PER_NODE:-8}"

# ── Training parameters ─────────────────────────────────────────────────
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-128}"
PROMPT_LENGTH="${PROMPT_LENGTH:-4096}"
RESPONSE_LENGTH="${RESPONSE_LENGTH:-131072}"
ACTOR_LR="${ACTOR_LR:-1e-6}"
TOTAL_EPOCHS="${TOTAL_EPOCHS:-10}"
SAVE_FREQ="${SAVE_FREQ:-10}"
TEST_FREQ="${TEST_FREQ:-10}"
PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-16}"

# ── Rollout parameters ──────────────────────────────────────────────────
ENGINE="${ENGINE:-vllm}"
TP="${TP:-4}"
ROLLOUT_GPU_MEM_UTIL="${ROLLOUT_GPU_MEM_UTIL:-0.7}"
N="${N:-8}"
TEMPERATURE="${TEMPERATURE:-1.0}"

# ── Megatron parallelism ────────────────────────────────────────────────
TRAIN_TP="${TRAIN_TP:-8}"
TRAIN_PP="${TRAIN_PP:-1}"
TRAIN_CP="${TRAIN_CP:-1}"
OFFLOAD="${OFFLOAD:-true}"
USE_MBRIDGE="${USE_MBRIDGE:-true}"

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
RUNNER_ARGS=(
    "actor_rollout_ref.rollout.custom.agent_framework.agent_runner_fqn=${AGENT_RUNNER_FQN}"
)
if [[ "${RUNNER}" != "uniagent" ]]; then
    RUNNER_ARGS+=(
        "+actor_rollout_ref.rollout.custom.agent_framework.agent_runner_kwargs.tool_image=${SWE_AGENT_TOOL_IMAGE}"
        "+actor_rollout_ref.rollout.custom.agent_framework.agent_runner_kwargs.run_timeout=${SWE_AGENT_RUN_TIMEOUT}"
    )
fi

# ── Logging ──────────────────────────────────────────────────────────────
PROJECT_NAME="${PROJECT_NAME:-swe_agent_blackbox}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-swe_agent_$(date +%Y%m%d_%H%M)}"
VERL_LOGGING_LEVEL="${VERL_LOGGING_LEVEL:-INFO}"

export SWE_AGENT_MAX_TURNS="${MAX_TURNS}"
export SWE_AGENT_EVAL_TIMEOUT="${SWE_AGENT_EVAL_TIMEOUT:-600}"
export VERL_LOGGING_LEVEL

# ── Environment for NCCL ────────────────────────────────────────────────
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}"
export NCCL_SHM_DISABLE="${NCCL_SHM_DISABLE:-1}"

echo "=== SWE-Agent Blackbox Megatron Sync Training ==="
echo "Model:       ${MODEL_PATH}"
echo "Train data:  ${TRAIN_DATA}"
echo "Val data:    ${VAL_DATA}"
echo "Engine:      ${ENGINE} (gen_tp=${TP}, train_tp=${TRAIN_TP})"
echo "Runner:      ${RUNNER}"
echo "Batch size:  ${TRAIN_BATCH_SIZE}, N=${N}"
echo "Sequence:    prompt=${PROMPT_LENGTH}, response=${RESPONSE_LENGTH}"
echo "==============================================="

python3 -m verl.trainer.main_ppo_sync \
    --config-name=swe_agent_blackbox_megatron_sync \
    --config-path="$(pwd)/examples/swe_agent_blackbox/config" \
    hydra.searchpath=[pkg://verl.trainer.config] \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    data.train_files="['${TRAIN_DATA}']" \
    data.val_files="['${VAL_DATA}']" \
    data.train_batch_size=${TRAIN_BATCH_SIZE} \
    data.max_prompt_length=${PROMPT_LENGTH} \
    data.max_response_length=${RESPONSE_LENGTH} \
    actor_rollout_ref.rollout.name=${ENGINE} \
    actor_rollout_ref.rollout.tensor_model_parallel_size=${TP} \
    actor_rollout_ref.rollout.gpu_memory_utilization=${ROLLOUT_GPU_MEM_UTIL} \
    actor_rollout_ref.rollout.n=${N} \
    actor_rollout_ref.rollout.temperature=${TEMPERATURE} \
    actor_rollout_ref.rollout.prompt_length=${PROMPT_LENGTH} \
    actor_rollout_ref.rollout.response_length=${RESPONSE_LENGTH} \
    actor_rollout_ref.rollout.max_model_len=$((PROMPT_LENGTH + RESPONSE_LENGTH)) \
    actor_rollout_ref.rollout.multi_turn.max_assistant_turns=${MAX_TURNS} \
    actor_rollout_ref.actor.optim.lr=${ACTOR_LR} \
    actor_rollout_ref.actor.ppo_mini_batch_size=${PPO_MINI_BATCH_SIZE} \
    actor_rollout_ref.actor.megatron.tensor_model_parallel_size=${TRAIN_TP} \
    actor_rollout_ref.actor.megatron.pipeline_model_parallel_size=${TRAIN_PP} \
    actor_rollout_ref.actor.megatron.context_parallel_size=${TRAIN_CP} \
    actor_rollout_ref.actor.megatron.param_offload=${OFFLOAD} \
    actor_rollout_ref.actor.megatron.grad_offload=${OFFLOAD} \
    actor_rollout_ref.actor.megatron.use_mbridge=${USE_MBRIDGE} \
    actor_rollout_ref.rollout.nnodes=${NNODES} \
    actor_rollout_ref.rollout.n_gpus_per_node=${NGPUS_PER_NODE} \
    trainer.nnodes=${NNODES} \
    trainer.n_gpus_per_node=${NGPUS_PER_NODE} \
    trainer.total_epochs=${TOTAL_EPOCHS} \
    trainer.save_freq=${SAVE_FREQ} \
    trainer.test_freq=${TEST_FREQ} \
    trainer.project_name=${PROJECT_NAME} \
    trainer.experiment_name=${EXPERIMENT_NAME} \
    actor_rollout_ref.rollout.custom.agent_framework.agent_runner_kwargs.agent_config_path="${AGENT_CONFIG_PATH}" \
    actor_rollout_ref.rollout.custom.agent_framework.completion_timeout_seconds=${COMPLETION_TIMEOUT} \
    "${RUNNER_ARGS[@]}" \
    "$@"
