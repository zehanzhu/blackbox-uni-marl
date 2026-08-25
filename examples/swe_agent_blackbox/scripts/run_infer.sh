#!/usr/bin/env bash
# Parallel inference for the blackbox SWE-agent recipe.
#
# Usage:
#   bash examples/swe_agent_blackbox/scripts/run_infer.sh

set -euo pipefail

# ── Model & data ─────────────────────────────────────────────────────────
MODEL_PATH="${MODEL_PATH:-$HOME/models/Qwen3.5-9B}"
DATA_PATH="${DATA_PATH:-$HOME/data/swe_agent/swe_bench_verified.parquet}"

# ── Inference parameters ─────────────────────────────────────────────────
MAX_SAMPLES="${MAX_SAMPLES:--1}"
PROMPT_LENGTH="${PROMPT_LENGTH:-4096}"
RESPONSE_LENGTH="${RESPONSE_LENGTH:-65536}"
TEMPERATURE="${TEMPERATURE:-1.0}"
TOP_P="${TOP_P:-1.0}"
N="${N:-8}"
ENGINE="${ENGINE:-vllm}"
TP="${TP:-4}"
N_GPUS_PER_NODE="${N_GPUS_PER_NODE:-8}"
GATEWAY_COUNT="${GATEWAY_COUNT:-1}"
MAX_CONCURRENT_SESSIONS="${MAX_CONCURRENT_SESSIONS:-2}"

# ── Agent parameters ─────────────────────────────────────────────────────
RUNNER="${RUNNER:-uniagent}"
AGENT_CONFIG_PATH="${AGENT_CONFIG_PATH:-examples/swe_agent_blackbox/config/agent_config.yaml}"
export SWE_AGENT_MAX_TURNS="${SWE_AGENT_MAX_TURNS:-100}"
export SWE_AGENT_EVAL_TIMEOUT="${SWE_AGENT_EVAL_TIMEOUT:-600}"
SWE_AGENT_TOOL_IMAGE="${SWE_AGENT_TOOL_IMAGE:-}"
SWE_AGENT_RUN_TIMEOUT="${SWE_AGENT_RUN_TIMEOUT:-7200}"

# ── Logging ──────────────────────────────────────────────────────────────
export VERL_LOGGING_LEVEL="${VERL_LOGGING_LEVEL:-INFO}"
export ROLLOUT_GPU_MEM_UTIL="${ROLLOUT_GPU_MEM_UTIL:-0.5}"

echo "=== SWE-Agent Blackbox Inference ==="
echo "Model: ${MODEL_PATH}"
echo "Data:  ${DATA_PATH}"
echo "Max samples: ${MAX_SAMPLES}"
echo "Engine: ${ENGINE} (TP=${TP})"
echo "Runner: ${RUNNER}"
echo "Gateway count: ${GATEWAY_COUNT}"
echo "Max concurrent sessions: ${MAX_CONCURRENT_SESSIONS}"
echo "====================================="

python examples/swe_agent_blackbox/parallel_infer.py \
    --model-path "${MODEL_PATH}" \
    --data-path "${DATA_PATH}" \
    --max-samples "${MAX_SAMPLES}" \
    --prompt-length "${PROMPT_LENGTH}" \
    --response-length "${RESPONSE_LENGTH}" \
    --temperature "${TEMPERATURE}" \
    --top-p "${TOP_P}" \
    --n "${N}" \
    --engine "${ENGINE}" \
    --tensor-parallel-size "${TP}" \
    --max-turns "${SWE_AGENT_MAX_TURNS}" \
    --runner "${RUNNER}" \
    --agent-config-path "${AGENT_CONFIG_PATH}" \
    --n-gpus-per-node "${N_GPUS_PER_NODE}" \
    --gateway-count "${GATEWAY_COUNT}" \
    --max-concurrent-sessions "${MAX_CONCURRENT_SESSIONS}" \
    --tool-image "${SWE_AGENT_TOOL_IMAGE}" \
    --run-timeout "${SWE_AGENT_RUN_TIMEOUT}"
