#!/usr/bin/env bash
# Build a SWE blackbox sidecar tool image.
#
# Usage:
#   bash examples/swe_agent_blackbox/build_tool.sh
#   bash examples/swe_agent_blackbox/build_tool.sh --tool claude_code
#   bash examples/swe_agent_blackbox/build_tool.sh --pip-index https://pypi.tuna.tsinghua.edu.cn/simple/
#   bash examples/swe_agent_blackbox/build_tool.sh --npm-registry https://registry.npmmirror.com
#   bash examples/swe_agent_blackbox/build_tool.sh --tool-version latest
#   bash examples/swe_agent_blackbox/build_tool.sh --registry reg.antgroup-inc.cn/myrepo
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
TOOL_KIND="${TOOL_KIND:-mini_swe}"
IMAGE_TAG="${TOOL_TAG:-latest}"
TOOL_VERSION="${TOOL_VERSION:-latest}"

# Parse args
REGISTRY=""
PIP_INDEX_URL="${PIP_INDEX_URL:-}"
NPM_REGISTRY="${NPM_REGISTRY:-}"
while [[ $# -gt 0 ]]; do
    case "$1" in
        --tool) TOOL_KIND="$2"; shift 2 ;;
        --registry) REGISTRY="$2"; shift 2 ;;
        --pip-index) PIP_INDEX_URL="$2"; shift 2 ;;
        --npm-registry) NPM_REGISTRY="$2"; shift 2 ;;
        --tool-version) TOOL_VERSION="$2"; shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

BUILD_ARGS=()
DOCKERFILE="${SCRIPT_DIR}/Dockerfile.mini-swe-agent-tool"
if [[ "${TOOL_KIND}" == "claude" ]]; then
    TOOL_KIND="claude_code"
fi
if [[ "${TOOL_KIND}" == "claude_code" ]]; then
    IMAGE_NAME="${TOOL_IMAGE:-claude-code-tool}"
    DOCKERFILE="${SCRIPT_DIR}/Dockerfile.claude-code-tool"
    BUILD_ARGS+=(--build-arg "TOOL_VERSION=${TOOL_VERSION}")
    if [[ -n "${NPM_REGISTRY}" ]]; then
        BUILD_ARGS+=(--build-arg "NPM_REGISTRY=${NPM_REGISTRY}")
    fi
elif [[ "${TOOL_KIND}" == "mini_swe" ]]; then
    IMAGE_NAME="${TOOL_IMAGE:-mini-swe-agent-tool}"
    if [[ -n "${PIP_INDEX_URL}" ]]; then
        BUILD_ARGS+=(--build-arg PIP_INDEX_URL="${PIP_INDEX_URL}")
    fi
else
    echo "Unknown tool: ${TOOL_KIND}; expected mini_swe or claude_code"
    exit 1
fi

echo "==> Building ${TOOL_KIND} tool image: ${IMAGE_NAME}:${IMAGE_TAG}"
docker build \
    -f "${DOCKERFILE}" \
    -t "${IMAGE_NAME}:${IMAGE_TAG}" \
    "${BUILD_ARGS[@]}" \
    "${SCRIPT_DIR}/"

if [[ -n "${REGISTRY}" ]]; then
    FULL_TAG="${REGISTRY}/${IMAGE_NAME}:${IMAGE_TAG}"
    echo "==> Tagging and pushing: ${FULL_TAG}"
    docker tag "${IMAGE_NAME}:${IMAGE_TAG}" "${FULL_TAG}"
    docker push "${FULL_TAG}"
    echo "    Pushed."
fi

echo ""
echo "Tool image ready: ${IMAGE_NAME}:${IMAGE_TAG}"
if [[ -n "${REGISTRY}" ]]; then
    echo "  Remote sandbox: ${FULL_TAG}"
fi
