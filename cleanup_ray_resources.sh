#!/usr/bin/env bash
# cleanup_multi_agent_e2e.sh
#
# 清理 multi-agent-blackbox e2e 训练失败后残留的资源：
#   1. 卡住的 driver / TaskRunner 进程（main_multi_agents_ppo）
#   2. vLLM multiproc 子进程（Worker_TP*，不是 Ray actor，可能成孤儿）
#   3. 本项目残留的 Ray actor 进程（vLLMHttpServer / CheckpointEngineWorker /
#      GatewayActor / RewardLoopWorker / MultiAgentsTaskRunner，按 class 名过滤）
#   4. 残留的 Placement Group（把 GPU 释放回 Ray 可用池）
#
# 不会停止 Ray 集群本身（raylet / GCS / dashboard 保留）。
#
# 用法（在能访问 GPU 集群的机器上）：
#   bash cleanup_multi_agent_e2e.sh

set -uo pipefail

RAY_ADDRESS="${RAY_ADDRESS:-10.122.121.62:20001}"
PYTHON="${PYTHON:-/mnt/bn/chenghao1026/resouces/libs/zzh_env/bin/python3}"

echo "=== [1/4] 杀掉卡住的 driver / TaskRunner / vLLM 子进程 ==="
DRIVER_PIDS="$(ps -eo pid,args | grep "main_multi_agents_ppo" | grep -v grep | awk '{print $1}' | sort -un | tr '\n' ' ')"
VLLM_PIDS="$(ps -eo pid,args | grep "multiproc_executor" | grep -v grep | awk '{print $1}' | sort -un | tr '\n' ' ')"
ALL_PIDS="${DRIVER_PIDS} ${VLLM_PIDS}"
if [[ -n "${ALL_PIDS// /}" ]]; then
    echo "kill -9: ${ALL_PIDS}"
    kill -9 ${ALL_PIDS} 2>/dev/null || true
else
    echo "无残留 driver / vLLM 子进程"
fi

echo
echo "=== [2/4] 通过 Ray 状态 API 杀掉本项目残留 actor 进程（按 class 名） ==="
RAY_ADDRESS="${RAY_ADDRESS}" "${PYTHON}" - <<'PY'
import os
import ray

ray.init(address="auto", ignore_reinit_error=True)

TARGETS = {
    "MultiAgentsTaskRunner",
    "vLLMHttpServer",
    "CheckpointEngineWorker",
    "GatewayActor",
    "RewardLoopWorker",
    "TransferQueueController",
}

try:
    import ray.util.state as state
    actors = state.list_actors()
except Exception as exc:
    print(f"list_actors 不可用（{exc}），跳过 actor 清理")
    actors = []

killed = 0
for actor in actors:
    cls = actor.get("class_name", "")
    if cls not in TARGETS:
        continue
    pid = actor.get("pid")
    if not pid or pid <= 0:
        continue
    try:
        os.kill(pid, 9)
        killed += 1
        print(f"  killed actor {cls} pid={pid} id={actor.get('actor_id')}")
    except ProcessLookupError:
        pass
    except Exception as exc:
        print(f"  kill {cls} pid={pid} 失败: {exc}")
print(f"共杀掉 {killed} 个目标 actor 进程")
PY

echo
echo "=== [3/4] 等 3 秒让 Ray 感知进程退出 ==="
sleep 3

echo
echo "=== [4/4] 清理本项目残留 Placement Group（保留 Ray） ==="
RAY_ADDRESS="${RAY_ADDRESS}" "${PYTHON}" - <<'PY'
import ray
import ray.util

ray.init(address="auto", ignore_reinit_error=True)

def gpu_available():
    return ray.available_resources().get("GPU", 0)

print(f"清理前 GPU available = {gpu_available()}")

table = ray.util.placement_group_table()
removed = 0
skipped = 0
for pg_id, info in table.items():
    name = info.get("name", "")
    state = info.get("state", "")
    # 只清本项目风格的名字（policy_*/pool_* + verl_group / global_pool）
    if not (("verl_group" in name) or ("global_pool" in name) or name.startswith("policy_")):
        continue
    if state in ("REMOVED", "DEAD"):
        # 历史残留，不占资源；客户端也查不到句柄，直接跳过
        skipped += 1
        continue
    try:
        # get_placement_group 按名字查找；死 job 的 PG 查不到属正常
        pg = ray.util.get_placement_group(name)
        ray.util.remove_placement_group(pg)
        removed += 1
        print(f"  removed PG {pg_id} name={name!r} state={state}")
    except Exception as exc:
        print(f"  跳过 PG {pg_id} name={name!r} state={state}（{exc}）")
print(f"共移除 {removed} 个，跳过历史残留 {skipped} 个")
print(f"清理后 GPU available = {gpu_available()}")
if gpu_available() == ray.cluster_resources().get("GPU", 0):
    print("GPU 已全部释放回可用池 ✓")
else:
    print("GPU 仍有占用：请用 nvidia-smi 检查是否有 vllm/ray.worker 残留进程")
print("清理完成；Ray 集群保留。若仍有 GPU 被占，请检查是否还有 vllm/ray.worker 残留进程。")
PY
