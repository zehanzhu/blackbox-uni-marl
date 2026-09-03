"""Ray remote entrypoint for external multi-agent runners.

The remote task intentionally receives only serializable rollout metadata. The
live Gateway runtime remains owned by ``MultiAgentFramework`` in the trainer
actor; this module only loads and executes the configured runner.
"""

from __future__ import annotations

import asyncio
from typing import Any

import ray

from verl.utils.import_utils import load_class_from_fqn


class _RemoteSessionRuntime:
    """Compatibility stub for runners that report reward through a runtime.

    It never contains Gateway actor handles. The parent framework remains the
    owner of the real rollout lifecycle.
    """

    def __init__(self) -> None:
        self.reward_info: dict[str, Any] | None = None

    async def complete_multi_agent_rollout(
        self,
        rollout_id: str,
        reward_info: dict[str, Any] | None = None,
    ) -> None:
        del rollout_id
        self.reward_info = dict(reward_info or {})

    async def complete_session(
        self,
        session_id: str,
        reward_info: dict[str, Any] | None = None,
    ) -> None:
        del session_id
        self.reward_info = dict(reward_info or {})


async def run_multi_agent_runner(
    *,
    runner_fqn: str,
    raw_prompt: Any,
    rollout: Any,
    sample_index: int,
    role_policy_mapping: dict[str, str],
    runner_kwargs: dict[str, Any] | None = None,
) -> Any:
    """Load and execute one configured multi-agent runner in a worker process."""
    runner = load_class_from_fqn(runner_fqn, description="multi-agent runner")
    runtime = _RemoteSessionRuntime()
    result = await runner(
        raw_prompt=raw_prompt,
        rollout=rollout,
        sample_index=sample_index,
        session_runtime=runtime,
        role_policy_mapping=role_policy_mapping,
        **dict(runner_kwargs or {}),
    )
    if result is None and runtime.reward_info is not None:
        return {"reward_info": runtime.reward_info}
    if isinstance(result, dict) and "reward_info" not in result and runtime.reward_info is not None:
        return {**result, "reward_info": runtime.reward_info}
    return result


def _remote_task(function):
    """Create a Ray task while remaining importable under lightweight test stubs."""
    try:
        return ray.remote(num_cpus=0)(function)
    except TypeError:
        return ray.remote(function)


@_remote_task
def remote_multi_agent_run(
    *,
    runner_fqn: str,
    raw_prompt: Any,
    rollout: Any,
    sample_index: int,
    role_policy_mapping: dict[str, str],
    runner_kwargs: dict[str, Any] | None = None,
) -> Any:
    """Execute one MAS rollout runner in an independent Ray worker process."""
    return asyncio.run(
        run_multi_agent_runner(
            runner_fqn=runner_fqn,
            raw_prompt=raw_prompt,
            rollout=rollout,
            sample_index=sample_index,
            role_policy_mapping=role_policy_mapping,
            runner_kwargs=runner_kwargs,
        )
    )
