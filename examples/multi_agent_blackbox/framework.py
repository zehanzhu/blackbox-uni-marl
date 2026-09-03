"""Example-specific multi-agent framework with isolated Ray runner tasks."""

from __future__ import annotations

import asyncio
import logging
import time
from functools import partial
from typing import Any

import ray

from uni_agent.trainer.framework.framework import MultiAgentFramework

from examples.multi_agent_blackbox.remote_runner import remote_multi_agent_run

logger = logging.getLogger(__name__)


class RemoteMultiAgentFramework(MultiAgentFramework):
    """Execute each external MAS rollout in an independent Ray worker process."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._remote_tasks: dict[str, Any] = {}

    def _resolve_runner(self) -> tuple[str, dict[str, Any]]:
        runner = self.multi_agent_runner
        runner_kwargs: dict[str, Any] = {}
        while isinstance(runner, partial):
            runner_kwargs = {**dict(runner.keywords or {}), **runner_kwargs}
            runner = runner.func
        return f"{runner.__module__}.{runner.__qualname__}", runner_kwargs

    async def _execute_multi_agent_runner(
        self,
        *,
        raw_prompt,
        rollout,
        rollout_id: str,
        sample_index: int,
        runner_kwargs: dict[str, object] | None,
    ):
        runner_fqn, configured_kwargs = self._resolve_runner()
        resolved_kwargs = {**configured_kwargs, **(runner_kwargs or {})}
        remote_ref = remote_multi_agent_run.remote(
            runner_fqn=runner_fqn,
            raw_prompt=raw_prompt,
            rollout=rollout,
            sample_index=sample_index,
            role_policy_mapping=self.role_policy_mapping,
            runner_kwargs=resolved_kwargs,
        )
        self._remote_tasks[rollout_id] = remote_ref
        try:
            return await asyncio.wrap_future(remote_ref.future())
        except asyncio.CancelledError:
            try:
                ray.cancel(remote_ref, force=True)
            except Exception:
                logger.warning("failed to cancel remote MAS rollout %s", rollout_id, exc_info=True)
            raise
        finally:
            self._remote_tasks.pop(rollout_id, None)

    def shutdown(self, timeout: float = 30.0) -> None:
        deadline = time.monotonic() + timeout
        remote_refs = list(self._remote_tasks.values())
        cancellation_errors = []
        for remote_ref in remote_refs:
            try:
                ray.cancel(remote_ref, force=True)
            except Exception as exc:
                cancellation_errors.append(exc)
                logger.warning("failed to cancel remote MAS rollout during shutdown", exc_info=True)

        if cancellation_errors:
            raise RuntimeError("failed to cancel one or more remote MAS rollouts") from cancellation_errors[0]

        if remote_refs:
            remaining_timeout = max(0.0, deadline - time.monotonic())
            _, pending_refs = ray.wait(
                remote_refs,
                num_returns=len(remote_refs),
                timeout=remaining_timeout,
            )
            if pending_refs:
                raise TimeoutError(
                    f"timed out waiting for {len(pending_refs)} remote MAS rollout task(s) to stop"
                )

        super().shutdown(timeout=max(0.0, deadline - time.monotonic()))
        self._remote_tasks.clear()
