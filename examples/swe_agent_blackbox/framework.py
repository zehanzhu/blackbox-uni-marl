"""SWE-agent specific framework subclass.

Injects reward_info (from agent_runner's complete_session call)
into sample_fields["extra_info"] so the reward worker's
compute_score can access it via extra_info.

Overrides _run_session to execute agent_runner in a separate Ray worker
process, preventing blocking operations from stalling the event loop.
"""

from __future__ import annotations

import asyncio
import functools
import logging
from dataclasses import replace
from uuid import uuid4

import ray

from uni_agent.trainer.framework.framework import OpenAICompatibleAgentFramework

from examples.swe_agent_blackbox.subprocess_runner import remote_agent_run

logger = logging.getLogger(__name__)


class SWEAgentFramework(OpenAICompatibleAgentFramework):

    async def _score_trajectories(self, session_trajectories, sample_fields):
        if session_trajectories and session_trajectories[-1].reward_info:
            reward_info = session_trajectories[-1].reward_info
            extra_info = dict(sample_fields.get("extra_info") or {})
            sample_fields = {**sample_fields, "extra_info": {**extra_info, **reward_info}}
        return await super()._score_trajectories(session_trajectories, sample_fields)

    def _resolve_runner(self) -> tuple[str, dict]:
        """Extract FQN and pre-bound kwargs from self.agent_runner.

        self.agent_runner may be a functools.partial (from_config wraps it),
        so we unpack the original function and its keywords.
        """
        fn = self.agent_runner
        kwargs = {}
        if isinstance(fn, functools.partial):
            kwargs = dict(fn.keywords)
            fn = fn.func
        fqn = f"{fn.__module__}.{fn.__qualname__}"
        return fqn, kwargs

    async def _run_session(
        self,
        *,
        prompts,
        raw_prompt,
        sample_index: int,
        session_id: str | None = None,
        runner_kwargs: dict | None = None,
    ):
        """Run agent_runner in a Ray worker process instead of in-process."""
        session_id = session_id or f"session-{sample_index}-0-{uuid4().hex}"
        sample_fields = self._extract_sample_fields(prompts=prompts, sample_index=sample_index)
        session = await self.session_runtime.create_session(session_id)
        agent_runner_fqn, resolved_kwargs = self._resolve_runner()

        try:
            if runner_kwargs:
                resolved_kwargs = {**resolved_kwargs, **runner_kwargs}

            ref = remote_agent_run.remote(
                agent_runner_fqn=agent_runner_fqn,
                raw_prompt=raw_prompt,
                session_id=session_id,
                base_url=session.base_url,
                sample_index=sample_index,
                runner_kwargs=resolved_kwargs,
            )
            loop = asyncio.get_running_loop()
            reward_info = await loop.run_in_executor(None, ray.get, ref)

            await self.session_runtime.complete_session(
                session_id, reward_info=reward_info,
            )
            session_trajectories = await self.session_runtime.finalize_session(session_id)

        except Exception as e:
            logger.error("_run_session failed: session=%s, sample=%d, runner=%s: %s",
                         session_id, sample_index, agent_runner_fqn, e, exc_info=True)
            await self.session_runtime.abort_session(session_id)
            raise

        if not self.reward_loop_worker_handles or not session_trajectories:
            return session_trajectories, sample_fields

        annotations = await self._score_trajectories(session_trajectories, sample_fields)
        scored_trajectories = []
        for traj, (score, extra) in zip(session_trajectories, annotations, strict=True):
            scored_trajectories.append(
                replace(
                    traj,
                    reward_score=score,
                    extra_fields={**traj.extra_fields, "reward_extra_info": extra},
                )
            )
        return scored_trajectories, sample_fields
