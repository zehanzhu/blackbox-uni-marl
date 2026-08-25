from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from uni_agent.trainer.framework.types import SessionHandle, Trajectory


class MalformedRequestError(ValueError):
    pass


class SessionPhase(str, Enum):
    ACTIVE = "ACTIVE"
    COMPLETED = "COMPLETED"
    FINALIZED = "FINALIZED"
    ABORTED = "ABORTED"


@dataclass
class RoleSessionInfo:
    role: str
    session_id: str  # Gateway session scoped to this rollout role.
    policy_name: str  # Trainable policy selected for this role.


@dataclass
class MultiAgentRolloutHandle:
    rollout_id: str
    base_url: str | None = None
    sessions: dict[str, RoleSessionInfo] = field(default_factory=dict)  # role -> role session info
    role_policy_mapping: dict[str, str] = field(default_factory=dict)  # role -> policy_name


@dataclass
class MultiAgentRolloutResult:
    rollout_id: str
    trajectories: list[Trajectory]
    sessions: dict[str, RoleSessionInfo] = field(default_factory=dict)
    reward_info: dict[str, Any] = field(default_factory=dict)  # rollout-level reward broadcast to trajectories


@dataclass
class MultiAgentRolloutState:
    handle: MultiAgentRolloutHandle
    metadata: dict[str, Any] = field(default_factory=dict)
    reward_info: dict[str, Any] = field(default_factory=dict)
    phase: SessionPhase = SessionPhase.ACTIVE
    completed: asyncio.Event = field(default_factory=asyncio.Event)
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    request_lock: asyncio.Lock = field(default_factory=asyncio.Lock)


@dataclass
class TrajectoryBuffer:
    prompt_ids: list[int]
    response_ids: list[int] = field(default_factory=list)
    response_mask: list[int] = field(default_factory=list)
    response_logprobs: list[float] = field(default_factory=list)


@dataclass
class GatewaySessionState:
    handle: SessionHandle
    metadata: dict[str, Any] = field(default_factory=dict)
    request_tools: list[dict[str, Any]] | None = None
    message_history: list[dict[str, Any]] = field(default_factory=list)
    image_data: list[Any] | None = None
    video_data: list[Any] | None = None
    active_trajectory: TrajectoryBuffer | None = None
    trajectories: list[Trajectory] = field(default_factory=list)
    reward_info: dict[str, Any] = field(default_factory=dict)
    completed: asyncio.Event = field(default_factory=asyncio.Event)
    phase: SessionPhase = SessionPhase.ACTIVE
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    request_lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    generation_lock: asyncio.Lock = field(default_factory=asyncio.Lock)
