"""Reward function for the multi-agent blackbox example."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


def _text(value: Any) -> str:
    return "" if value is None else str(value)


def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: Any,
    extra_info: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Return a rollout-level reward using verl's custom reward API.

    ``MultiAgentFramework`` injects the MAS ``final_result`` and
    ``agent_outputs`` into ``extra_info`` before RewardLoopWorker invokes this
    function. Production integrations should replace this with a task evaluator.
    """
    extra_info = dict(extra_info or {})
    final_result = extra_info.get("final_result", solution_str)
    agent_outputs = extra_info.get("agent_outputs", {})

    expected = ground_truth
    if isinstance(expected, Mapping):
        expected = (
            expected.get("target")
            or expected.get("expected")
            or expected.get("expected_text")
            or expected.get("answer")
            or expected.get("ground_truth")
        )
    # Support list/tuple ground truth (take the first candidate).
    if isinstance(expected, (list, tuple)):
        expected = expected[0] if expected else None
    # Support stringified lists like "['2,718']".
    if isinstance(expected, str):
        _stripped = expected.strip()
        if _stripped.startswith("[") and _stripped.endswith("]"):
            try:
                import ast

                _parsed = ast.literal_eval(_stripped)
                if isinstance(_parsed, (list, tuple)):
                    expected = _parsed[0] if _parsed else None
            except Exception:
                pass

    final_text = _text(final_result).strip()
    if expected is not None:
        expected_text = _text(expected).strip()
        score = 1.0 if expected_text and expected_text.lower() in final_text.lower() else 0.0
        reward_source = "expected_substring"
    else:
        score = 1.0 if final_text else 0.0
        reward_source = "non_empty_final_result"

    return {
        "score": score,
        "reward_extra_info": {
            "reward_source": reward_source,
            "data_source": data_source,
            "num_agent_outputs": len(agent_outputs),
        },
    }
