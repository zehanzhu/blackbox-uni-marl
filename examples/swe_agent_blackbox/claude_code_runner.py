"""Claude Code runner for the blackbox SWE-agent recipe."""

from __future__ import annotations

import json
import logging
import os
import shlex
import time

from uni_agent.trainer.framework.types import SessionHandle, SessionRuntime

logger = logging.getLogger(__name__)

DEFAULT_TOOL_IMAGE = "claude-code-tool:latest"
TOOL_TARGET = "/opt/claude-code"


def extract_task(raw_prompt) -> str:
    if isinstance(raw_prompt, str):
        return raw_prompt
    return next(
        (m["content"] for m in raw_prompt if isinstance(m, dict) and m.get("role") == "user"),
        str(raw_prompt),
    )


def _extract_issue_text(task: str) -> str:
    start = task.find("<issue_description>")
    end = task.find("</issue_description>")
    if start >= 0 and end > start:
        return task[start + len("<issue_description>"):end].strip()
    marker = "\nFollow these steps to resolve the issue:"
    if marker in task:
        return task.split(marker, 1)[0].strip()
    return task.strip()


def _decode_metadata_list(value) -> list[str]:
    if not value:
        return []
    if isinstance(value, list):
        return [str(item) for item in value]
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return [value]
        if isinstance(parsed, list):
            return [str(item) for item in parsed]
    return [str(value)]


def build_claude_task(raw_prompt, tools_kwargs: dict | None = None) -> str:
    tools_kwargs = tools_kwargs or {}
    task = extract_task(raw_prompt)
    metadata = ((tools_kwargs.get("reward") or {}).get("metadata") or {})
    issue = metadata.get("problem_statement") or _extract_issue_text(task)
    tests = _decode_metadata_list(metadata.get("FAIL_TO_PASS"))
    if not tests:
        tests = _decode_metadata_list(metadata.get("PASS_TO_PASS"))[:3]
    tests_block = "\n".join(f"- {test}" for test in tests) if tests else "- Run the closest relevant tests you identify."

    return (
        "You are fixing a SWE-bench task in /testbed.\n\n"
        "Issue:\n"
        f"{issue}\n\n"
        "Rules:\n"
        "- Edit source files only. Do not modify tests.\n"
        "- The development environment is already installed; do not install packages unless a test command proves it is necessary.\n"
        "- There is no submit tool in this environment. Do not try to submit.\n"
        "- Do not create extra edge-case test files after the relevant tests pass.\n"
        "- Do not run `pytest --collect-only`, `git log`, or any other command that does not directly validate the fix.\n"
        "- Do not analyze unrelated `is_separable` behavior.\n"
        "- Do not run additional ad-hoc verification after the listed relevant pytest command passes.\n"
        "- Do not commit.\n"
        "- After the minimal fix is applied and a relevant pytest command passes, print a one-line summary and exit immediately.\n\n"
        "Relevant tests to run after the fix:\n"
        f"{tests_block}\n"
    )


def build_claude_command(
    *,
    task: str,
    base_url: str,
    max_turns: int,
    model: str = "default",
    permission_mode: str = "bypassPermissions",
    conda_env: str | None = "testbed",
    disable_web_tools: bool = True,
    disable_slash_commands: bool = True,
) -> str:
    env = {
        "ANTHROPIC_BASE_URL": base_url,
        "ANTHROPIC_API_KEY": "not-needed",
        "ANTHROPIC_MODEL": model,
        "ANTHROPIC_DEFAULT_HAIKU_MODEL": model,
        "ANTHROPIC_DEFAULT_SONNET_MODEL": model,
        "ANTHROPIC_DEFAULT_OPUS_MODEL": model,
        "ANTHROPIC_SMALL_FAST_MODEL": model,
        "CLAUDE_CODE_DISABLE_BACKGROUND_TASKS": "1",
        "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1",
        "CLAUDE_CODE_FORK_SUBAGENT": "0",
        "CLAUDE_CODE_SUBAGENT_MODEL": model,
        "DISABLE_AUTOUPDATER": "1",
        "IS_SANDBOX": "1",
    }
    env_assignments = [f"{key}={shlex.quote(value)}" for key, value in env.items()]
    if conda_env:
        conda_prefix = f"/opt/miniconda3/envs/{conda_env}"
        env_assignments.extend(
            [
                f"CONDA_DEFAULT_ENV={shlex.quote(conda_env)}",
                f"CONDA_PREFIX={shlex.quote(conda_prefix)}",
                f"PATH={shlex.quote(conda_prefix + '/bin')}:/opt/miniconda3/bin:$PATH",
            ]
        )
    env_prefix = " ".join(env_assignments)
    argv = [
        "/opt/claude-code/bin/claude",
        "-p",
        task,
        "--model",
        model,
        "--max-turns",
        str(max_turns),
        "--permission-mode",
        permission_mode,
    ]
    if disable_slash_commands:
        argv.append("--disable-slash-commands")
    if disable_web_tools:
        argv.extend(["--disallowedTools", "Agent", "Task", "WebFetch", "WebSearch"])
    return (
        "unset HTTP_PROXY HTTPS_PROXY http_proxy https_proxy NO_PROXY no_proxy; "
        "cd /testbed; "
        f"{env_prefix} "
        + shlex.join(argv)
    )


async def _create_claude_sandbox(
    *,
    image: str,
    sidecar_image: str,
    gateway_url: str,
):
    from examples.swe_agent_blackbox.sandbox import YRSandbox, extract_upstream

    upstream = extract_upstream(gateway_url) if gateway_url else ""
    return await YRSandbox.create(
        image=image,
        sidecar_image=sidecar_image,
        sidecar_target=TOOL_TARGET,
        upstream=upstream,
    )


async def claude_code_runner(
    *,
    raw_prompt,
    session: SessionHandle,
    sample_index: int,
    session_runtime: SessionRuntime,
    tools_kwargs: dict | None = None,
    tool_image: str = DEFAULT_TOOL_IMAGE,
    run_timeout: int = 7200,
    **kwargs,
) -> None:
    from examples.swe_agent_blackbox.dataset import extract_image
    from examples.swe_agent_blackbox.mini_swe_agent_runner import SandboxEnvForReward
    from examples.swe_agent_blackbox.reward import build_reward_context, evaluate_in_env

    tools_kwargs = tools_kwargs or {}
    task = build_claude_task(raw_prompt, tools_kwargs)
    env_config = tools_kwargs.get("env", {})
    image = extract_image(env_config)
    if not image:
        raise ValueError(f"No Docker image found in tools_kwargs.env for sample {sample_index}")

    gateway_url = session.base_url
    if not gateway_url:
        raise ValueError(f"gateway_url is empty for sample {sample_index}")

    sandbox = await _create_claude_sandbox(
        image=image,
        sidecar_image=tool_image,
        gateway_url=gateway_url,
    )

    try:
        post_setup_cmd = env_config.get("post_setup_cmd", "")
        if post_setup_cmd:
            setup_result = await sandbox.run(post_setup_cmd, timeout=120)
            if setup_result.exit_code != 0:
                logger.warning("post_setup_cmd failed rc=%s: %.300s", setup_result.exit_code, setup_result.stdout + setup_result.stderr)

        from examples.swe_agent_blackbox.sandbox import rewrite_gateway_url

        claude_base_url = rewrite_gateway_url(gateway_url, strip_v1=True)
        max_turns = int(os.environ.get("SWE_AGENT_MAX_TURNS", "100"))
        agent_cmd = build_claude_command(
            task=task,
            base_url=claude_base_url,
            max_turns=max_turns,
        )

        started_at = time.perf_counter()
        result = await sandbox.run(agent_cmd, timeout=int(run_timeout))
        elapsed = time.perf_counter() - started_at
        logger.info("[sample %d] claude-code finished rc=%s elapsed=%.1fs", sample_index, result.exit_code, elapsed)
        if result.exit_code != 0:
            logger.warning(
                "[sample %d] claude-code failed stdout_tail=%r stderr_tail=%r",
                sample_index,
                (result.stdout or "")[-4000:],
                (result.stderr or "")[-4000:],
            )

        metadata, eval_timeout = build_reward_context(tools_kwargs)
        score, eval_result = await evaluate_in_env(SandboxEnvForReward(sandbox), metadata, eval_timeout)
        logger.info("[sample %d] reward done score=%s resolved=%s", sample_index, score, eval_result.get("resolved"))

        reward_info = {
            "reward_score": score,
            "claude_code_exit_code": result.exit_code,
            **eval_result,
        }
        await session_runtime.complete_session(session.session_id, reward_info=reward_info)
    finally:
        await sandbox.cleanup()
