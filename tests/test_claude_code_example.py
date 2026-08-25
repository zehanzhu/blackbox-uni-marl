import importlib.util
import asyncio
import sys
import types
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def _install_package_stub(name: str, relative_path: str):
    module = sys.modules.get(name)
    if module is None:
        module = types.ModuleType(name)
        sys.modules[name] = module
    module.__path__ = [str(REPO_ROOT / relative_path)]
    return module


def _load_module(name: str, relative_path: str):
    spec = importlib.util.spec_from_file_location(name, REPO_ROOT / relative_path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


_install_package_stub("uni_agent", "uni_agent")
_install_package_stub("uni_agent.trainer", "uni_agent/trainer")
_install_package_stub("uni_agent.trainer.framework", "uni_agent/trainer/framework")
_install_package_stub("uni_agent.trainer.gateway", "uni_agent/trainer/gateway")
_load_module("uni_agent.trainer.framework.types", "uni_agent/trainer/framework/types.py")
_load_module("uni_agent.trainer.gateway.types", "uni_agent/trainer/gateway/types.py")


def test_anthropic_payload_to_openai_basic():
    module = _load_module("anthropic_compat_test", "uni_agent/trainer/gateway/anthropic_compat.py")

    converted = module.anthropic_payload_to_openai(
        {
            "model": "default",
            "system": "Be concise.",
            "max_tokens": 16,
            "messages": [{"role": "user", "content": "hello"}],
        }
    )
    assert converted["messages"] == [
        {"role": "system", "content": "Be concise."},
        {"role": "user", "content": "hello"},
    ]
    assert converted["model"] == "default"
    assert converted["max_tokens"] == 16


def test_anthropic_payload_accepts_system_role_message():
    module = _load_module("anthropic_compat_test", "uni_agent/trainer/gateway/anthropic_compat.py")

    converted = module.anthropic_payload_to_openai(
        {
            "messages": [
                {"role": "system", "content": "Use tools."},
                {"role": "user", "content": "hello"},
            ],
        }
    )
    assert converted["messages"] == [
        {"role": "system", "content": "Use tools."},
        {"role": "user", "content": "hello"},
    ]


def test_anthropic_payload_strips_claude_code_billing_header():
    module = _load_module("anthropic_compat_test", "uni_agent/trainer/gateway/anthropic_compat.py")

    converted = module.anthropic_payload_to_openai(
        {
            "system": "x-anthropic-billing-header: cch=abc123\nUse tools.",
            "messages": [
                {
                    "role": "system",
                    "content": [{"type": "text", "text": "x-anthropic-billing-header: cch=def456\nBe concise."}],
                },
                {"role": "user", "content": "hello"},
            ],
        }
    )
    assert converted["messages"] == [
        {"role": "system", "content": "Use tools.\nBe concise."},
        {"role": "user", "content": "hello"},
    ]


def test_anthropic_payload_moves_all_system_content_to_front():
    module = _load_module("anthropic_compat_test", "uni_agent/trainer/gateway/anthropic_compat.py")

    converted = module.anthropic_payload_to_openai(
        {
            "system": "Top system.",
            "messages": [
                {"role": "user", "content": "hello"},
                {"role": "system", "content": [{"type": "text", "text": "Late system."}]},
                {"role": "assistant", "content": "ok"},
            ],
        }
    )
    assert converted["messages"] == [
        {"role": "system", "content": "Top system.\nLate system."},
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "ok"},
    ]


def test_anthropic_tools_are_normalized_for_qwen_tool_parser():
    module = _load_module("anthropic_compat_test", "uni_agent/trainer/gateway/anthropic_compat.py")

    converted = module.anthropic_payload_to_openai(
        {
            "messages": [{"role": "user", "content": "hello"}],
            "tools": [
                {
                    "name": "TaskUpdate",
                    "description": "Update task state",
                    "input_schema": {
                        "type": "object",
                        "properties": {
                            "status": {
                                "description": "New status",
                                "anyOf": [
                                    {"type": "string", "enum": ["pending", "completed"]},
                                    {"type": "string", "const": "deleted"},
                                ],
                            },
                            "metadata": {
                                "description": "Free-form metadata",
                                "type": "object",
                                "additionalProperties": {},
                            },
                        },
                        "required": ["status"],
                    },
                }
            ],
        }
    )
    tool = converted["tools"][0]
    assert tool == {
        "type": "function",
        "function": {
            "name": "TaskUpdate",
            "description": "Update task state",
            "parameters": {
                "type": "object",
                "properties": {
                    "status": {
                        "type": "string",
                        "description": "New status",
                        "enum": ["pending", "completed", "deleted"],
                    },
                    "metadata": {
                        "type": "object",
                        "description": "Free-form metadata",
                    },
                },
                "required": ["status"],
            },
        },
    }


def test_openai_completion_to_anthropic_tool_use():
    module = _load_module("anthropic_compat_test", "uni_agent/trainer/gateway/anthropic_compat.py")

    result = module.openai_completion_to_anthropic_message(
        {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [
                            {
                                "id": "call_1",
                                "type": "function",
                                "function": {"name": "edit", "arguments": '{"path": "a.py"}'},
                            }
                        ],
                    },
                    "finish_reason": "tool_calls",
                }
            ],
            "usage": {"prompt_tokens": 5, "completion_tokens": 7},
        },
        request_model="default",
    )
    assert result["stop_reason"] == "tool_use"
    assert result["content"] == [{"type": "tool_use", "id": "call_1", "name": "edit", "input": {"path": "a.py"}}]
    assert result["usage"] == {"input_tokens": 5, "output_tokens": 7}


def test_anthropic_stream_events_for_text():
    module = _load_module("anthropic_compat_test", "uni_agent/trainer/gateway/anthropic_compat.py")

    message = {
        "id": "msg_1",
        "type": "message",
        "role": "assistant",
        "model": "default",
        "content": [{"type": "text", "text": "hello"}],
        "stop_reason": "end_turn",
        "stop_sequence": None,
        "usage": {"input_tokens": 3, "output_tokens": 2},
    }
    events = module.anthropic_stream_events(message)
    assert [event["type"] for event in events] == [
        "message_start",
        "content_block_start",
        "content_block_delta",
        "content_block_stop",
        "message_delta",
        "message_stop",
    ]
    assert events[0]["message"]["content"] == []
    assert events[2]["delta"] == {"type": "text_delta", "text": "hello"}
    encoded = module.encode_anthropic_sse_event(events[2])
    assert encoded.startswith("event: content_block_delta\n")
    assert encoded.endswith("\n\n")


def test_anthropic_stream_events_for_tool_use():
    module = _load_module("anthropic_compat_test", "uni_agent/trainer/gateway/anthropic_compat.py")

    message = {
        "id": "msg_1",
        "type": "message",
        "role": "assistant",
        "model": "default",
        "content": [{"type": "tool_use", "id": "call_1", "name": "edit", "input": {"path": "a.py"}}],
        "stop_reason": "tool_use",
        "stop_sequence": None,
        "usage": {"input_tokens": 3, "output_tokens": 2},
    }
    events = module.anthropic_stream_events(message)
    assert events[1]["content_block"] == {"type": "tool_use", "id": "call_1", "name": "edit", "input": {}}
    assert events[2]["delta"]["type"] == "input_json_delta"
    assert events[2]["delta"]["partial_json"] == '{"path": "a.py"}'


def test_claude_command_uses_anthropic_base_url_and_no_proxy():
    module = _load_module("claude_code_runner_test", "examples/swe_agent_blackbox/claude_code_runner.py")

    cmd = module.build_claude_command(
        task="fix bug",
        base_url="http://127.0.0.1:38197/sessions/s",
        max_turns=3,
    )
    assert "ANTHROPIC_BASE_URL=http://127.0.0.1:38197/sessions/s" in cmd
    assert "ANTHROPIC_API_KEY=not-needed" in cmd
    assert "ANTHROPIC_MODEL=default" in cmd
    assert "ANTHROPIC_DEFAULT_HAIKU_MODEL=default" in cmd
    assert "ANTHROPIC_DEFAULT_SONNET_MODEL=default" in cmd
    assert "ANTHROPIC_DEFAULT_OPUS_MODEL=default" in cmd
    assert "ANTHROPIC_SMALL_FAST_MODEL=default" in cmd
    assert "CLAUDE_CODE_DISABLE_BACKGROUND_TASKS=1" in cmd
    assert "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC=1" in cmd
    assert "CLAUDE_CODE_FORK_SUBAGENT=0" in cmd
    assert "CLAUDE_CODE_SUBAGENT_MODEL=default" in cmd
    assert "unset HTTP_PROXY HTTPS_PROXY http_proxy https_proxy NO_PROXY no_proxy" in cmd
    assert "cd /testbed" in cmd
    assert "CONDA_DEFAULT_ENV=testbed" in cmd
    assert "CONDA_PREFIX=/opt/miniconda3/envs/testbed" in cmd
    assert "PATH=/opt/miniconda3/envs/testbed/bin:/opt/miniconda3/bin:$PATH" in cmd
    assert "/opt/claude-code/bin/claude -p 'fix bug'" in cmd
    assert "--max-turns 3" in cmd
    assert "--permission-mode bypassPermissions" in cmd
    assert "--disable-slash-commands" in cmd
    assert "--disallowedTools Agent Task WebFetch WebSearch" in cmd


def test_claude_task_rewrites_swe_prompt_for_claude_code():
    module = _load_module("claude_code_runner_test", "examples/swe_agent_blackbox/claude_code_runner.py")

    prompt = (
        "<issue_description>\n"
        "Fix nested separability.\n"
        "</issue_description>\n\n"
        "Follow these steps to resolve the issue:\n"
        "Create additional test cases in /testbed/edge_case_tests.py.\n"
        "Submit your solution using the submit tool."
    )
    task = module.build_claude_task(
        [{"role": "user", "content": prompt}],
        {
            "reward": {
                "metadata": {
                    "FAIL_TO_PASS": '["astropy/modeling/tests/test_separable.py::test_case"]',
                }
            }
        },
    )
    assert "Fix nested separability." in task
    assert "astropy/modeling/tests/test_separable.py::test_case" in task
    assert "There is no submit tool" in task
    assert "print a one-line summary and exit immediately" in task
    assert "Do not run additional ad-hoc verification" in task
    assert "Do not run `pytest --collect-only`, `git log`, or any other command" in task
    assert "Do not analyze unrelated `is_separable` behavior" in task
    assert "Create additional test cases" not in task


def test_claude_runner_uses_explicit_runner_kwargs(monkeypatch):
    _install_package_stub("examples", "examples")
    _install_package_stub("examples.swe_agent_blackbox", "examples/swe_agent_blackbox")
    module = _load_module("claude_code_runner_runtime_test", "examples/swe_agent_blackbox/claude_code_runner.py")

    calls = {}

    class _FakeSandbox:
        async def run(self, cmd, timeout):
            calls.setdefault("runs", []).append({"cmd": cmd, "timeout": timeout})
            return types.SimpleNamespace(exit_code=0, stdout="done", stderr="")

        async def cleanup(self):
            calls["cleaned"] = True

    async def _fake_create_claude_sandbox(**kwargs):
        calls["sandbox_kwargs"] = kwargs
        return _FakeSandbox()

    module._create_claude_sandbox = _fake_create_claude_sandbox

    dataset_mod = types.ModuleType("examples.swe_agent_blackbox.dataset")
    dataset_mod.extract_image = lambda env_config: env_config.get("image")
    sys.modules["examples.swe_agent_blackbox.dataset"] = dataset_mod

    mini_mod = types.ModuleType("examples.swe_agent_blackbox.mini_swe_agent_runner")
    mini_mod.SandboxEnvForReward = lambda sandbox: sandbox
    sys.modules["examples.swe_agent_blackbox.mini_swe_agent_runner"] = mini_mod

    async def _fake_evaluate_in_env(env, metadata, timeout):
        calls["reward"] = {"metadata": metadata, "timeout": timeout}
        return 1.0, {"resolved": True}

    reward_mod = types.ModuleType("examples.swe_agent_blackbox.reward")
    reward_mod.build_reward_context = lambda tools_kwargs: ({"case": "ok"}, 321)
    reward_mod.evaluate_in_env = _fake_evaluate_in_env
    sys.modules["examples.swe_agent_blackbox.reward"] = reward_mod

    sandbox_mod = types.ModuleType("examples.swe_agent_blackbox.sandbox")
    sandbox_mod.rewrite_gateway_url = lambda gateway_url, strip_v1=False: "http://127.0.0.1:38197/sessions/s"
    sys.modules["examples.swe_agent_blackbox.sandbox"] = sandbox_mod

    monkeypatch.setenv("SWE_AGENT_TOOL_IMAGE", "env-image:bad")
    monkeypatch.setenv("SWE_AGENT_RUN_TIMEOUT", "999")
    monkeypatch.setenv("SWE_AGENT_MAX_TURNS", "7")

    class _Runtime:
        async def complete_session(self, session_id, reward_info=None):
            calls["complete_session"] = {"session_id": session_id, "reward_info": reward_info}

    asyncio.run(
        module.claude_code_runner(
            raw_prompt="fix bug",
            session=types.SimpleNamespace(base_url="http://gw/sessions/s/v1", session_id="sess-1"),
            sample_index=0,
            session_runtime=_Runtime(),
            tools_kwargs={"env": {"image": "testbed:latest"}},
            tool_image="runner-image:ok",
            run_timeout=123,
        )
    )

    assert calls["sandbox_kwargs"]["sidecar_image"] == "runner-image:ok"
    assert calls["runs"][0]["timeout"] == 123
    assert "ANTHROPIC_BASE_URL=http://127.0.0.1:38197/sessions/s" in calls["runs"][0]["cmd"]
    assert "--max-turns 7" in calls["runs"][0]["cmd"]
    assert calls["complete_session"]["reward_info"]["claude_code_exit_code"] == 0
    assert calls["cleaned"] is True


def test_mini_swe_runner_uses_explicit_runner_kwargs(monkeypatch):
    _install_package_stub("examples", "examples")
    _install_package_stub("examples.swe_agent_blackbox", "examples/swe_agent_blackbox")

    dataset_mod = types.ModuleType("examples.swe_agent_blackbox.dataset")
    dataset_mod.extract_image = lambda env_config: env_config.get("image")
    sys.modules["examples.swe_agent_blackbox.dataset"] = dataset_mod

    async def _fake_evaluate_in_env(env, metadata, timeout):
        calls["reward"] = {"metadata": metadata, "timeout": timeout}
        return 1.0, {"resolved": True}

    reward_mod = types.ModuleType("examples.swe_agent_blackbox.reward")
    reward_mod.build_reward_context = lambda tools_kwargs: ({"case": "ok"}, 321)
    reward_mod.evaluate_in_env = _fake_evaluate_in_env
    sys.modules["examples.swe_agent_blackbox.reward"] = reward_mod

    calls = {}

    class _FakeSandbox:
        sandbox_id = "sandbox-1"

        async def run(self, cmd, timeout):
            calls.setdefault("runs", []).append({"cmd": cmd, "timeout": timeout})
            if "run_agent.py" in cmd:
                return types.SimpleNamespace(exit_code=0, stdout='{"exit_status":"submitted","submission":"patch"}', stderr="")
            return types.SimpleNamespace(exit_code=0, stdout="", stderr="")

        async def cleanup(self):
            calls["cleaned"] = True

    class _FakeYRSandbox:
        @classmethod
        async def create(cls, **kwargs):
            calls["sandbox_kwargs"] = kwargs
            return _FakeSandbox()

    sandbox_mod = types.ModuleType("examples.swe_agent_blackbox.sandbox")
    sandbox_mod.CommandResult = object
    sandbox_mod.YRSandbox = _FakeYRSandbox
    sandbox_mod.extract_upstream = lambda gateway_url: "8.8.8.8:1234"
    sandbox_mod.rewrite_gateway_url = lambda gateway_url: "http://127.0.0.1:38197/sessions/s/v1"
    sys.modules["examples.swe_agent_blackbox.sandbox"] = sandbox_mod

    module = _load_module("mini_swe_agent_runner_runtime_test", "examples/swe_agent_blackbox/mini_swe_agent_runner.py")

    monkeypatch.setenv("SWE_AGENT_TOOL_IMAGE", "env-image:bad")
    monkeypatch.setenv("SWE_AGENT_MAX_TURNS", "5")

    class _Runtime:
        async def complete_session(self, session_id, reward_info=None):
            calls["complete_session"] = {"session_id": session_id, "reward_info": reward_info}

    asyncio.run(
        module.mini_swe_agent_runner(
            raw_prompt="fix bug",
            session=types.SimpleNamespace(base_url="http://gw/sessions/s/v1", session_id="sess-1"),
            sample_index=0,
            session_runtime=_Runtime(),
            tools_kwargs={"env": {"image": "testbed:latest"}},
            tool_image="mini-image:ok",
            run_timeout=123,
        )
    )

    assert calls["sandbox_kwargs"]["sidecar_image"] == "mini-image:ok"
    assert calls["runs"][0]["timeout"] == 123
    assert "/opt/mini-swe-agent/bin/run_agent.py" in calls["runs"][0]["cmd"]
    assert calls["complete_session"]["reward_info"]["reward_score"] == 1.0
    assert calls["cleaned"] is True


def test_rewrite_gateway_url_removes_v1_for_anthropic_sdk_base():
    module = _load_module("claude_sandbox_test", "examples/swe_agent_blackbox/sandbox.py")

    assert (
        module.rewrite_gateway_url("http://8.8.8.8:1234/sessions/abc/v1", strip_v1=True)
        == "http://127.0.0.1:38197/sessions/abc"
    )


def test_rewrite_gateway_url_keeps_v1_by_default_for_openai_gateway():
    module = _load_module("claude_sandbox_test", "examples/swe_agent_blackbox/sandbox.py")

    assert (
        module.rewrite_gateway_url("http://8.8.8.8:1234/sessions/abc/v1")
        == "http://127.0.0.1:38197/sessions/abc/v1"
    )
