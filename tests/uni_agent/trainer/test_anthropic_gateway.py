"""Tests for the Anthropic Messages API support in the gateway/framework.

Tiers:
  1. Pure conversion unit tests (anthropic_compat).
  2. Round-trip prefix stability against the gateway's prefix comparison.
  3. Framework session-handle preparation.
  4. End-to-end gateway test with the real ``anthropic`` SDK over an ASGI
     transport (fake tokenizer + fake generation backend, no GPU / Ray cluster).
"""

import asyncio
import json
from types import SimpleNamespace

import pytest

from uni_agent.trainer.gateway.anthropic_compat import (
    anthropic_payload_to_openai,
    openai_completion_to_anthropic_message,
)
from uni_agent.trainer.gateway.types import MalformedRequestError


# =====================================================================
# Tier 1 — request conversion
# =====================================================================


class TestAnthropicRequestConversion:
    def test_system_and_user_text(self):
        payload = {
            "model": "default",
            "max_tokens": 128,
            "temperature": 0.7,
            "system": "Be terse.",
            "messages": [{"role": "user", "content": "hello"}],
            "stop_sequences": ["\nDONE"],
        }
        converted = anthropic_payload_to_openai(payload)
        assert converted["messages"] == [
            {"role": "system", "content": "Be terse."},
            {"role": "user", "content": "hello"},
        ]
        assert converted["max_tokens"] == 128
        assert converted["temperature"] == 0.7
        assert converted["stop"] == ["\nDONE"]
        assert converted["model"] == "default"

    def test_system_block_list(self):
        payload = {
            "system": [{"type": "text", "text": "a"}, {"type": "text", "text": "b"}],
            "messages": [{"role": "user", "content": "x"}],
        }
        converted = anthropic_payload_to_openai(payload)
        assert converted["messages"][0] == {"role": "system", "content": "a\nb"}

    def test_tools_conversion(self):
        payload = {
            "messages": [{"role": "user", "content": "x"}],
            "tools": [
                {
                    "name": "get_weather",
                    "description": "Get weather",
                    "input_schema": {"type": "object", "properties": {"location": {"type": "string"}}},
                }
            ],
        }
        converted = anthropic_payload_to_openai(payload)
        assert converted["tools"] == [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "parameters": {"type": "object", "properties": {"location": {"type": "string"}}},
                    "description": "Get weather",
                },
            }
        ]

    def test_server_tool_rejected(self):
        payload = {
            "messages": [{"role": "user", "content": "x"}],
            "tools": [{"type": "web_search_20260209", "name": "web_search"}],
        }
        with pytest.raises(MalformedRequestError):
            anthropic_payload_to_openai(payload)

    def test_stream_rejected(self):
        payload = {"stream": True, "messages": [{"role": "user", "content": "x"}]}
        with pytest.raises(MalformedRequestError):
            anthropic_payload_to_openai(payload)

    def test_assistant_tool_use_to_tool_calls(self):
        payload = {
            "messages": [
                {"role": "user", "content": "x"},
                {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": "checking"},
                        {"type": "tool_use", "id": "call_1", "name": "get_weather", "input": {"location": "Paris"}},
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "tool_result", "tool_use_id": "call_1", "content": "sunny"},
                    ],
                },
            ],
        }
        converted = anthropic_payload_to_openai(payload)
        assert converted["messages"][1] == {
            "role": "assistant",
            "content": "checking",
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "get_weather", "arguments": '{"location": "Paris"}'},
                }
            ],
        }
        assert converted["messages"][2] == {"role": "tool", "tool_call_id": "call_1", "content": "sunny"}

    def test_tool_result_block_list_content(self):
        payload = {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "call_1",
                            "content": [{"type": "text", "text": "line1"}, {"type": "text", "text": "line2"}],
                        },
                        {"type": "text", "text": "continue"},
                    ],
                },
            ],
        }
        converted = anthropic_payload_to_openai(payload)
        assert converted["messages"] == [
            {"role": "tool", "tool_call_id": "call_1", "content": "line1\nline2"},
            {"role": "user", "content": "continue"},
        ]

    def test_image_block(self):
        payload = {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "what is this"},
                        {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "AAAA"}},
                    ],
                },
            ],
        }
        converted = anthropic_payload_to_openai(payload)
        assert converted["messages"][0]["content"] == [
            {"type": "text", "text": "what is this"},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}},
        ]

    def test_thinking_blocks_skipped(self):
        payload = {
            "messages": [
                {"role": "user", "content": "x"},
                {
                    "role": "assistant",
                    "content": [
                        {"type": "thinking", "thinking": "...", "signature": "s"},
                        {"type": "text", "text": "answer"},
                    ],
                },
                {"role": "user", "content": "y"},
            ],
        }
        converted = anthropic_payload_to_openai(payload)
        assert converted["messages"][1] == {"role": "assistant", "content": "answer"}

    def test_unknown_role_rejected(self):
        payload = {"messages": [{"role": "tool", "content": "x"}]}
        with pytest.raises(MalformedRequestError):
            anthropic_payload_to_openai(payload)

    def test_empty_messages_rejected(self):
        with pytest.raises(MalformedRequestError):
            anthropic_payload_to_openai({"messages": []})


# =====================================================================
# Tier 1 — response conversion
# =====================================================================


def _completion(message, finish_reason, prompt_tokens=10, completion_tokens=5):
    return {
        "id": "chatcmpl-abc",
        "object": "chat.completion",
        "choices": [{"index": 0, "message": message, "finish_reason": finish_reason}],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }


class TestAnthropicResponseConversion:
    def test_text_response(self):
        result = openai_completion_to_anthropic_message(
            _completion({"role": "assistant", "content": "hi"}, "stop"), request_model="m"
        )
        assert result["type"] == "message"
        assert result["role"] == "assistant"
        assert result["model"] == "m"
        assert result["content"] == [{"type": "text", "text": "hi"}]
        assert result["stop_reason"] == "end_turn"
        assert result["usage"] == {"input_tokens": 10, "output_tokens": 5}

    def test_tool_call_response(self):
        message = {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "get_weather", "arguments": '{"location": "Paris"}'},
                }
            ],
        }
        result = openai_completion_to_anthropic_message(_completion(message, "tool_calls"))
        assert result["stop_reason"] == "tool_use"
        assert result["content"] == [
            {"type": "tool_use", "id": "call_1", "name": "get_weather", "input": {"location": "Paris"}}
        ]

    def test_length_maps_to_max_tokens(self):
        result = openai_completion_to_anthropic_message(
            _completion({"role": "assistant", "content": "trunc"}, "length")
        )
        assert result["stop_reason"] == "max_tokens"

    def test_invalid_tool_arguments_fall_back_to_empty_input(self):
        message = {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {"id": "call_1", "type": "function", "function": {"name": "f", "arguments": "{not json"}}
            ],
        }
        result = openai_completion_to_anthropic_message(_completion(message, "tool_calls"))
        assert result["content"][0]["input"] == {}


# =====================================================================
# Tier 2 — round-trip prefix stability
# =====================================================================


class TestRoundTripPrefixStability:
    """An Anthropic agent echoing the previous turn must reproduce the
    normalized OpenAI messages the gateway stored, so the gateway's prefix
    check keeps extending the same trajectory."""

    def test_tool_use_round_trip_is_prefix(self):
        from uni_agent.trainer.gateway.gateway import _is_message_prefix, _normalize_message

        request_1 = {
            "system": "sys",
            "messages": [{"role": "user", "content": "weather in paris?"}],
        }
        converted_1 = anthropic_payload_to_openai(request_1)

        # Assistant message exactly as the gateway's _decode_response builds it.
        assistant_msg = {
            "role": "assistant",
            "content": "Let me check.",
            "tool_calls": [
                {
                    "id": "call_abc",
                    "type": "function",
                    "function": {"name": "get_weather", "arguments": '{"location":"Paris"}'},
                }
            ],
        }
        stored_history = [_normalize_message(m) for m in converted_1["messages"]] + [
            _normalize_message(assistant_msg)
        ]

        # The agent receives the converted Anthropic response and echoes it back.
        anthropic_response = openai_completion_to_anthropic_message(_completion(assistant_msg, "tool_calls"))
        request_2 = {
            "system": "sys",
            "messages": [
                {"role": "user", "content": "weather in paris?"},
                {"role": "assistant", "content": anthropic_response["content"]},
                {
                    "role": "user",
                    "content": [{"type": "tool_result", "tool_use_id": "call_abc", "content": "sunny, 21C"}],
                },
            ],
        }
        converted_2 = [_normalize_message(m) for m in anthropic_payload_to_openai(request_2)["messages"]]
        assert _is_message_prefix(stored_history, converted_2)
        # The suffix beyond the stored prefix is exactly the tool result.
        assert converted_2[len(stored_history):] == [
            {"role": "tool", "tool_call_id": "call_abc", "content": "sunny, 21C"}
        ]


# =====================================================================
# Tier 3 — framework session handle
# =====================================================================


class TestAnthropicFramework:
    def test_prepare_session_handle_strips_v1(self):
        from uni_agent.trainer.framework.framework import AnthropicCompatibleAgentFramework
        from uni_agent.trainer.framework.types import SessionHandle

        framework = AnthropicCompatibleAgentFramework(session_runtime=None, agent_runner=None)
        handle = SessionHandle(session_id="s", base_url="http://h:1/sessions/s/v1")
        prepared = framework._prepare_session_handle(handle)
        assert prepared.base_url == "http://h:1/sessions/s"
        assert prepared.session_id == "s"

    def test_prepare_session_handle_passthrough(self):
        from uni_agent.trainer.framework.framework import AnthropicCompatibleAgentFramework
        from uni_agent.trainer.framework.types import SessionHandle

        framework = AnthropicCompatibleAgentFramework(session_runtime=None, agent_runner=None)
        handle = SessionHandle(session_id="s", base_url=None)
        assert framework._prepare_session_handle(handle) is handle

    def test_framework_export(self):
        from uni_agent.trainer.framework import AnthropicCompatibleAgentFramework  # noqa: F401


# =====================================================================
# Tier 4 — end-to-end through the gateway app with the anthropic SDK
# =====================================================================


class FakeTokenizer:
    """Deterministic chat template: utf-8 bytes of a tagged text rendering."""

    eos_token_id = 0

    def _render(self, messages, tools=None, add_generation_prompt=True):
        text = ""
        if tools:
            text += f"[tools:{json.dumps(tools, sort_keys=True)}]"
        for message in messages:
            content = message.get("content", "")
            if isinstance(content, (list, dict)):
                content = json.dumps(content, sort_keys=True)
            if message.get("tool_calls"):
                content += json.dumps(message["tool_calls"], sort_keys=True)
            text += f"<{message['role']}>{content}</>"
        if add_generation_prompt:
            text += "<assistant>"
        return text

    def apply_chat_template(
        self, messages, tokenize=True, add_generation_prompt=True, tools=None, return_dict=False, **kwargs
    ):
        text = self._render(messages, tools=tools, add_generation_prompt=add_generation_prompt)
        if not tokenize:
            return text
        return list(text.encode("utf-8"))

    def decode(self, ids, skip_special_tokens=True):
        return bytes(int(i) for i in ids).decode("utf-8", errors="ignore")


class FakeBackend:
    """Replays queued (text, stop_reason) generations as token ids."""

    def __init__(self, outputs):
        self.outputs = list(outputs)
        self.requests = []

    async def generate(self, request_id, prompt_ids, sampling_params, image_data=None, video_data=None, **kwargs):
        self.requests.append({"prompt_ids": list(prompt_ids), "sampling_params": dict(sampling_params)})
        text, stop_reason = self.outputs.pop(0)
        return SimpleNamespace(
            token_ids=list(text.encode("utf-8")),
            log_probs=None,
            stop_reason=stop_reason,
        )


def _make_gateway(backend):
    from uni_agent.trainer.gateway.gateway import _GatewayActor

    actor = _GatewayActor(FakeTokenizer(), backend, tool_parser_name="hermes")
    # Mark as started without binding a real uvicorn server; tests drive the
    # FastAPI app directly through an ASGI transport.
    actor._server_base_url = "http://testserver"
    return actor


TOOL_CALL_TEXT = (
    'I will check.\n<tool_call>\n{"name": "get_weather", "arguments": {"location": "Paris"}}\n</tool_call>'
)


class TestGatewayAnthropicEndpoint:
    def test_two_turn_tool_session_with_anthropic_sdk(self):
        asyncio.run(self._run_two_turn_tool_session())

    async def _run_two_turn_tool_session(self):
        import httpx
        from anthropic import AsyncAnthropic

        backend = FakeBackend([(TOOL_CALL_TEXT, "completed"), ("It is sunny in Paris.", "completed")])
        actor = _make_gateway(backend)
        handle = await actor.create_session("anthropic-e2e")

        # AnthropicCompatibleAgentFramework hands runners base_url without /v1;
        # the SDK appends /v1/messages itself.
        base_url = handle.base_url.removesuffix("/v1")
        client = AsyncAnthropic(
            base_url=base_url,
            api_key="not-needed",
            http_client=httpx.AsyncClient(transport=httpx.ASGITransport(app=actor._app)),
            max_retries=0,
        )
        tools = [
            {
                "name": "get_weather",
                "description": "Get weather for a location",
                "input_schema": {
                    "type": "object",
                    "properties": {"location": {"type": "string"}},
                    "required": ["location"],
                },
            }
        ]

        first = await client.messages.create(
            model="default",
            max_tokens=256,
            system="You are a weather agent.",
            tools=tools,
            messages=[{"role": "user", "content": "What's the weather in Paris?"}],
        )
        assert first.stop_reason == "tool_use"
        tool_use = next(block for block in first.content if block.type == "tool_use")
        assert tool_use.name == "get_weather"
        assert tool_use.input == {"location": "Paris"}
        assert first.usage.input_tokens > 0 and first.usage.output_tokens > 0

        second = await client.messages.create(
            model="default",
            max_tokens=256,
            system="You are a weather agent.",
            tools=tools,
            messages=[
                {"role": "user", "content": "What's the weather in Paris?"},
                {"role": "assistant", "content": first.content},
                {
                    "role": "user",
                    "content": [
                        {"type": "tool_result", "tool_use_id": tool_use.id, "content": "sunny, 21C"}
                    ],
                },
            ],
        )
        assert second.stop_reason == "end_turn"
        assert second.content[0].text == "It is sunny in Paris."

        trajectories = await actor.finalize_session("anthropic-e2e")
        # Prefix matching must extend a single trajectory across both turns,
        # not materialize one per request.
        assert len(trajectories) == 1
        trajectory = trajectories[0]
        # First generation tokens are trainable (mask 1); the tool-result
        # continuation is masked 0; second generation is mask 1 again.
        mask = trajectory.response_mask
        first_gen = len(TOOL_CALL_TEXT.encode("utf-8"))
        second_gen = len("It is sunny in Paris.".encode("utf-8"))
        assert mask[:first_gen] == [1] * first_gen
        assert mask[-second_gen:] == [1] * second_gen
        assert 0 in mask[first_gen:-second_gen]
        assert len(trajectory.response_ids) == len(mask)

        await client.close()

    def test_error_envelope_for_unknown_session(self):
        asyncio.run(self._run_unknown_session())

    async def _run_unknown_session(self):
        import httpx
        from anthropic import AsyncAnthropic, NotFoundError

        backend = FakeBackend([])
        actor = _make_gateway(backend)
        client = AsyncAnthropic(
            base_url="http://testserver/sessions/missing",
            api_key="not-needed",
            http_client=httpx.AsyncClient(transport=httpx.ASGITransport(app=actor._app)),
            max_retries=0,
        )
        with pytest.raises(NotFoundError):
            await client.messages.create(
                model="default",
                max_tokens=16,
                messages=[{"role": "user", "content": "hi"}],
            )
        await client.close()

    def test_openai_route_still_works(self):
        asyncio.run(self._run_openai_route())

    async def _run_openai_route(self):
        import httpx

        backend = FakeBackend([("plain answer", "completed")])
        actor = _make_gateway(backend)
        await actor.create_session("openai-regression")

        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=actor._app), base_url="http://testserver"
        ) as client:
            response = await client.post(
                "/sessions/openai-regression/v1/chat/completions",
                json={"messages": [{"role": "user", "content": "hi"}], "max_tokens": 16},
            )
        assert response.status_code == 200
        body = response.json()
        assert body["object"] == "chat.completion"
        assert body["choices"][0]["message"]["content"] == "plain answer"
        assert body["choices"][0]["finish_reason"] == "stop"
