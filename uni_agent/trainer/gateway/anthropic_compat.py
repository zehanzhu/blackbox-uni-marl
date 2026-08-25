"""Anthropic Messages API <-> internal OpenAI-format conversion.

The gateway tracks sessions in the OpenAI Chat Completions message format
(the format consumed by chat templates and the prefix-comparison logic in
``gateway.py``).  This module converts an Anthropic ``/v1/messages`` request
into that internal format, and converts the gateway's chat-completion result
back into an Anthropic Message response.

Round-trip stability matters: when an Anthropic-API agent echoes a previous
assistant turn back (text + tool_use blocks) the conversion must reproduce
exactly the normalized OpenAI message the gateway stored in
``session.message_history``, otherwise the prefix check fails and the gateway
re-encodes the conversation as a new trajectory.  Both directions are kept
deterministic for that reason (text blocks joined with "\\n", tool arguments
serialized/parsed as JSON, which ``_canonicalize_tool_arguments_for_comparison``
treats as equivalent).
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any
from uuid import uuid4

from uni_agent.trainer.gateway.types import MalformedRequestError

logger = logging.getLogger(__name__)

# Anthropic request fields copied through for _build_sampling_params.
# "stop_sequences" is renamed to OpenAI's "stop" (only honored when the
# gateway operator adds "stop" to allowed_request_sampling_param_keys).
_PASSTHROUGH_SAMPLING_KEYS = ("temperature", "top_p", "top_k", "max_tokens")

_CLAUDE_CODE_BILLING_HEADER_RE = re.compile(
    r"^\s*x-anthropic-billing-header:[^\n]*\n?",
    re.IGNORECASE,
)

# OpenAI finish_reason -> Anthropic stop_reason.
_STOP_REASON_MAP = {
    "stop": "end_turn",
    "length": "max_tokens",
    "tool_calls": "tool_use",
    "function_call": "tool_use",
    "content_filter": "refusal",
}

# FastAPI HTTP status -> Anthropic error.type.
ANTHROPIC_ERROR_TYPE_BY_STATUS = {
    400: "invalid_request_error",
    401: "authentication_error",
    403: "permission_error",
    404: "not_found_error",
    409: "invalid_request_error",
    413: "request_too_large",
    429: "rate_limit_error",
    500: "api_error",
    529: "overloaded_error",
}


def _join_text_blocks(blocks: list[Any], *, context: str) -> str:
    texts = []
    for block in blocks:
        if not isinstance(block, dict) or block.get("type") != "text":
            raise MalformedRequestError(f"{context} only supports text blocks, got: {block!r}")
        texts.append(str(block.get("text", "")))
    return "\n".join(texts)


def _convert_system(system: Any) -> str:
    if isinstance(system, str):
        return _CLAUDE_CODE_BILLING_HEADER_RE.sub("", system)
    if isinstance(system, list):
        return _CLAUDE_CODE_BILLING_HEADER_RE.sub("", _join_text_blocks(system, context="system"))
    raise MalformedRequestError("system must be a string or a list of text blocks")


def _image_block_to_openai_part(block: dict[str, Any]) -> dict[str, Any]:
    source = block.get("source")
    if not isinstance(source, dict):
        raise MalformedRequestError("image block requires a source object")
    source_type = source.get("type")
    if source_type == "base64":
        media_type = source.get("media_type", "image/png")
        url = f"data:{media_type};base64,{source.get('data', '')}"
    elif source_type == "url":
        url = source.get("url", "")
    else:
        raise MalformedRequestError(f"unsupported image source type: {source_type!r}")
    return {"type": "image_url", "image_url": {"url": url}}


def _tool_result_content_to_openai(content: Any) -> str | list[dict[str, Any]]:
    """Convert tool_result content (string or text/image blocks) to OpenAI tool content."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        raise MalformedRequestError("tool_result content must be a string or a list of blocks")

    parts: list[dict[str, Any]] = []
    all_text = True
    for block in content:
        if not isinstance(block, dict):
            raise MalformedRequestError("tool_result content blocks must be objects")
        block_type = block.get("type")
        if block_type == "text":
            parts.append({"type": "text", "text": str(block.get("text", ""))})
        elif block_type == "image":
            parts.append(_image_block_to_openai_part(block))
            all_text = False
        else:
            raise MalformedRequestError(f"unsupported tool_result block type: {block_type!r}")
    if all_text:
        return "\n".join(part["text"] for part in parts)
    return parts


def _convert_user_message(content: Any) -> list[dict[str, Any]]:
    """Convert one Anthropic user message into OpenAI messages.

    tool_result blocks become individual ``role: "tool"`` messages (emitted
    first, directly after the preceding assistant tool_calls turn); the
    remaining text/image blocks form a trailing ``role: "user"`` message.
    """
    if isinstance(content, str):
        return [{"role": "user", "content": content}]
    if not isinstance(content, list):
        raise MalformedRequestError("user message content must be a string or a list of blocks")

    tool_messages: list[dict[str, Any]] = []
    user_parts: list[dict[str, Any]] = []
    has_image = False
    for block in content:
        if not isinstance(block, dict):
            raise MalformedRequestError("message content blocks must be objects")
        block_type = block.get("type")
        if block_type == "tool_result":
            tool_use_id = block.get("tool_use_id")
            if not tool_use_id:
                raise MalformedRequestError("tool_result block requires tool_use_id")
            tool_messages.append(
                {
                    "role": "tool",
                    "tool_call_id": str(tool_use_id),
                    "content": _tool_result_content_to_openai(block.get("content")),
                }
            )
        elif block_type == "text":
            user_parts.append({"type": "text", "text": str(block.get("text", ""))})
        elif block_type == "image":
            user_parts.append(_image_block_to_openai_part(block))
            has_image = True
        else:
            raise MalformedRequestError(f"unsupported user content block type: {block_type!r}")

    messages = list(tool_messages)
    if user_parts:
        if has_image:
            messages.append({"role": "user", "content": user_parts})
        else:
            messages.append({"role": "user", "content": "\n".join(part["text"] for part in user_parts)})
    return messages


def _convert_assistant_message(content: Any) -> dict[str, Any]:
    if isinstance(content, str):
        return {"role": "assistant", "content": content}
    if not isinstance(content, list):
        raise MalformedRequestError("assistant message content must be a string or a list of blocks")

    texts: list[str] = []
    tool_calls: list[dict[str, Any]] = []
    for block in content:
        if not isinstance(block, dict):
            raise MalformedRequestError("message content blocks must be objects")
        block_type = block.get("type")
        if block_type == "text":
            texts.append(str(block.get("text", "")))
        elif block_type == "tool_use":
            tool_calls.append(
                {
                    "id": str(block.get("id") or f"call_{uuid4().hex[:8]}"),
                    "type": "function",
                    "function": {
                        "name": str(block.get("name", "")),
                        "arguments": json.dumps(block.get("input") or {}, ensure_ascii=False),
                    },
                }
            )
        elif block_type in ("thinking", "redacted_thinking"):
            # Thinking blocks have no chat-template representation; the model's
            # reasoning tokens are already captured in the token trajectory.
            continue
        else:
            raise MalformedRequestError(f"unsupported assistant content block type: {block_type!r}")

    message: dict[str, Any] = {"role": "assistant", "content": "\n".join(texts)}
    if tool_calls:
        message["tool_calls"] = tool_calls
    return message


def _infer_json_schema_type(schema: dict[str, Any]) -> str:
    schema_type = schema.get("type")
    if isinstance(schema_type, str):
        return schema_type
    if isinstance(schema_type, list) and schema_type:
        non_null = [item for item in schema_type if item != "null"]
        if non_null:
            return str(non_null[0])
        return str(schema_type[0])

    for key in ("anyOf", "oneOf", "allOf"):
        variants = schema.get(key)
        if isinstance(variants, list):
            for variant in variants:
                if isinstance(variant, dict):
                    variant_type = _infer_json_schema_type(variant)
                    if variant_type:
                        return variant_type

    if "const" in schema:
        value = schema["const"]
        if isinstance(value, bool):
            return "boolean"
        if isinstance(value, int | float):
            return "number"
        if isinstance(value, list):
            return "array"
        if isinstance(value, dict):
            return "object"
        return "string"

    enum = schema.get("enum")
    if isinstance(enum, list) and enum:
        value = next((item for item in enum if item is not None), enum[0])
        if isinstance(value, bool):
            return "boolean"
        if isinstance(value, int | float):
            return "number"
        return "string"

    if "properties" in schema or "additionalProperties" in schema:
        return "object"
    if "items" in schema:
        return "array"
    return "string"


def _collect_json_schema_enum(schema: dict[str, Any]) -> list[Any] | None:
    enum_values: list[Any] = []
    if isinstance(schema.get("enum"), list):
        enum_values.extend(schema["enum"])
    if "const" in schema:
        enum_values.append(schema["const"])

    for key in ("anyOf", "oneOf", "allOf"):
        variants = schema.get(key)
        if not isinstance(variants, list):
            continue
        for variant in variants:
            if not isinstance(variant, dict):
                continue
            variant_enum = _collect_json_schema_enum(variant)
            if variant_enum:
                enum_values.extend(variant_enum)

    if not enum_values:
        return None

    deduped: list[Any] = []
    for value in enum_values:
        if value not in deduped:
            deduped.append(value)
    return deduped


def _normalize_openai_property_schema(schema: Any) -> dict[str, Any]:
    if not isinstance(schema, dict):
        return {"type": "string", "description": str(schema)}

    normalized: dict[str, Any] = {"type": _infer_json_schema_type(schema)}
    if isinstance(schema.get("description"), str):
        normalized["description"] = schema["description"]
    enum = _collect_json_schema_enum(schema)
    if enum is not None:
        normalized["enum"] = enum
    return normalized


def _normalize_openai_parameters_schema(schema: Any) -> dict[str, Any]:
    if not isinstance(schema, dict):
        return {"type": "object", "properties": {}, "required": []}

    properties = schema.get("properties")
    if not isinstance(properties, dict):
        properties = {}

    required = schema.get("required")
    if not isinstance(required, list):
        required = []

    return {
        "type": "object",
        "properties": {
            str(name): _normalize_openai_property_schema(property_schema)
            for name, property_schema in properties.items()
        },
        "required": [str(name) for name in required],
    }


def _convert_tools(tools: Any) -> list[dict[str, Any]] | None:
    if tools is None:
        return None
    if not isinstance(tools, list):
        raise MalformedRequestError("tools must be a list")
    converted = []
    for tool in tools:
        if not isinstance(tool, dict):
            raise MalformedRequestError("tools entries must be objects")
        tool_type = tool.get("type")
        if tool_type not in (None, "custom"):
            raise MalformedRequestError(
                f"unsupported tool type: {tool_type!r} (only client tools with input_schema are supported)"
            )
        if not tool.get("name"):
            raise MalformedRequestError("tool requires a name")
        function: dict[str, Any] = {
            "name": str(tool["name"]),
            "description": str(tool.get("description") or ""),
            "parameters": _normalize_openai_parameters_schema(tool.get("input_schema")),
        }
        converted.append({"type": "function", "function": function})
    return converted


def anthropic_payload_to_openai(payload: dict[str, Any]) -> dict[str, Any]:
    """Convert an Anthropic ``/v1/messages`` request payload to the OpenAI
    Chat Completions payload shape consumed by the gateway."""
    if not isinstance(payload, dict):
        raise MalformedRequestError("request body must be a JSON object")

    raw_messages = payload.get("messages")
    if not isinstance(raw_messages, list) or not raw_messages:
        raise MalformedRequestError("messages must be non-empty")

    system_texts: list[str] = []
    messages: list[dict[str, Any]] = []
    if payload.get("system") is not None:
        system_texts.append(_convert_system(payload["system"]))

    for message in raw_messages:
        if not isinstance(message, dict):
            raise MalformedRequestError("messages entries must be objects")
        role = message.get("role")
        if role == "system":
            system_texts.append(_convert_system(message.get("content")))
        elif role == "user":
            messages.extend(_convert_user_message(message.get("content")))
        elif role == "assistant":
            messages.append(_convert_assistant_message(message.get("content")))
        else:
            raise MalformedRequestError(f"unsupported message role: {role!r}")

    if system_texts:
        messages.insert(0, {"role": "system", "content": "\n".join(system_texts)})

    openai_payload: dict[str, Any] = {"messages": messages}
    tools = _convert_tools(payload.get("tools"))
    if tools is not None:
        openai_payload["tools"] = tools
    for key in _PASSTHROUGH_SAMPLING_KEYS:
        if key in payload:
            openai_payload[key] = payload[key]
    if "stop_sequences" in payload:
        openai_payload["stop"] = payload["stop_sequences"]
    if "model" in payload:
        openai_payload["model"] = payload["model"]
    return openai_payload


def _tool_call_input(arguments: Any) -> dict[str, Any]:
    if isinstance(arguments, dict):
        return arguments
    if isinstance(arguments, str):
        try:
            parsed = json.loads(arguments)
        except json.JSONDecodeError:
            logger.warning("tool call arguments are not valid JSON; returning empty input: %.200s", arguments)
            return {}
        if isinstance(parsed, dict):
            return parsed
    logger.warning("tool call arguments are not a JSON object; returning empty input: %.200s", arguments)
    return {}


def openai_completion_to_anthropic_message(
    completion: dict[str, Any],
    *,
    request_model: str | None = None,
) -> dict[str, Any]:
    """Convert the gateway's chat-completion response dict to an Anthropic Message."""
    choice = completion["choices"][0]
    assistant_message = choice["message"]
    finish_reason = choice.get("finish_reason")

    content: list[dict[str, Any]] = []
    text = assistant_message.get("content")
    if text:
        content.append({"type": "text", "text": text})
    for tool_call in assistant_message.get("tool_calls") or []:
        function = tool_call.get("function") or {}
        content.append(
            {
                "type": "tool_use",
                "id": tool_call.get("id") or f"call_{uuid4().hex[:8]}",
                "name": function.get("name", ""),
                "input": _tool_call_input(function.get("arguments")),
            }
        )

    usage = completion.get("usage") or {}
    return {
        "id": f"msg_{uuid4().hex}",
        "type": "message",
        "role": "assistant",
        "model": request_model or completion.get("model") or "default",
        "content": content,
        "stop_reason": _STOP_REASON_MAP.get(finish_reason, "end_turn"),
        "stop_sequence": None,
        "usage": {
            "input_tokens": int(usage.get("prompt_tokens", 0)),
            "output_tokens": int(usage.get("completion_tokens", 0)),
        },
    }


def anthropic_error_body(status_code: int, message: str) -> dict[str, Any]:
    return {
        "type": "error",
        "error": {
            "type": ANTHROPIC_ERROR_TYPE_BY_STATUS.get(status_code, "api_error"),
            "message": message,
        },
    }


def anthropic_stream_events(message: dict[str, Any]) -> list[dict[str, Any]]:
    """Build Anthropic Messages SSE events from a non-streaming Message body.

    The gateway backend currently generates a whole assistant turn at once.
    For clients that require Anthropic streaming, emit a valid SSE event
    sequence with the generated content split by content block.
    """
    content = list(message.get("content") or [])
    start_message = dict(message)
    start_message["content"] = []

    events: list[dict[str, Any]] = [
        {"type": "message_start", "message": start_message},
    ]
    for index, block in enumerate(content):
        block_type = block.get("type")
        if block_type == "text":
            events.append(
                {
                    "type": "content_block_start",
                    "index": index,
                    "content_block": {"type": "text", "text": ""},
                }
            )
            text = block.get("text", "")
            if text:
                events.append(
                    {
                        "type": "content_block_delta",
                        "index": index,
                        "delta": {"type": "text_delta", "text": text},
                    }
                )
            events.append({"type": "content_block_stop", "index": index})
        elif block_type == "tool_use":
            events.append(
                {
                    "type": "content_block_start",
                    "index": index,
                    "content_block": {
                        "type": "tool_use",
                        "id": block.get("id"),
                        "name": block.get("name", ""),
                        "input": {},
                    },
                }
            )
            input_json = json.dumps(block.get("input") or {}, ensure_ascii=False)
            if input_json:
                events.append(
                    {
                        "type": "content_block_delta",
                        "index": index,
                        "delta": {"type": "input_json_delta", "partial_json": input_json},
                    }
                )
            events.append({"type": "content_block_stop", "index": index})

    events.append(
        {
            "type": "message_delta",
            "delta": {
                "stop_reason": message.get("stop_reason"),
                "stop_sequence": message.get("stop_sequence"),
            },
            "usage": {"output_tokens": int((message.get("usage") or {}).get("output_tokens", 0))},
        }
    )
    events.append({"type": "message_stop"})
    return events


def encode_anthropic_sse_event(event: dict[str, Any]) -> str:
    return f"event: {event['type']}\ndata: {json.dumps(event, ensure_ascii=False)}\n\n"
