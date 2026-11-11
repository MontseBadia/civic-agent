"""Hand-written tool loop for condition C, then one JSON-mode call for the answer."""

import hashlib
import json
import time
from dataclasses import dataclass
from typing import Any, Callable

from agent.answer import ANSWER_SCHEMA
from agent.llm import GenerationSettings, LLMClient, LLMResponse, ToolCall
from agent.prompts import C_FINAL_INSTRUCTION

MAX_TOOL_CALLS = 8
# Bug guard above get_evidence's 35,000 budget, not a context budget.
MAX_RESULT_CHARS = 40_000

_JSON_TYPES = {"string": str, "integer": int, "number": (int, float), "boolean": bool}

# The JSON Schema subset validate_arguments understands. Anything else would
# be silently ignored, so Tool refuses it.
_SCHEMA_KEYS = {"type", "properties", "required"}
_PROPERTY_KEYS = {"type", "description"}


class ToolError(Exception):
    """Expected tool failure (not found, bad input). Sent back to the model."""


class InvalidArguments(Exception):
    """Arguments don't fit the tool's parameter schema. Sent back to the model."""


@dataclass(frozen=True)
class Tool:
    name: str
    description: str
    parameters: dict  # JSON schema: type object, properties, required
    function: Callable[..., Any]

    def __post_init__(self):
        schema = self.parameters
        if schema.get("type") != "object":
            raise ValueError(f"{self.name}: parameters must be type object")
        if extra := schema.keys() - _SCHEMA_KEYS:
            raise ValueError(f"{self.name}: unsupported schema keys: {', '.join(sorted(extra))}")
        properties = schema.get("properties", {})
        for prop, spec in properties.items():
            if extra := spec.keys() - _PROPERTY_KEYS:
                raise ValueError(f"{self.name}.{prop}: unsupported keys: {', '.join(sorted(extra))}")
            if spec.get("type") not in _JSON_TYPES:
                raise ValueError(f"{self.name}.{prop}: type must be one of {', '.join(_JSON_TYPES)}")
        if undeclared := set(schema.get("required", [])) - properties.keys():
            raise ValueError(f"{self.name}: required but not declared: {', '.join(sorted(undeclared))}")

    def to_api(self) -> dict:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


@dataclass(frozen=True)
class ModelStep:
    final: bool
    finish_reason: str | None
    prompt_tokens: int | None
    completion_tokens: int | None
    duration_s: float
    tool_call_count: int


@dataclass(frozen=True)
class ToolStep:
    call_id: str
    name: str
    raw_arguments: str
    outcome: str  # ok, invalid_arguments, unknown_tool, tool_error, over_limit
    result: Any  # the tool's return value when outcome is ok, else None
    content: str  # what the model was sent
    duration_s: float


@dataclass(frozen=True)
class LoopResult:
    final: LLMResponse
    trace: tuple[ModelStep | ToolStep, ...]
    messages: tuple[dict, ...]
    limit_reached: bool


def tools_hash(tools: list[Tool]) -> str:
    """Hash of exactly what the model is sent about the tools. Logged next to prompt_hash."""
    payload = json.dumps([t.to_api() for t in tools], sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()[:12]


def validate_arguments(arguments: Any, schema: dict) -> None:
    """Subset of JSON Schema the tools use: types, required, no unknown keys."""
    if not isinstance(arguments, dict):
        raise InvalidArguments("arguments must be a JSON object")
    properties = schema.get("properties", {})
    if unknown := arguments.keys() - properties.keys():
        raise InvalidArguments(f"unknown arguments: {', '.join(sorted(unknown))}")
    if missing := set(schema.get("required", [])) - arguments.keys():
        raise InvalidArguments(f"missing arguments: {', '.join(sorted(missing))}")
    for name, value in arguments.items():
        expected = properties[name]["type"]  # Tool guarantees every property has one
        # bool is a subclass of int in Python, so true would pass as an integer.
        if isinstance(value, bool) and expected != "boolean":
            raise InvalidArguments(f"{name} must be {expected}, got boolean")
        if not isinstance(value, _JSON_TYPES[expected]):
            raise InvalidArguments(f"{name} must be {expected}, got {type(value).__name__}")


def _error_step(call: ToolCall, outcome: str, message: str, start: float) -> ToolStep:
    return ToolStep(
        call_id=call.id,
        name=call.name,
        raw_arguments=call.arguments,
        outcome=outcome,
        result=None,
        content=json.dumps({"error": message}),
        duration_s=time.monotonic() - start,
    )


def _run_call(call: ToolCall, tools: dict[str, Tool]) -> ToolStep:
    start = time.monotonic()
    tool = tools.get(call.name)
    if tool is None:
        available = ", ".join(sorted(tools))
        return _error_step(call, "unknown_tool", f"unknown tool {call.name!r}, available: {available}", start)
    try:
        arguments = json.loads(call.arguments)
        validate_arguments(arguments, tool.parameters)
    except json.JSONDecodeError as e:
        return _error_step(call, "invalid_arguments", f"arguments are not valid JSON: {e}", start)
    except InvalidArguments as e:
        return _error_step(call, "invalid_arguments", str(e), start)
    try:
        result = tool.function(**arguments)
    except ToolError as e:
        return _error_step(call, "tool_error", str(e), start)
    # Any other exception from the tool propagates: it's our bug.

    content = json.dumps(result)
    if len(content) > MAX_RESULT_CHARS:
        raise ValueError(f"{call.name} returned {len(content)} chars, max {MAX_RESULT_CHARS}")
    return ToolStep(
        call_id=call.id,
        name=call.name,
        raw_arguments=call.arguments,
        outcome="ok",
        result=result,
        content=content,
        duration_s=time.monotonic() - start,
    )


def _model_step(response: LLMResponse, final: bool) -> ModelStep:
    return ModelStep(
        final=final,
        finish_reason=response.finish_reason,
        prompt_tokens=response.prompt_tokens,
        completion_tokens=response.completion_tokens,
        duration_s=response.duration_s,
        tool_call_count=len(response.tool_calls),
    )


def _assistant_message(response: LLMResponse) -> dict:
    return {
        "role": "assistant",
        "content": response.content,
        "tool_calls": [
            {"id": c.id, "type": "function", "function": {"name": c.name, "arguments": c.arguments}}
            for c in response.tool_calls
        ],
    }


def run_tool_loop(
    llm: LLMClient,
    messages: list[dict],
    tools: list[Tool],
    settings: GenerationSettings,
    max_tool_calls: int = MAX_TOOL_CALLS,
) -> LoopResult:
    by_name = {t.name: t for t in tools}
    if len(by_name) != len(tools):
        raise ValueError("duplicate tool names")
    api_tools = [t.to_api() for t in tools]
    history = list(messages)
    trace: list[ModelStep | ToolStep] = []
    calls_used = 0
    limit_reached = False

    # Each turn with tool calls uses budget or hits the limit, so this ends.
    while not limit_reached:
        response = llm.chat(history, settings, tools=api_tools)
        trace.append(_model_step(response, final=False))
        if not response.tool_calls:
            if response.content:
                history.append({"role": "assistant", "content": response.content})
            break

        history.append(_assistant_message(response))
        # Every tool call id needs a tool message back, refused ones included.
        for call in response.tool_calls:
            if calls_used >= max_tool_calls:
                limit_reached = True
                step = _error_step(call, "over_limit", "not run: tool call limit reached", time.monotonic())
            else:
                calls_used += 1
                step = _run_call(call, by_name)
            trace.append(step)
            history.append({"role": "tool", "tool_call_id": call.id, "content": step.content})

    history.append({"role": "user", "content": C_FINAL_INSTRUCTION})
    final = llm.chat(history, settings, response_schema=ANSWER_SCHEMA)
    trace.append(_model_step(final, final=True))
    return LoopResult(final=final, trace=tuple(trace), messages=tuple(history), limit_reached=limit_reached)
