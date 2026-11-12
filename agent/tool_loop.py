"""Hand-written tool loop for condition C, then one JSON-mode call for the answer."""

import hashlib
import json
import time
from dataclasses import dataclass
from typing import Any, Callable

from agent.answer import ANSWER_SCHEMA, Answer
from agent.llm import GenerationSettings, LLMClient, LLMResponse, ToolCall
from agent.prompts import C_FINAL_INSTRUCTION

# The needed path is 4 calls (gene, variant, diseases, evidence); 2 spare.
MAX_TOOL_CALLS = 6
# Two repeats means the model has nothing new to look up; go to the answer.
MAX_DUPLICATE_CALLS = 2
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
    outcome: str  # ok, invalid_arguments, unknown_tool, tool_error, duplicate_call, over_limit
    result: Any  # the tool's return value when outcome is ok, else None
    content: str  # what the model was sent
    duration_s: float


@dataclass(frozen=True)
class LoopResult:
    """Result of any condition, so the runner and run log treat A, B and C the same."""

    final: LLMResponse | None  # the answer call; None when code ended the question (B)
    code_answer: Answer | None  # written by code when a B rule fires; else None
    trace: tuple[ModelStep | ToolStep, ...]
    messages: tuple[dict, ...]
    stop_reason: str | None  # C: model_stopped, duplicates, budget. B: which step ended it. A: None

    def __post_init__(self):
        if self.final is not None and self.code_answer is not None:
            raise ValueError("an answer comes from the model or from code, not both")


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


def run_call(call: ToolCall, tools: dict[str, Tool]) -> ToolStep:
    """Also used by B, so B and C run lookups through the same code."""
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


def _casefold_strings(value: Any) -> Any:
    if isinstance(value, str):
        return value.casefold()
    if isinstance(value, dict):
        return {k: _casefold_strings(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_casefold_strings(v) for v in value]
    return value


def _call_key(call: ToolCall) -> str | None:
    """Same tool + same arguments, ignoring key order and text case. None if not valid JSON."""
    try:
        arguments = json.loads(call.arguments)
    except json.JSONDecodeError:
        return None
    return json.dumps([call.name, _casefold_strings(arguments)], sort_keys=True)


def model_step(response: LLMResponse, final: bool) -> ModelStep:
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
    max_duplicate_calls: int = MAX_DUPLICATE_CALLS,
) -> LoopResult:
    by_name = {t.name: t for t in tools}
    if len(by_name) != len(tools):
        raise ValueError("duplicate tool names")
    api_tools = [t.to_api() for t in tools]
    history = list(messages)
    trace: list[ModelStep | ToolStep] = []
    calls_used = 0
    limit_reached = False
    seen_calls: set[str] = set()
    duplicates = 0
    stopped_on_duplicates = False

    # Each turn with tool calls uses budget, so this ends after at most
    # max_tool_calls tool turns.
    while not (limit_reached or stopped_on_duplicates):
        response = llm.chat(history, settings, tools=api_tools)
        trace.append(model_step(response, final=False))
        if not response.tool_calls:
            if response.content:
                history.append({"role": "assistant", "content": response.content})
            break

        history.append(_assistant_message(response))
        # Every tool call id needs a tool message back, refused ones included.
        for call in response.tool_calls:
            if calls_used >= max_tool_calls:
                # Only within one turn that asks for more than the budget left.
                step = _error_step(call, "over_limit", "not run: tool call limit reached", time.monotonic())
            else:
                calls_used += 1
                key = _call_key(call)
                if key is not None and key in seen_calls:
                    # Not run: the result is already in the history. The error is the
                    # latest message, where the model looks, unlike the system prompt rule.
                    message = f"already called {call.name} with these arguments; the result is earlier in the conversation"
                    step = _error_step(call, "duplicate_call", message, time.monotonic())
                    duplicates += 1
                else:
                    step = run_call(call, by_name)
                    if key is not None:
                        seen_calls.add(key)
            trace.append(step)
            history.append({"role": "tool", "tool_call_id": call.id, "content": step.content})
        # Checked after the turn, since every tool call id in it needs a reply.
        # Budget used up: go straight to the final answer. Asking the model once
        # more only to refuse its call cost a full model turn (60 to 150 s).
        limit_reached = calls_used >= max_tool_calls
        stopped_on_duplicates = duplicates >= max_duplicate_calls

    history.append({"role": "user", "content": C_FINAL_INSTRUCTION})
    final = llm.chat(history, settings, response_schema=ANSWER_SCHEMA)
    trace.append(model_step(final, final=True))
    # One reason, so reports count each stop once. When the second duplicate is
    # also the last budgeted call, duplicates is the more specific reason.
    if stopped_on_duplicates:
        stop_reason = "duplicates"
    elif limit_reached:
        stop_reason = "budget"
    else:
        stop_reason = "model_stopped"
    return LoopResult(final=final, code_answer=None, trace=tuple(trace), messages=tuple(history),
                      stop_reason=stop_reason)
