"""One chat call to an OpenAI-compatible endpoint, returned in a fixed shape."""

import time
from dataclasses import dataclass
from typing import Any

import openai

from agent.config import LLMConfig

DEFAULT_TIMEOUT_S = 300.0

# The values that switch top-p and top-k off (mlx-vlm: generation.py applies top_k
# only when > 0, top_p only when < 1). Greedy decoding uses temperature 0 with both off.
TOP_P_OFF = 1.0
TOP_K_OFF = 0


@dataclass(frozen=True)
class GenerationSettings:
    """No defaults on purpose: every call states every value that changes the output."""

    temperature: float
    seed: int
    max_tokens: int
    top_p: float
    top_k: int

    def __post_init__(self):
        # The server accepts bad or misspelled values silently, so check here.
        if self.temperature < 0:
            raise ValueError(f"temperature must be >= 0, got {self.temperature}")
        if not 0 < self.top_p <= 1:
            raise ValueError(f"top_p must be in (0, 1], got {self.top_p}")
        if self.top_k < 0:
            raise ValueError(f"top_k must be >= 0 (0 is off), got {self.top_k}")
        if self.max_tokens <= 0:
            raise ValueError(f"max_tokens must be > 0, got {self.max_tokens}")


@dataclass(frozen=True)
class ToolCall:
    id: str
    name: str
    arguments: str  # raw JSON text from the model; the tool loop parses it


@dataclass(frozen=True)
class LLMResponse:
    content: str | None
    tool_calls: tuple[ToolCall, ...]
    finish_reason: str | None
    model: str
    prompt_tokens: int | None
    completion_tokens: int | None
    duration_s: float


class LLMClient:
    def __init__(
        self,
        config: LLMConfig,
        timeout_s: float = DEFAULT_TIMEOUT_S,
        client: Any = None,
    ):
        self.model = config.model
        # max_retries=0: every logged call is exactly one generation.
        if client is None:
            client = openai.OpenAI(
                base_url=config.base_url,
                api_key=config.api_key,
                max_retries=0,
                timeout=timeout_s,
            )
        self._client = client

    def chat(
        self,
        messages: list[dict],
        settings: GenerationSettings,
        tools: list[dict] | None = None,
        response_schema: dict | None = None,
    ) -> LLMResponse:
        request: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": settings.temperature,
            "seed": settings.seed,
            "max_tokens": settings.max_tokens,
            "top_p": settings.top_p,
            # top_k isn't in OpenAI's API; extra_body is merged into the JSON body as is.
            "extra_body": {"top_k": settings.top_k},
        }
        if tools and response_schema is not None:
            # JSON mode constrains every token, so a tool call couldn't be emitted.
            raise ValueError("response_schema and tools can't be used in the same call")
        if tools:
            request["tools"] = tools
        if response_schema is not None:
            request["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": "answer", "schema": response_schema},
            }

        start = time.monotonic()
        completion = self._client.chat.completions.create(**request)
        duration_s = time.monotonic() - start

        choice = completion.choices[0]
        tool_calls = tuple(
            ToolCall(id=c.id, name=c.function.name, arguments=c.function.arguments)
            for c in choice.message.tool_calls or []
        )
        usage = completion.usage
        return LLMResponse(
            content=choice.message.content,
            tool_calls=tool_calls,
            finish_reason=choice.finish_reason,
            model=completion.model,
            prompt_tokens=usage.prompt_tokens if usage else None,
            completion_tokens=usage.completion_tokens if usage else None,
            duration_s=duration_s,
        )
