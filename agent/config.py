"""LLM endpoint settings from .env, overridable by environment variables."""

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

# The openai client refuses an empty key; a local server ignores it.
LOCAL_API_KEY_PLACEHOLDER = "unused"


@dataclass(frozen=True)
class LLMConfig:
    base_url: str
    api_key: str
    model: str


def parse_env_file(path: Path) -> dict[str, str]:
    """One KEY=VALUE per line. Blank lines and # comments are skipped."""
    values = {}
    for line_no, raw in enumerate(path.read_text().splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        key, sep, value = line.partition("=")
        if not sep:
            raise ValueError(f"{path}:{line_no}: expected KEY=VALUE")
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        values[key.strip()] = value
    return values


def load_config(
    env_path: Path = Path(".env"),
    environ: Mapping[str, str] = os.environ,
) -> LLMConfig:
    """Environment variables win over .env, so one-off runs need no file edit."""
    values = parse_env_file(env_path) if env_path.exists() else {}
    for key in ("LLM_BASE_URL", "LLM_API_KEY", "LLM_MODEL"):
        if environ.get(key):
            values[key] = environ[key]

    missing = [k for k in ("LLM_BASE_URL", "LLM_MODEL") if not values.get(k)]
    if missing:
        raise ValueError(f"missing in .env or environment: {', '.join(missing)}")

    return LLMConfig(
        base_url=values["LLM_BASE_URL"],
        api_key=values.get("LLM_API_KEY") or LOCAL_API_KEY_PLACEHOLDER,
        model=values["LLM_MODEL"],
    )
