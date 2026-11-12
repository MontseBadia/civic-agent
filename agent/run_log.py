"""Run log: one JSONL file per run, a header line, then one record per question."""

import hashlib
import json
import subprocess
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import IO

from agent.answer import AnswerParseError, parse_answer
from agent.config import LLMConfig
from agent.llm import GenerationSettings
from agent.tool_loop import LoopResult, ToolStep


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def git_state(repo: Path) -> dict:
    """Commit plus a dirty flag: a run from uncommitted code can't be traced to a commit."""
    def git(*args: str) -> str:
        return subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True, text=True).stdout.strip()

    return {"commit": git("rev-parse", "HEAD"), "dirty": bool(git("status", "--porcelain"))}


def run_header(
    *,
    label: str,
    llm_config: LLMConfig,
    settings: GenerationSettings,
    prompt_hash: str,
    tools_hash: str | None,
    manifest_path: Path,
    server: dict,
    git: dict,
    loop_limits: dict | None,
) -> dict:
    """server: what the API can't report, e.g. mlx-vlm version and --prefill-step-size.
    loop_limits: the tool loop's limits for C (None for conditions without tools)."""
    manifest_bytes = manifest_path.read_bytes()
    return {
        "type": "run",
        "label": label,
        "started_at": _now(),
        "git": git,
        "model": llm_config.model,
        "base_url": llm_config.base_url,  # never the API key
        "settings": asdict(settings),
        "prompt_hash": prompt_hash,
        "tools_hash": tools_hash,
        "loop_limits": loop_limits,
        "snapshot": {
            "release": json.loads(manifest_bytes)["release"],
            "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        },
        "server": server,
    }


def _retrieved_ids(result: LoopResult) -> list[str]:
    ids = []
    for step in result.trace:
        if isinstance(step, ToolStep) and step.outcome == "ok" and isinstance(step.result, dict):
            ids += [row["evidence_id"] for row in step.result.get("rows", [])]
    return list(dict.fromkeys(ids))  # unique, first-seen order


def record_from_loop(question: str, condition: str, result: LoopResult, duration_s: float) -> dict:
    if result.final is not None:
        answered_by = "model"
        try:
            answer, parse_error = asdict(parse_answer(result.final.content)), None
        except AnswerParseError as e:
            answer, parse_error = None, str(e)
    elif result.code_answer is not None:
        answered_by, answer, parse_error = "code", asdict(result.code_answer), None
    else:
        # B stopped before any answer existed (extraction failed): wrong, like unparseable.
        answered_by, answer, parse_error = None, None, f"no answer: stopped at {result.stop_reason}"
    return {
        "type": "question",
        "question": question,
        "condition": condition,
        "duration_s": duration_s,
        "stop_reason": result.stop_reason,
        "answered_by": answered_by,
        "final_content": result.final.content if result.final is not None else None,
        "answer": answer,
        "parse_error": parse_error,
        "retrieved_ids": _retrieved_ids(result),
        "trace": [{"kind": type(s).__name__, **asdict(s)} for s in result.trace],
        "messages": list(result.messages),
        "error": None,
    }


def record_from_error(question: str, condition: str, error: Exception, duration_s: float) -> dict:
    """The run continues; the question counts as wrong."""
    return {
        "type": "question",
        "question": question,
        "condition": condition,
        "duration_s": duration_s,
        "error": {
            "type": type(error).__name__,
            "message": str(error),
            "server_body": getattr(error, "body", None),  # e.g. Metal out of memory
        },
    }


class RunLog:
    """Appends one JSON line per record and flushes, so a crash keeps what was written."""

    def __init__(self, path: Path, header: dict):
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        self._file: IO[str] = path.open("x")  # never overwrite an earlier run
        self.write(header)

    def write(self, record: dict) -> None:
        self._file.write(json.dumps(record) + "\n")  # no default=: unserializable values raise
        self._file.flush()

    def close(self) -> None:
        self._file.close()

    def __enter__(self) -> "RunLog":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def new_run_path(runs_dir: Path, label: str) -> Path:
    return runs_dir / f"{datetime.now(timezone.utc):%Y%m%d-%H%M%S}_{label}.jsonl"
