"""Run one condition over a question file and write a run log.

python -m evaluation.run --condition B --questions <file.jsonl> --config configs/greedy.json --label dev
"""

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path

import openai

from agent.civic import load_snapshot
from agent.civic_tools import EVIDENCE_CHAR_BUDGET, make_tools
from agent.conditions import run_a, run_b
from agent.config import load_config
from agent.llm import GenerationSettings, LLMClient
from agent.prompts import prompt_hash, system_prompt
from agent.run_log import RunLog, git_state, new_run_path, record_from_error, record_from_loop, run_header
from agent.tool_loop import (MAX_DUPLICATE_CALLS, MAX_RESULT_CHARS, MAX_TOOL_CALLS, LoopResult, Tool,
                             run_tool_loop, tools_hash)

CONDITIONS = ("A", "B", "C")
RUNS_DIR = Path("runs")

# Value types per settings field; bool is refused where a number is expected.
_SETTINGS_TYPES = {"temperature": (int, float), "seed": int, "max_tokens": int, "top_p": (int, float), "top_k": int}
# What the API can't report; must match how the server was started.
SERVER_KEYS = ("mlx_vlm", "prefill_step_size", "hf_hub_offline", "apc_enabled")


class RunConfigError(ValueError):
    pass


@dataclass(frozen=True)
class RunConfig:
    snapshot: Path
    settings: GenerationSettings
    server: dict


@dataclass(frozen=True)
class Question:
    id: str
    text: str


def _exact_keys(data: object, expected, where: str) -> dict:
    if not isinstance(data, dict):
        raise RunConfigError(f"{where}: expected an object")
    if missing := set(expected) - data.keys():
        raise RunConfigError(f"{where}: missing {', '.join(sorted(missing))}")
    if extra := data.keys() - set(expected):
        raise RunConfigError(f"{where}: unexpected {', '.join(sorted(extra))}")
    return data


def load_run_config(path: Path) -> RunConfig:
    """Every value is required: the file is what gets frozen before the test run."""
    data = _exact_keys(json.loads(path.read_text()), ("snapshot", "settings", "server"), str(path))
    settings = _exact_keys(data["settings"], _SETTINGS_TYPES, f"{path}: settings")
    for key, types in _SETTINGS_TYPES.items():
        if isinstance(settings[key], bool) or not isinstance(settings[key], types):
            raise RunConfigError(f"{path}: settings.{key} has the wrong type: {settings[key]!r}")
    return RunConfig(
        snapshot=Path(data["snapshot"]),
        settings=GenerationSettings(**settings),  # also checks ranges
        server=_exact_keys(data["server"], SERVER_KEYS, f"{path}: server"),
    )


def load_questions(path: Path) -> list[Question]:
    """JSONL, one object per line with id and question. Other fields (reference
    answers) are allowed and never reach the model."""
    questions, seen = [], set()
    for line_no, line in enumerate(path.read_text().splitlines(), start=1):
        if not line.strip():
            continue
        data = json.loads(line)
        qid, text = data.get("id"), data.get("question")
        if not isinstance(qid, str) or not qid or not isinstance(text, str) or not text.strip():
            raise ValueError(f"{path}:{line_no}: needs a non-empty string id and question")
        if qid in seen:
            raise ValueError(f"{path}:{line_no}: duplicate id {qid!r}")
        seen.add(qid)
        questions.append(Question(id=qid, text=text))
    if not questions:
        raise ValueError(f"{path}: no questions")
    return questions


def answer_question(condition: str, llm: LLMClient, tools: list[Tool], question: str,
                    settings: GenerationSettings) -> LoopResult:
    if condition == "A":
        return run_a(llm, question, settings)
    if condition == "B":
        return run_b(llm, tools, question, settings)
    if condition == "C":
        messages = [{"role": "system", "content": system_prompt("C")}, {"role": "user", "content": question}]
        return run_tool_loop(llm, messages, tools, settings)
    raise ValueError(f"unknown condition {condition!r}, expected one of {CONDITIONS}")


def _skips_question(error: Exception) -> bool:
    """A failed request (server error status, timeout) costs one question; the run goes on.
    Can't connect at all: the server is down, every question would fail, so stop."""
    return isinstance(error, (openai.APIStatusError, openai.APITimeoutError))


def run_condition(condition: str, questions: list[Question], llm: LLMClient, tools: list[Tool],
                  settings: GenerationSettings, log) -> None:
    for i, q in enumerate(questions, start=1):
        start = time.monotonic()
        try:
            result = answer_question(condition, llm, tools, q.text, settings)
        except Exception as e:
            if not _skips_question(e):
                raise  # our bug or the server is down: fail loudly, the log keeps what was written
            duration = time.monotonic() - start
            log.write({"question_id": q.id, **record_from_error(q.text, condition, e, duration)})
            print(f"{i}/{len(questions)} {q.id} ERROR {type(e).__name__} {duration:.0f}s", flush=True)
            continue
        duration = time.monotonic() - start
        log.write({"question_id": q.id, **record_from_loop(q.text, condition, result, duration)})
        print(f"{i}/{len(questions)} {q.id} {result.stop_reason or '-'} {duration:.0f}s", flush=True)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--condition", required=True, choices=CONDITIONS)
    parser.add_argument("--questions", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--label", required=True, help="e.g. dev, test; goes in the log file name")
    args = parser.parse_args(argv)

    config = load_run_config(args.config)
    questions = load_questions(args.questions)
    llm_config = load_config()
    tools = make_tools(load_snapshot(config.snapshot))  # A gets none; loading also checks the manifest
    label = f"{args.label}_{args.condition}"
    header = run_header(
        label=label,
        llm_config=llm_config,
        settings=config.settings,
        prompt_hash=prompt_hash(),
        # Tool descriptions are prompt text only for C; B calls the functions directly.
        tools_hash=tools_hash(tools) if args.condition == "C" else None,
        manifest_path=config.snapshot / "manifest.json",
        server=config.server,
        git=git_state(Path(".")),
        loop_limits={"max_tool_calls": MAX_TOOL_CALLS, "max_duplicate_calls": MAX_DUPLICATE_CALLS,
                     "max_result_chars": MAX_RESULT_CHARS, "evidence_char_budget": EVIDENCE_CHAR_BUDGET}
        if args.condition == "C" else None,
    )
    header["config_file"] = str(args.config)
    header["questions_file"] = str(args.questions)
    with RunLog(new_run_path(RUNS_DIR, label), header) as log:
        print(f"log: {log.path}", flush=True)
        run_condition(args.condition, questions, LLMClient(llm_config), tools, config.settings, log)
    print(f"done: {log.path}")


if __name__ == "__main__":
    main()
