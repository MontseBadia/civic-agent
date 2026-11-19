"""Ask one CIViC question from the command line. Research software, not clinical advice.

Same conditions, settings and snapshot as the evaluation. Every question is logged to runs/cli/.
"""

import argparse
import re
import sys
import time
from pathlib import Path

import openai

from agent.civic import CivicSnapshot, load_snapshot
from agent.civic_tools import EVIDENCE_CHAR_BUDGET, make_tools
from agent.config import load_config
from agent.llm import LLMClient
from agent.prompts import prompt_hash
from agent.run_log import RunLog, git_state, new_run_path, record_from_error, record_from_loop, run_header
from agent.tool_loop import MAX_DUPLICATE_CALLS, MAX_RESULT_CHARS, MAX_TOOL_CALLS, tools_hash
# Imported, not moved: the evaluated code path stays as it was for the test run.
from evaluation.run import CONDITIONS, answer_question, load_run_config

RUNS_DIR = Path("runs/cli")
DISCLAIMER = "Research software. Not clinical advice."
# Same pattern as the snapshot's evidence_civic_url column.
EVIDENCE_URL = "https://civicdb.org/links/evidence_items/{}"
TABLE_COLUMNS = ("EID", "Disease", "Type", "Significance", "Direction", "Level", "Citation", "Link")
_EVIDENCE_ID = re.compile(r"EID(\d+)")


def evidence_rows(cited: list[str], snapshot: CivicSnapshot) -> list[tuple[str, ...]]:
    """One row per cited ID, values straight from the snapshot. No model in this step."""
    rows = []
    for cited_id in cited:
        match = _EVIDENCE_ID.fullmatch(cited_id)
        evidence = snapshot.evidence(int(match.group(1))) if match else None
        if evidence is None:
            rows.append((cited_id, "not in snapshot"))
            continue
        rows.append((
            cited_id, evidence.disease, evidence.evidence_type, evidence.significance,
            evidence.evidence_direction, evidence.evidence_level,
            f"{evidence.citation} ({evidence.citation_id})", EVIDENCE_URL.format(evidence.id),
        ))
    return rows


def format_table(rows: list[tuple[str, ...]]) -> list[str]:
    lines = [TABLE_COLUMNS, *rows]
    widths = [max(len(line[i]) for line in lines if i < len(line)) for i in range(len(TABLE_COLUMNS))]
    return ["  ".join(value.ljust(widths[i]) for i, value in enumerate(line)).rstrip() for line in lines]


def render(record: dict, snapshot: CivicSnapshot, log_path: Path) -> str:
    """Printed from the logged record, so the screen shows what the log holds."""
    lines = [DISCLAIMER, ""]
    answer = record["answer"]
    if answer is None:
        lines.append(f"No usable answer: {record['parse_error']}")
        if record["final_content"] is not None:
            lines += ["Raw model output:", record["final_content"]]
    else:
        if answer["abstain"]:
            lines.append(f"Abstained: {answer['abstain_reason']}")
        if answer["drugs"]:
            lines.append(f"Drugs: {', '.join(answer['drugs'])}")
        lines.append(f"Explanation: {answer['explanation']}")
        if answer["evidence_ids"]:
            lines += ["", "Evidence cited", *format_table(evidence_rows(list(answer["evidence_ids"]), snapshot))]
    tool_calls = sum(step["kind"] == "ToolStep" for step in record["trace"])
    lines += ["", f"{record['condition']}, {tool_calls} tool calls, stop: {record['stop_reason'] or '-'}, "
                  f"{record['duration_s']:.0f} s. Log: {log_path}"]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--condition", required=True, choices=CONDITIONS)
    parser.add_argument("--config", required=True, type=Path, help="run settings, e.g. configs/greedy.json")
    parser.add_argument("question")
    args = parser.parse_args(argv)
    if not args.question.strip():
        parser.error("the question is empty")

    config = load_run_config(args.config)
    llm_config = load_config()
    snapshot = load_snapshot(config.snapshot)  # also checks the manifest
    tools = make_tools(snapshot)
    header = run_header(
        label=f"cli_{args.condition}",
        llm_config=llm_config,
        settings=config.settings,
        prompt_hash=prompt_hash(),
        tools_hash=tools_hash(tools) if args.condition == "C" else None,
        manifest_path=config.snapshot / "manifest.json",
        server=config.server,
        git=git_state(Path(".")),
        loop_limits={"max_tool_calls": MAX_TOOL_CALLS, "max_duplicate_calls": MAX_DUPLICATE_CALLS,
                     "max_result_chars": MAX_RESULT_CHARS, "evidence_char_budget": EVIDENCE_CHAR_BUDGET}
        if args.condition == "C" else None,
    )
    header["config_file"] = str(args.config)

    with RunLog(new_run_path(RUNS_DIR, f"cli_{args.condition}"), header) as log:
        start = time.monotonic()
        try:
            result = answer_question(args.condition, LLMClient(llm_config), tools, args.question, config.settings)
        except openai.APIError as e:  # server error, timeout, can't connect; anything else is a bug and raises
            log.write(record_from_error(args.question, args.condition, e, time.monotonic() - start))
            print(f"{DISCLAIMER}\n\nRequest failed: {type(e).__name__}: {e}\nLog: {log.path}", file=sys.stderr)
            return 1
        record = record_from_loop(args.question, args.condition, result, time.monotonic() - start)
        log.write(record)
    print(render(record, snapshot, log.path))
    return 0 if record["answer"] is not None else 1


if __name__ == "__main__":
    sys.exit(main())
