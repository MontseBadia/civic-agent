"""Score run logs against the question files.

python -m evaluation.score --questions <q.jsonl> [...] --runs <log.jsonl> [...] --snapshot data/raw/01-Nov-2026 --out <dir>

The snapshot is read with the generator's reader, not the agent's, so scoring
doesn't depend on the code being scored.
"""

import argparse
import json
import re
import statistics
from collections import Counter
from math import comb
from pathlib import Path

from agent.run_log import git_state
from evaluation.generate import INTERACTIONS, read_snapshot, split_therapies

ABSTAIN_REASONS = ("no_evidence", "ambiguous", "unknown_gene")
SIGNIFICANCE = {"T1": "Sensitivity/Response", "T2": "Resistance"}
# A comparison across these would be meaningless, and on test it would break the freeze.
MUST_MATCH = ("model", "settings", "prompt_hash")
QUESTION_KEYS = ("id", "kind", "template", "profile_id", "disease", "reference_drugs",
                 "reference_evidence_ids", "expected_abstain", "expected_abstain_reason")
PAIRS = (("B", "A"), ("C", "B"))  # B vs A: value of evidence. C vs B: value of agency.
_EID = re.compile(r"(?i)\s*(?:eid\s*)?(\d+)\s*")


class ScoreError(ValueError):
    pass


# --- loading ---

def load_questions(paths: list[Path]) -> list[dict]:
    questions, seen = [], set()
    for path in paths:
        for line_no, line in enumerate(path.read_text().splitlines(), start=1):
            if not line.strip():
                continue
            q = json.loads(line)
            where = f"{path}:{line_no}"
            if missing := [k for k in QUESTION_KEYS if k not in q]:
                raise ScoreError(f"{where}: missing {', '.join(missing)}")
            if q["id"] in seen:
                raise ScoreError(f"{where}: duplicate id {q['id']!r}")
            if q["expected_abstain_reason"] not in (None, *ABSTAIN_REASONS):
                raise ScoreError(f"{where}: abstain reason {q['expected_abstain_reason']!r}")
            if q["expected_abstain"] != (q["expected_abstain_reason"] is not None):
                raise ScoreError(f"{where}: expected_abstain and expected_abstain_reason disagree")
            seen.add(q["id"])
            questions.append(q)
    return questions


def load_run(path: Path, question_ids: list[str]) -> tuple[dict, str, dict[str, dict]]:
    """Header, condition, records by question id. Refuses incomplete or mixed logs."""
    lines = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if not lines or lines[0].get("type") != "run":
        raise ScoreError(f"{path}: first line is not a run header")
    header, records = lines[0], {}
    conditions = {r["condition"] for r in lines[1:]}
    if len(conditions) != 1:
        raise ScoreError(f"{path}: expected one condition, found {sorted(conditions)}")
    for r in lines[1:]:
        if r["question_id"] in records:
            raise ScoreError(f"{path}: two records for {r['question_id']!r}")
        records[r["question_id"]] = r
    if missing := [q for q in question_ids if q not in records]:
        raise ScoreError(f"{path}: incomplete run, no record for {', '.join(missing)}")
    if extra := sorted(records.keys() - set(question_ids)):
        raise ScoreError(f"{path}: records for questions not in the question files: {', '.join(extra)}")
    return header, conditions.pop(), records


def check_comparable(headers: dict[str, dict], snapshot_sha: str) -> None:
    for condition, h in headers.items():
        if h["snapshot"]["manifest_sha256"] != snapshot_sha:
            raise ScoreError(f"{condition}: run used another snapshot than --snapshot")
    first = next(iter(headers))
    for condition, h in headers.items():
        for key in MUST_MATCH:
            if h[key] != headers[first][key]:
                raise ScoreError(f"{key} differs between {first} and {condition}: "
                                 f"{headers[first][key]!r} vs {h[key]!r}")


# --- per question ---

def normalize_id(cited: str) -> str:
    """EID86, eid 86 and 86 all become EID86. Anything else is kept and can't match."""
    match = _EID.fullmatch(cited)
    return f"EID{match[1]}" if match else cited.strip()


def _matches_question(row: dict | None, q: dict) -> bool:
    return (row is not None and q["profile_id"] is not None
            and row["molecular_profile_id"] == str(q["profile_id"]) and row["disease"] == q["disease"])


def _lookup_ok(record: dict, q: dict) -> bool | None:
    """Did any get_evidence call reach the question's profile and disease? None: no profile."""
    if q["profile_id"] is None:
        return None
    for step in record.get("trace") or []:
        if step["kind"] == "ToolStep" and step["name"] == "get_evidence" and step["outcome"] == "ok":
            args = json.loads(step["raw_arguments"])
            if (args.get("molecular_profile_id") == q["profile_id"]
                    and str(args.get("disease", "")).strip().casefold() == q["disease"].casefold()):
                return True
    return False


def score_question(q: dict, record: dict, evidence: dict[str, dict]) -> dict:
    """evidence: snapshot rows by evidence_id (no prefix)."""
    if record.get("error") is not None:
        status = "error"
    elif record.get("answer") is None:
        status = "unparseable"
    else:
        status = "answered"
    answer = record.get("answer") or {}
    given = sorted({d.strip().lower() for d in answer.get("drugs", []) if d.strip()})
    cited = sorted({normalize_id(c) for c in answer.get("evidence_ids", [])})
    reference = set(q["reference_drugs"])
    reference_ids = set(q["reference_evidence_ids"])
    abstained = bool(answer.get("abstain"))

    if status != "answered":
        correct = False
    elif q["expected_abstain"]:
        correct = abstained and answer.get("abstain_reason") == q["expected_abstain_reason"] and not given
    else:
        correct = not abstained and set(given) == reference

    rows = {c: evidence.get(c.removeprefix("EID")) for c in cited}
    # A cited Does Not Support row for this profile and disease doesn't count for
    # or against precision: citing it to say no effect was found isn't wrong.
    counted = [c for c in cited
               if not (_matches_question(rows[c], q) and rows[c]["evidence_direction"] == "Does Not Support")]

    def supported(drug: str) -> bool:
        return any(_matches_question(row, q) and row["evidence_type"] == "Predictive"
                   and row["evidence_direction"] == "Supports"
                   and row["significance"] == SIGNIFICANCE[q["template"]]
                   and row["therapy_interaction_type"] in INTERACTIONS
                   and drug in {t.lower() for t in split_therapies(row["therapies"])}
                   for row in rows.values())

    retrieved = set(record.get("retrieved_ids") or [])
    trace = record.get("trace") or []
    return {
        "question_id": q["id"],
        "condition": record["condition"],
        "kind": q["kind"],
        "group": q.get("group"),
        "status": status,
        "correct": correct,
        "expected_abstain": q["expected_abstain"],
        "abstained": abstained,
        "abstain_reason": answer.get("abstain_reason"),
        "expected_abstain_reason": q["expected_abstain_reason"],
        "drugs_given": given,
        "drugs_right": sorted(set(given) & reference),
        "drugs_missing": sorted(reference - set(given)),
        "drugs_extra": sorted(set(given) - reference),
        "drugs_supported": [d for d in given if supported(d)],
        "reference_drugs": sorted(reference),
        "cited": cited,
        "cited_counted": counted,
        "cited_right": sorted(set(counted) & reference_ids),
        "cited_valid": [c for c in cited if _matches_question(rows[c], q)],
        "cited_invented": [c for c in cited if rows[c] is None],
        "reference_ids": sorted(reference_ids),
        "reference_retrieved": sorted(reference_ids & retrieved),
        "lookup_ok": _lookup_ok(record, q),
        "stop_reason": record.get("stop_reason"),
        "parse_error": record.get("parse_error"),
        "error": record.get("error"),
        "duration_s": record["duration_s"],
        "model_calls": sum(s["kind"] == "ModelStep" for s in trace) if trace else None,
        "tool_calls": sum(s["kind"] == "ToolStep" for s in trace) if trace else None,
        "prompt_tokens": sum(s["prompt_tokens"] or 0 for s in trace if s["kind"] == "ModelStep") if trace else None,
        "completion_tokens": (sum(s["completion_tokens"] or 0 for s in trace if s["kind"] == "ModelStep")
                              if trace else None),
        # Only the mechanical tag; every other failure tag is assigned by hand.
        "failure_tag": "format_error" if status == "unparseable" else None,
    }


# --- summaries ---

def _ratio(num: int, den: int) -> float | None:
    return num / den if den else None


def summarize(rows: list[dict]) -> dict:
    """Micro-averaged: pool drugs or IDs across questions, divide once."""
    n = len(rows)
    answerable = [r for r in rows if not r["expected_abstain"]]
    answered = [r for r in rows if r["status"] == "answered"]
    count = lambda key: sum(len(r[key]) for r in rows)  # noqa: E731
    summary = {
        "n": n,
        "correct": sum(r["correct"] for r in rows),
        "accuracy": _ratio(sum(r["correct"] for r in rows), n),
        "errors": sum(r["status"] == "error" for r in rows),
        "unparseable": sum(r["status"] == "unparseable" for r in rows),
        # Drugs given where an abstention was expected count as false positives.
        "drug_precision": _ratio(count("drugs_right"), count("drugs_given")),
        "drug_recall": _ratio(sum(len(r["drugs_right"]) for r in answerable),
                              sum(len(r["reference_drugs"]) for r in answerable)),
        # Answered questions only: errors and unparseable answers are counted above, not twice.
        "missed_abstentions": sum(r["expected_abstain"] and not r["abstained"] for r in answered),
        "false_abstentions": sum(not r["expected_abstain"] and r["abstained"] for r in answered),
        "wrong_abstain_reason": sum(r["expected_abstain"] and r["abstained"]
                                    and r["abstain_reason"] != r["expected_abstain_reason"] for r in answered),
        "cited_precision": _ratio(count("cited_right"), count("cited_counted")),
        "cited_recall": _ratio(count("cited_right"), count("reference_ids")),
        "citation_validity": _ratio(count("cited_valid"), count("cited")),
        "invented_citations": _ratio(count("cited_invented"), count("cited")),
        "claim_support": _ratio(count("drugs_supported"), count("drugs_given")),
        "retrieved_recall": _ratio(count("reference_retrieved"), count("reference_ids")),
        "lookup_accuracy": _ratio(sum(r["lookup_ok"] is True for r in rows),
                                  sum(r["lookup_ok"] is not None for r in rows)),
    }
    durations = [r["duration_s"] for r in rows]
    traced = [r for r in rows if r["model_calls"] is not None]
    summary["cost"] = {
        "mean_s": statistics.mean(durations) if durations else None,
        "median_s": statistics.median(durations) if durations else None,
        **{f"mean_{k}": (statistics.mean(r[k] for r in traced) if traced else None)
           for k in ("model_calls", "tool_calls", "prompt_tokens", "completion_tokens")},
    }
    return summary


def mcnemar(first: list[bool], second: list[bool]) -> dict:
    """Exact two-sided McNemar on paired correctness: binomial test on the discordant pairs."""
    b = sum(f and not s for f, s in zip(first, second, strict=True))  # first right, second wrong
    c = sum(s and not f for f, s in zip(first, second, strict=True))
    n = b + c
    p = 1.0 if n == 0 else min(1.0, 2 * sum(comb(n, k) for k in range(min(b, c) + 1)) / 2 ** n)
    return {"first_only": b, "second_only": c, "p_value": p}


def split_of(row: dict) -> str:
    return "hand" if row["kind"] == "hand" else "templated"


def build_report(rows: list[dict], conditions: list[str]) -> dict:
    report = {"by_condition": {}, "by_kind": {}, "mcnemar": {}}
    for condition in conditions:
        mine = [r for r in rows if r["condition"] == condition]
        report["by_condition"][condition] = {
            split: summarize([r for r in mine if split_of(r) == split]) for split in ("templated", "hand")
            if any(split_of(r) == split for r in mine)}
        kinds = Counter(r["group"] or r["kind"] for r in mine)
        report["by_kind"][condition] = {
            k: summarize([r for r in mine if (r["group"] or r["kind"]) == k])["accuracy"] for k in sorted(kinds)}
    for second, first in PAIRS:
        if first in conditions and second in conditions:
            for split in ("templated", "hand"):
                ids = sorted({r["question_id"] for r in rows if split_of(r) == split})
                if ids:
                    correct = {(r["condition"], r["question_id"]): r["correct"] for r in rows}
                    report["mcnemar"][f"{second} vs {first}, {split}"] = mcnemar(
                        [correct[(first, i)] for i in ids], [correct[(second, i)] for i in ids])
    return report


def print_report(report: dict) -> None:
    fmt = lambda v: "-" if v is None else f"{v:.2f}" if isinstance(v, float) else str(v)  # noqa: E731
    keys = ["n", "correct", "accuracy", "errors", "unparseable", "drug_precision", "drug_recall",
            "missed_abstentions", "false_abstentions", "wrong_abstain_reason", "cited_precision",
            "cited_recall", "citation_validity", "invented_citations", "claim_support",
            "retrieved_recall", "lookup_accuracy"]
    for split in ("templated", "hand"):
        conditions = [c for c, s in report["by_condition"].items() if split in s]
        if not conditions:
            continue
        print(f"\n== {split}")
        print(f"{'':22}" + "".join(f"{c:>9}" for c in conditions))
        for key in keys:
            print(f"{key:22}" + "".join(f"{fmt(report['by_condition'][c][split][key]):>9}" for c in conditions))
        for key in ("mean_s", "median_s", "mean_model_calls", "mean_tool_calls", "mean_prompt_tokens",
                    "mean_completion_tokens"):
            print(f"{key:22}" + "".join(f"{fmt(report['by_condition'][c][split]['cost'][key]):>9}"
                                        for c in conditions))
    print("\n== accuracy by kind / group")
    for condition, kinds in report["by_kind"].items():
        print(f"{condition}: " + ", ".join(f"{k} {fmt(v)}" for k, v in kinds.items()))
    print("\n== McNemar (exact)")
    for name, m in report["mcnemar"].items():
        print(f"{name}: only first right {m['first_only']}, only second right {m['second_only']}, "
              f"p={m['p_value']:.3f}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--questions", required=True, type=Path, nargs="+")
    parser.add_argument("--runs", required=True, type=Path, nargs="+", help="one log per condition")
    parser.add_argument("--snapshot", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args(argv)

    outputs = [args.out / "per_question.jsonl", args.out / "summary.json"]
    if existing := [str(p) for p in outputs if p.exists()]:
        raise SystemExit(f"already exists, not overwriting: {', '.join(existing)}")
    questions = load_questions(args.questions)
    ids = [q["id"] for q in questions]
    snapshot = read_snapshot(args.snapshot)
    evidence = {r["evidence_id"]: r for r in snapshot.evidence}
    headers, records, files = {}, {}, {}
    for path in args.runs:
        header, condition, recs = load_run(path, ids)
        if condition in headers:
            raise ScoreError(f"two runs for condition {condition}")
        headers[condition], records[condition], files[condition] = header, recs, path
    check_comparable(headers, snapshot.manifest_sha256)

    conditions = sorted(headers)
    rows = [score_question(q, records[c][q["id"]], evidence) for c in conditions for q in questions]
    report = build_report(rows, conditions)
    args.out.mkdir(parents=True, exist_ok=True)
    with outputs[0].open("x") as f:
        f.writelines(json.dumps(r) + "\n" for r in rows)
    with outputs[1].open("x") as f:
        json.dump({"scorer_git": git_state(Path(".")), "questions": [str(p) for p in args.questions],
                   "runs": {c: {"file": str(files[c]), "label": headers[c]["label"], "git": headers[c]["git"],
                                "prompt_hash": headers[c]["prompt_hash"]} for c in conditions},
                   **report}, f, indent=2)
    print_report(report)
    print(f"\nwritten: {', '.join(map(str, outputs))}")


if __name__ == "__main__":
    main()
