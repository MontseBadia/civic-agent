"""Answer format shared by A, B and C, and the parser that decides parseable or not."""

import json
from dataclasses import dataclass

ABSTAIN_REASONS = ("no_evidence", "ambiguous", "unknown_gene")

# Above the 5-drug cap so it doesn't leak the answer size; stops runaway lists.
MAX_LIST_ITEMS = 10

# Property order is generation order under JSON mode: reasoning first, then
# whether to answer, then the lists.
ANSWER_SCHEMA = {
    "type": "object",
    "properties": {
        "gene": {"type": "string"},
        "variant": {"type": "string"},
        "disease": {"type": "string"},
        "explanation": {"type": "string"},
        "abstain": {"type": "boolean"},
        "abstain_reason": {"enum": [*ABSTAIN_REASONS, None]},
        "drugs": {"type": "array", "items": {"type": "string"}, "maxItems": MAX_LIST_ITEMS},
        "evidence_ids": {"type": "array", "items": {"type": "string"}, "maxItems": MAX_LIST_ITEMS},
    },
    "required": ["gene", "variant", "disease", "explanation", "abstain",
                 "abstain_reason", "drugs", "evidence_ids"],
    "additionalProperties": False,
}


class AnswerParseError(ValueError):
    """The model output is not a valid answer. Counts as wrong, no retry."""


@dataclass(frozen=True)
class Answer:
    gene: str
    variant: str
    disease: str
    explanation: str
    abstain: bool
    abstain_reason: str | None
    drugs: tuple[str, ...]
    evidence_ids: tuple[str, ...]


def _string_list(data: dict, key: str) -> tuple[str, ...]:
    value = data[key]
    if not isinstance(value, list) or not all(isinstance(v, str) for v in value):
        raise AnswerParseError(f"{key} must be a list of strings")
    if len(value) > MAX_LIST_ITEMS:
        raise AnswerParseError(f"{key} has {len(value)} items, max {MAX_LIST_ITEMS}")
    return tuple(value)


def parse_answer(text: str | None) -> Answer:
    """Same rules as ANSWER_SCHEMA. Consistency (abstain with drugs) is the scorer's job."""
    if text is None:
        raise AnswerParseError("no content")
    try:
        data = json.loads(text)
    except json.JSONDecodeError as e:
        raise AnswerParseError(f"invalid JSON: {e}") from e
    if not isinstance(data, dict):
        raise AnswerParseError(f"expected an object, got {type(data).__name__}")

    expected = set(ANSWER_SCHEMA["required"])
    if missing := expected - data.keys():
        raise AnswerParseError(f"missing keys: {', '.join(sorted(missing))}")
    if extra := data.keys() - expected:
        raise AnswerParseError(f"unexpected keys: {', '.join(sorted(extra))}")

    for key in ("gene", "variant", "disease", "explanation"):
        if not isinstance(data[key], str):
            raise AnswerParseError(f"{key} must be a string")
    if type(data["abstain"]) is not bool:
        raise AnswerParseError("abstain must be true or false")
    if data["abstain_reason"] not in (*ABSTAIN_REASONS, None):
        raise AnswerParseError(f"abstain_reason not allowed: {data['abstain_reason']!r}")

    return Answer(
        gene=data["gene"],
        variant=data["variant"],
        disease=data["disease"],
        explanation=data["explanation"],
        abstain=data["abstain"],
        abstain_reason=data["abstain_reason"],
        drugs=_string_list(data, "drugs"),
        evidence_ids=_string_list(data, "evidence_ids"),
    )
