"""Conditions A (closed-book) and B (fixed pipeline). C is run_tool_loop over make_tools."""

import json
from dataclasses import dataclass

from agent.answer import ANSWER_SCHEMA, Answer
from agent.civic import normalize_gene, normalize_variant
from agent.llm import GenerationSettings, LLMClient, ToolCall
from agent.prompts import B_EXTRACTION_PROMPT, B_RECORDS_INTRO, system_prompt
from agent.tool_loop import LoopResult, ModelStep, Tool, ToolStep, model_step, run_call

EXTRACTION_SCHEMA = {
    "type": "object",
    "properties": {
        "gene": {"type": "string"},
        "variant": {"type": "string"},
        "disease": {"type": "string"},
    },
    "required": ["gene", "variant", "disease"],
    "additionalProperties": False,
}


class ExtractionParseError(ValueError):
    """B's extraction output is unusable. The question counts as wrong, no retry."""


@dataclass(frozen=True)
class Extraction:
    gene: str
    variant: str
    disease: str


def parse_extraction(text: str | None) -> Extraction:
    if text is None:
        raise ExtractionParseError("no content")
    try:
        data = json.loads(text)
    except json.JSONDecodeError as e:
        raise ExtractionParseError(f"invalid JSON: {e}") from e
    expected = EXTRACTION_SCHEMA["required"]
    if not isinstance(data, dict) or sorted(data) != sorted(expected):
        raise ExtractionParseError(f"expected an object with keys {', '.join(expected)}")
    for key in expected:
        # Empty text can't be looked up; the tools would refuse it.
        if not isinstance(data[key], str) or not data[key].strip():
            raise ExtractionParseError(f"{key} must be a non-empty string")
    return Extraction(**data)


def _prefer_exact_name(matches: list[dict], query: str, normalize) -> list[dict]:
    """Of several matches, the one whose own name is the query wins over alias matches.

    In CIViC a symbol can also be another gene's alias (AR is an alias of AREG) and a
    variant name another variant's alias (D835Y of D835H/Y). Still ambiguous when no
    name, or more than one, equals the query.
    """
    if len(matches) <= 1:
        return matches
    exact = [m for m in matches if normalize(m["name"]) == normalize(query)]
    return exact if len(exact) == 1 else matches


def run_a(llm: LLMClient, question: str, settings: GenerationSettings) -> LoopResult:
    messages = [
        {"role": "system", "content": system_prompt("A")},
        {"role": "user", "content": question},
    ]
    final = llm.chat(messages, settings, response_schema=ANSWER_SCHEMA)
    return LoopResult(final=final, code_answer=None, trace=(model_step(final, final=True),),
                      messages=tuple(messages), stop_reason=None)


def run_b(llm: LLMClient, tools: list[Tool], question: str, settings: GenerationSettings) -> LoopResult:
    """Extract, look up with C's tool functions, then synthesize from the rows.

    stop_reason says which step ended the question: extraction_failed, unknown_gene,
    ambiguous_gene, no_variant, ambiguous_variant, no_rows or synthesis.
    messages holds both conversations one after the other, each starting with its
    system message: extraction (with the model's reply), then synthesis.
    """
    by_name = {t.name: t for t in tools}
    trace: list[ModelStep | ToolStep] = []

    extraction_messages = [
        {"role": "system", "content": B_EXTRACTION_PROMPT},
        {"role": "user", "content": question},
    ]
    response = llm.chat(extraction_messages, settings, response_schema=EXTRACTION_SCHEMA)
    trace.append(model_step(response, final=False))
    messages = [*extraction_messages, {"role": "assistant", "content": response.content}]

    def stop(reason: str, answer: Answer | None) -> LoopResult:
        return LoopResult(final=None, code_answer=answer, trace=tuple(trace),
                          messages=tuple(messages), stop_reason=reason)

    try:
        extracted = parse_extraction(response.content)
    except ExtractionParseError:
        return stop("extraction_failed", None)  # raw reply is in messages

    def lookup(name: str, **arguments) -> dict:
        call = ToolCall(id=f"b{len(trace)}", name=name, arguments=json.dumps(arguments))
        step = run_call(call, by_name)
        trace.append(step)
        if step.outcome != "ok":
            # B only sends non-empty extracted text and IDs from earlier results,
            # so a refused lookup is our bug, not the model's.
            raise RuntimeError(f"B lookup {name}({arguments}) failed: {step.content}")
        return step.result

    def abstain(reason: str, abstain_reason: str, explanation: str) -> LoopResult:
        answer = Answer(gene=extracted.gene, variant=extracted.variant, disease=extracted.disease,
                        explanation=explanation, abstain=True, abstain_reason=abstain_reason,
                        drugs=(), evidence_ids=())
        return stop(reason, answer)

    genes = _prefer_exact_name(lookup("search_gene", gene=extracted.gene)["matches"],
                               extracted.gene, normalize_gene)
    if not genes:
        return abstain("unknown_gene", "unknown_gene", f"{extracted.gene} is not a gene in CIViC.")
    if len(genes) > 1:
        names = ", ".join(g["name"] for g in genes)
        return abstain("ambiguous_gene", "ambiguous", f"{extracted.gene} matches several CIViC genes: {names}.")
    gene = genes[0]

    variants = _prefer_exact_name(
        lookup("search_variant", gene_id=gene["gene_id"], variant=extracted.variant)["matches"],
        extracted.variant, normalize_variant)
    if not variants:
        return abstain("no_variant", "no_evidence", f"CIViC has no {gene['name']} variant {extracted.variant}.")
    if len(variants) > 1:
        names = ", ".join(v["name"] for v in variants)
        return abstain("ambiguous_variant", "ambiguous",
                       f"{extracted.variant} matches several {gene['name']} variants: {names}.")
    profile = variants[0]

    evidence = lookup("get_evidence", molecular_profile_id=profile["molecular_profile_id"],
                      disease=extracted.disease)
    if evidence["truncated"]:
        # B must see every row. Every profile + disease pair fits in the 01-Nov-2026 snapshot.
        raise RuntimeError(f"get_evidence truncated for {profile['molecular_profile_name']} / {extracted.disease}")
    if not evidence["rows"]:
        # Exact disease name only: no rows can also mean the question didn't use CIViC's name.
        return abstain("no_rows", "no_evidence",
                       f"No CIViC evidence rows for {profile['molecular_profile_name']} "
                       f"under the disease name '{extracted.disease}'.")

    synthesis_messages = [
        {"role": "system", "content": system_prompt("B")},
        {"role": "user", "content": f"{question}\n\n{B_RECORDS_INTRO}\n{json.dumps(evidence['rows'])}"},
    ]
    messages += synthesis_messages
    final = llm.chat(synthesis_messages, settings, response_schema=ANSWER_SCHEMA)
    trace.append(model_step(final, final=True))
    return LoopResult(final=final, code_answer=None, trace=tuple(trace),
                      messages=tuple(messages), stop_reason="synthesis")
