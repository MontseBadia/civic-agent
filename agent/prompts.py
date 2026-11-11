"""Prompt texts for A, B and C. Edit only on dev; every run logs prompt_hash()."""

import hashlib

# Same task definition for all three conditions; they differ only in where
# the evidence comes from.
CORE = """\
You answer questions about cancer variant evidence in CIViC, a curated \
knowledgebase. This is research software, not clinical advice.

A therapy belongs in drugs only if at least one CIViC evidence item for \
this exact molecular profile and disease supports the significance the \
question asks about (sensitivity or resistance). Evidence items with \
direction "Does Not Support" don't count. Don't add related drugs, drug \
classes or standard treatments without such an item.

Use generic (INN) drug names, not brand names or development codes.

Cite in evidence_ids only IDs that appear in records given to you in this \
conversation, written as EID followed by the number (EID86). If none were \
given, leave it empty.

Abstain (abstain true, empty drugs) when:
- no_evidence: no evidence item supports the requested significance for \
this profile and disease
- unknown_gene: the gene isn't in CIViC
- ambiguous: the variant or disease can't be pinned down

Keep the explanation to three sentences or fewer."""

CONDITION_PARTS = {
    "A": "No records are provided. Answer from what you know about CIViC.",
    "B": "Records retrieved from CIViC for this question are given in the user message.",
    "C": "Use the tools to find the evidence. When done, stop calling tools.",
}

B_RECORDS_INTRO = "These records were retrieved from CIViC for this question:"

C_FINAL_INSTRUCTION = "Now give your answer."


def system_prompt(condition: str) -> str:
    if condition not in CONDITION_PARTS:
        raise ValueError(f"unknown condition: {condition!r}, expected one of {sorted(CONDITION_PARTS)}")
    return f"{CORE}\n\n{CONDITION_PARTS[condition]}"


def prompt_hash() -> str:
    """One hash over every prompt text, so a change anywhere shows in every run log."""
    parts = [CORE, B_RECORDS_INTRO, C_FINAL_INSTRUCTION]
    parts += [f"{name}:{text}" for name, text in sorted(CONDITION_PARTS.items())]
    return hashlib.sha256("\n\x00\n".join(parts).encode()).hexdigest()[:12]
