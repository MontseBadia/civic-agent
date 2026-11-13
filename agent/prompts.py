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

Only the disease named in the question counts, not its subtypes or \
related diseases. Sensitivity means significance Sensitivity/Response; \
Reduced Sensitivity doesn't count. Any evidence level counts, including \
preclinical. Evidence items for therapies given in combination or in \
sequence (therapy_interaction_type Combination or Sequential) don't count. \
Substitutes means each listed therapy was tested on its own, so each one \
counts.

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
    "C": """\
Use the tools to find the evidence. Do not repeat a call with the same arguments.

Stop calling tools and produce your final answer when any of these conditions is met:

- `get_evidence` returned all rows (`truncated: false`) for the question's molecular profile and disease.
- The gene or variant is not found in CIViC, or the disease cannot be matched to a disease listed by `list_diseases`.
- The gene or variant matches multiple entries that cannot be distinguished from the question.

Do not make additional calls once one of these conditions is met.""",
}

B_RECORDS_INTRO = "These records were retrieved from CIViC for this question:"

# B's first call. Copy only: aliases and notation (Val600Glu) are resolved by
# the lookup code, the same code C's tools use.
B_EXTRACTION_PROMPT = """\
Copy the gene, the variant and the disease named in the question, each \
exactly as written there. Don't correct, expand or translate them. For a \
fusion written as GENE1::GENE2 Fusion (COL1A1::PDGFB Fusion), the gene is \
GENE1::GENE2 and the variant is Fusion."""

C_FINAL_INSTRUCTION = "Now give your answer."


def system_prompt(condition: str) -> str:
    if condition not in CONDITION_PARTS:
        raise ValueError(f"unknown condition: {condition!r}, expected one of {sorted(CONDITION_PARTS)}")
    return f"{CORE}\n\n{CONDITION_PARTS[condition]}"


def prompt_hash() -> str:
    """One hash over every prompt text, so a change anywhere shows in every run log."""
    parts = [CORE, B_RECORDS_INTRO, B_EXTRACTION_PROMPT, C_FINAL_INSTRUCTION]
    parts += [f"{name}:{text}" for name, text in sorted(CONDITION_PARTS.items())]
    return hashlib.sha256("\n\x00\n".join(parts).encode()).hexdigest()[:12]
