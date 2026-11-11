"""The four CIViC tools condition C can call, over the snapshot lookups B uses directly."""

import json

from agent.civic import CivicSnapshot
from agent.tool_loop import Tool, ToolError

# A character budget, not a row count: rows run from ~800 to ~3,700 characters.
# Every profile + disease result in the 01-Nov-2026 snapshot fits (largest 29,979).
EVIDENCE_CHAR_BUDGET = 35_000


def _require_text(name: str, value: str) -> str:
    if not value.strip():
        raise ToolError(f"{name} must not be empty")
    return value


def _profile(snapshot: CivicSnapshot, profile_id: int):
    try:
        return snapshot.profile_variant(profile_id)
    except LookupError as e:
        raise ToolError(str(e)) from e


def make_tools(snapshot: CivicSnapshot) -> list[Tool]:
    def search_gene(gene: str) -> dict:
        _require_text("gene", gene)
        return {
            "query": gene,
            "matches": [
                {"gene_id": f.id, "name": f.name, "feature_type": f.feature_type, "aliases": list(f.aliases)}
                for f in snapshot.find_genes(gene)
            ],
        }

    def search_variant(gene_id: int, variant: str) -> dict:
        _require_text("variant", variant)
        try:
            feature = snapshot.feature(gene_id)
        except KeyError:
            raise ToolError(f"no gene_id {gene_id}, use search_gene first") from None
        return {
            "gene_id": gene_id,
            "gene": feature.name,
            "query": variant,
            "matches": [
                {
                    "variant_id": v.id,
                    "name": v.name,
                    "molecular_profile_id": v.profile_id,
                    "molecular_profile_name": v.profile_name,
                    "aliases": list(v.aliases),
                }
                for v in snapshot.find_variants(gene_id, variant)
            ],
        }

    def list_diseases(molecular_profile_id: int) -> dict:
        variant = _profile(snapshot, molecular_profile_id)
        return {
            "molecular_profile_id": molecular_profile_id,
            "molecular_profile_name": variant.profile_name,
            "diseases": [
                {"disease": d, "rows": n} for d, n in snapshot.diseases_for(molecular_profile_id).items()
            ],
        }

    def get_evidence(molecular_profile_id: int, disease: str | None = None) -> dict:
        variant = _profile(snapshot, molecular_profile_id)
        if disease is not None:
            _require_text("disease", disease)
        rows = [e.to_record() for e in snapshot.evidence_for(molecular_profile_id, disease)]
        returned, used = [], 0
        for record in rows:
            size = len(json.dumps(record)) + 2  # separator
            if used + size > EVIDENCE_CHAR_BUDGET:
                break
            returned.append(record)
            used += size
        return {
            "molecular_profile_id": molecular_profile_id,
            "molecular_profile_name": variant.profile_name,
            "disease": disease,
            "total": len(rows),
            "returned": len(returned),
            "truncated": len(returned) < len(rows),
            "rows": returned,
        }

    return [
        Tool(
            name="search_gene",
            description=(
                "Find CIViC genes by symbol or alias, case-insensitive. Returns every match: "
                "none means the gene isn't in CIViC, several means the name is ambiguous."
            ),
            parameters={
                "type": "object",
                "properties": {"gene": {"type": "string", "description": "Gene symbol or alias, e.g. BRAF"}},
                "required": ["gene"],
            },
            function=search_gene,
        ),
        Tool(
            name="search_variant",
            description=(
                "Find variants of a gene by name, alias or protein change (V600E, Val600Glu, "
                "p.Val600Glu). Returns every match with its molecular_profile_id."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "gene_id": {"type": "integer", "description": "gene_id from search_gene"},
                    "variant": {"type": "string", "description": "Variant as written, e.g. V600E"},
                },
                "required": ["gene_id", "variant"],
            },
            function=search_variant,
        ),
        Tool(
            name="list_diseases",
            description="Diseases that have CIViC evidence for a molecular profile, with row counts.",
            parameters={
                "type": "object",
                "properties": {
                    "molecular_profile_id": {"type": "integer", "description": "From search_variant"},
                },
                "required": ["molecular_profile_id"],
            },
            function=list_diseases,
        ),
        Tool(
            name="get_evidence",
            description=(
                "CIViC evidence items for a molecular profile, optionally for one disease "
                "(exact CIViC disease name, any case). All types and directions are returned. "
                "If truncated is true, not all rows fit: narrow by disease."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "molecular_profile_id": {"type": "integer", "description": "From search_variant"},
                    "disease": {"type": "string", "description": "Exact CIViC disease name, from list_diseases"},
                },
                "required": ["molecular_profile_id"],
            },
            function=get_evidence,
        ),
    ]
