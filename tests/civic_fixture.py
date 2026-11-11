"""Made-up CIViC snapshot written to a temp dir, manifest included. Not real evidence."""

import csv
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

RELEASE = "01-Jan-2000"

FEATURE_COLUMNS = ["feature_id", "name", "feature_type", "feature_aliases"]
VARIANT_COLUMNS = ["variant_id", "feature_id", "feature_name", "variant", "variant_aliases",
                   "hgvs_descriptions", "single_variant_molecular_profile_id"]
EVIDENCE_COLUMNS = ["evidence_id", "molecular_profile", "molecular_profile_id", "disease", "therapies",
                    "therapy_interaction_type", "evidence_type", "evidence_direction", "evidence_level",
                    "significance", "evidence_statement", "citation_id", "citation", "rating",
                    "variant_origin", "is_flagged"]

FEATURES = [
    {"feature_id": "1", "name": "GENEA", "feature_type": "Gene", "feature_aliases": "GA1,SHARED"},
    {"feature_id": "2", "name": "GENEB", "feature_type": "Gene", "feature_aliases": "SHARED"},
]

VARIANTS = [
    {"variant_id": "101", "feature_id": "1", "feature_name": "GENEA", "variant": "V600E",
     "variant_aliases": "VAL600GLU,RS1",
     "hgvs_descriptions": "NP_000001.1:p.Val600Glu,NC_000001.11:g.100A>T",
     "single_variant_molecular_profile_id": "1001"},
    {"variant_id": "102", "feature_id": "1", "feature_name": "GENEA", "variant": "K601E",
     "variant_aliases": "RS1", "hgvs_descriptions": "", "single_variant_molecular_profile_id": "1002"},
    {"variant_id": "103", "feature_id": "2", "feature_name": "GENEB", "variant": "Amplification",
     "variant_aliases": "", "hgvs_descriptions": "", "single_variant_molecular_profile_id": "1003"},
]


def evidence_row(eid, profile_id, disease, therapies="", interaction="", etype="Predictive",
                 direction="Supports", significance="Sensitivity/Response", flagged="false",
                 statement="Made-up statement.", rating="3"):
    return {
        "evidence_id": str(eid), "molecular_profile": "x", "molecular_profile_id": str(profile_id),
        "disease": disease, "therapies": therapies, "therapy_interaction_type": interaction,
        "evidence_type": etype, "evidence_direction": direction, "evidence_level": "B",
        "significance": significance, "evidence_statement": statement, "citation_id": "1",
        "citation": "Made-up et al., 2000", "rating": rating, "variant_origin": "Somatic",
        "is_flagged": flagged,
    }


EVIDENCE = [
    evidence_row(9001, 1001, "Disease X", therapies="Druga"),
    evidence_row(9002, 1001, "Disease X", therapies="Drugb", direction="Does Not Support",
                 significance="Resistance"),
    evidence_row(9003, 1001, "Disease X", therapies="Drugc,Drugd", interaction="Combination"),
    evidence_row(9004, 1001, "Disease X", therapies="Druge", flagged="true"),
    evidence_row(9005, 1001, "Disease X Subtype", therapies="Drugf"),
    evidence_row(9006, 1001, "Disease Y", etype="Prognostic", significance="Poor Outcome", rating=""),
    evidence_row(9007, 5000, "Disease X", therapies="Drugg"),  # combined profile, not single-variant
    evidence_row(9008, 1003, "Disease W", therapies="Drugh,HER2 Inhibitor CP-724,714,2,4-pyrimidinediamine",
                 interaction="Substitutes"),  # real CIViC names that contain a comma
] + [
    # Long rows for the character budget: 40 x ~1,300 chars is over 35,000.
    evidence_row(9100 + i, 1002, "Disease Z", therapies="Drugz", statement="s" * 1000)
    for i in range(40)
]


def _write_tsv(path: Path, columns: list[str], rows: list[dict]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def write_snapshot(directory: Path) -> Path:
    files = {
        f"{RELEASE}-FeatureSummaries.tsv": (FEATURE_COLUMNS, FEATURES),
        f"{RELEASE}-VariantSummaries.tsv": (VARIANT_COLUMNS, VARIANTS),
        f"{RELEASE}-AcceptedClinicalEvidenceSummaries.tsv": (EVIDENCE_COLUMNS, EVIDENCE),
    }
    entries = []
    for name, (columns, rows) in files.items():
        path = directory / name
        _write_tsv(path, columns, rows)
        entries.append({"name": name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    (directory / "manifest.json").write_text(json.dumps({"release": RELEASE, "files": entries}))
    return directory


class SnapshotTestCase(unittest.TestCase):
    """Each test gets its own made-up snapshot, removed after the test."""

    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.snapshot_dir = write_snapshot(Path(tmp.name))
