"""CIViC snapshot: loading with manifest check, name normalization, lookups for B and C."""

import csv
import hashlib
import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

# Protein-change forms only, via the 20 amino acids.
AMINO_ACIDS = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
}
_THREE_LETTER_CHANGE = re.compile(r"([A-Za-z]{3})(\d+)([A-Za-z]{3})")


class SnapshotError(Exception):
    """The snapshot on disk doesn't match its manifest."""


@dataclass(frozen=True)
class Feature:
    id: int
    name: str
    feature_type: str
    aliases: tuple[str, ...]


@dataclass(frozen=True)
class Variant:
    id: int
    feature_id: int
    name: str
    aliases: tuple[str, ...]
    hgvs: tuple[str, ...]
    profile_id: int
    profile_name: str


@dataclass(frozen=True)
class Evidence:
    id: int
    profile_id: int
    disease: str
    therapies: tuple[str, ...]
    therapy_interaction_type: str | None
    evidence_type: str
    evidence_direction: str
    significance: str
    evidence_level: str
    rating: int | None
    variant_origin: str
    statement: str
    citation: str
    citation_id: str

    def to_record(self) -> dict:
        """What B and C are given. CIViC values verbatim so the prompt wording matches."""
        return {
            "evidence_id": f"EID{self.id}",
            "disease": self.disease,
            "therapies": list(self.therapies),
            "therapy_interaction_type": self.therapy_interaction_type,
            "evidence_type": self.evidence_type,
            "evidence_direction": self.evidence_direction,
            "significance": self.significance,
            "evidence_level": self.evidence_level,
            "rating": self.rating,
            "variant_origin": self.variant_origin,
            "statement": self.statement,
            "citation": self.citation,
        }


def _split(value: str) -> tuple[str, ...]:
    return tuple(part.strip() for part in value.split(",") if part.strip())


# CIViC therapy lists are comma-separated without escaping, and these names
# contain a comma. Found on 01-Nov-2026 by looking for split pieces that don't
# look like drug names (EIDs 1153 and 1192). A new snapshot needs the same check.
THERAPY_NAMES_WITH_COMMAS = frozenset({"HER2 Inhibitor CP-724,714", "2,4-pyrimidinediamine"})


def _split_therapies(value: str) -> tuple[str, ...]:
    parts, merged, i = _split(value), [], 0
    while i < len(parts):
        if i + 1 < len(parts) and f"{parts[i]},{parts[i + 1]}" in THERAPY_NAMES_WITH_COMMAS:
            merged.append(f"{parts[i]},{parts[i + 1]}")
            i += 2
        else:
            merged.append(parts[i])
            i += 1
    return tuple(merged)


def _read_tsv(path: Path) -> list[dict]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def normalize_gene(text: str) -> str:
    return text.strip().upper()


def normalize_variant(text: str) -> str:
    """V600E, Val600Glu, p.Val600Glu and VAL600GLU all become V600E."""
    s = text.strip()
    if s[:2].lower() == "p.":
        s = s[2:]
    # Only convert when the whole string is a 3-letter change, so names that
    # happen to contain "Ser" or "Leu" are left alone.
    match = _THREE_LETTER_CHANGE.fullmatch(s)
    if match:
        ref, pos, alt = match.groups()
        if ref.upper() in AMINO_ACIDS and alt.upper() in AMINO_ACIDS:
            s = f"{AMINO_ACIDS[ref.upper()]}{pos}{AMINO_ACIDS[alt.upper()]}"
    return s.upper()


def verify_manifest(snapshot_dir: Path) -> dict:
    """Every file in manifest.json must exist with the recorded sha256."""
    manifest = json.loads((snapshot_dir / "manifest.json").read_text())
    problems = []
    for entry in manifest["files"]:
        path = snapshot_dir / entry["name"]
        if not path.exists():
            problems.append(f"{entry['name']}: missing")
            continue
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != entry["sha256"]:
            problems.append(f"{entry['name']}: sha256 {digest}, manifest {entry['sha256']}")
    if problems:
        raise SnapshotError("snapshot doesn't match manifest.json:\n" + "\n".join(problems))
    return manifest


def _file_ending(snapshot_dir: Path, manifest: dict, suffix: str) -> Path:
    names = [e["name"] for e in manifest["files"] if e["name"].endswith(suffix)]
    if len(names) != 1:
        raise SnapshotError(f"expected one file ending in {suffix} in manifest.json, found {len(names)}")
    return snapshot_dir / names[0]


def _unique_by(items: list, key: str, label: str) -> dict:
    """Index items by an ID that must be unique; a duplicate would silently overwrite."""
    counts = Counter(getattr(item, key) for item in items)
    if duplicates := sorted(k for k, n in counts.items() if n > 1):
        raise SnapshotError(f"duplicate {label}: {', '.join(map(str, duplicates[:10]))}")
    return {getattr(item, key): item for item in items}


class CivicSnapshot:
    def __init__(self, features: list[Feature], variants: list[Variant], evidence: list[Evidence]):
        self._features = _unique_by(features, "id", "feature IDs")
        self._variants = _unique_by(variants, "id", "variant IDs")
        self._profiles = _unique_by(variants, "profile_id", "single-variant profile IDs")
        _unique_by(evidence, "id", "evidence IDs")

        self._genes_by_name: dict[str, set[int]] = defaultdict(set)
        for f in features:
            for name in (f.name, *f.aliases):
                self._genes_by_name[normalize_gene(name)].add(f.id)

        self._variants_by_name: dict[tuple[int, str], set[int]] = defaultdict(set)
        for v in variants:
            protein_forms = [h.split(":", 1)[1] for h in v.hgvs if ":p." in h]
            for name in (v.name, *v.aliases, *protein_forms):
                self._variants_by_name[(v.feature_id, normalize_variant(name))].add(v.id)

        self._evidence_by_profile: dict[int, list[Evidence]] = defaultdict(list)
        for e in sorted(evidence, key=lambda e: e.id):
            self._evidence_by_profile[e.profile_id].append(e)

    def find_genes(self, name: str) -> list[Feature]:
        """Every feature whose name or alias matches. Several means ambiguous."""
        ids = self._genes_by_name.get(normalize_gene(name), set())
        return [self._features[i] for i in sorted(ids)]

    def find_variants(self, feature_id: int, name: str) -> list[Variant]:
        """Every variant of the feature whose name, alias or p. form matches."""
        ids = self._variants_by_name.get((feature_id, normalize_variant(name)), set())
        return [self._variants[i] for i in sorted(ids)]

    def feature(self, feature_id: int) -> Feature:
        return self._features[feature_id]

    def profile_variant(self, profile_id: int) -> Variant:
        """Combined profiles (AND/OR) are out of scope, so they're unknown here."""
        if profile_id not in self._profiles:
            raise LookupError(f"no single-variant molecular profile {profile_id}")
        return self._profiles[profile_id]

    def evidence_for(self, profile_id: int, disease: str | None = None) -> list[Evidence]:
        """All non-flagged rows of every type and direction. Disease: exact name, any case."""
        self.profile_variant(profile_id)
        rows = self._evidence_by_profile.get(profile_id, [])
        if disease is None:
            return list(rows)
        wanted = disease.strip().casefold()
        return [e for e in rows if e.disease.casefold() == wanted]

    def diseases_for(self, profile_id: int) -> dict[str, int]:
        self.profile_variant(profile_id)
        counts = Counter(e.disease for e in self._evidence_by_profile.get(profile_id, []))
        return dict(sorted(counts.items()))


def load_snapshot(snapshot_dir: Path) -> CivicSnapshot:
    """Refuses to load if any file differs from manifest.json. Flagged rows are dropped."""
    manifest = verify_manifest(snapshot_dir)

    features = [
        Feature(
            id=int(r["feature_id"]),
            name=r["name"],
            feature_type=r["feature_type"],
            aliases=_split(r["feature_aliases"]),
        )
        for r in _read_tsv(_file_ending(snapshot_dir, manifest, "-FeatureSummaries.tsv"))
    ]
    variants = [
        Variant(
            id=int(r["variant_id"]),
            feature_id=int(r["feature_id"]),
            name=r["variant"],
            aliases=_split(r["variant_aliases"]),
            hgvs=_split(r["hgvs_descriptions"]),
            profile_id=int(r["single_variant_molecular_profile_id"]),
            profile_name=f"{r['feature_name']} {r['variant']}",
        )
        for r in _read_tsv(_file_ending(snapshot_dir, manifest, "-VariantSummaries.tsv"))
    ]
    evidence = [
        Evidence(
            id=int(r["evidence_id"]),
            profile_id=int(r["molecular_profile_id"]),
            disease=r["disease"],
            therapies=_split_therapies(r["therapies"]),
            therapy_interaction_type=r["therapy_interaction_type"] or None,
            evidence_type=r["evidence_type"],
            evidence_direction=r["evidence_direction"],
            significance=r["significance"],
            evidence_level=r["evidence_level"],
            rating=int(r["rating"]) if r["rating"] else None,
            variant_origin=r["variant_origin"],
            statement=r["evidence_statement"],
            citation=r["citation"],
            citation_id=r["citation_id"],
        )
        for r in _read_tsv(_file_ending(snapshot_dir, manifest, "-AcceptedClinicalEvidenceSummaries.tsv"))
        if r["is_flagged"] != "true"
    ]
    return CivicSnapshot(features, variants, evidence)
