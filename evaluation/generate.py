"""Generate the templated questions and their reference answers from the CIViC snapshot.

python -m evaluation.generate --snapshot data/raw/01-Nov-2026 --seed 7 --out data/questions

Reads the TSVs itself, on purpose not through agent/civic.py: reference answers
built by the code B retrieves with would share its bugs.
"""

import argparse
import csv
import hashlib
import json
import random
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

TEMPLATES = {
    "T1": "Which therapies have CIViC evidence items supporting sensitivity in {disease} with {profile}?",
    "T2": "Which therapies have CIViC evidence items supporting resistance in {disease} with {profile}?",
}
SIGNIFICANCE = {"Sensitivity/Response": "T1", "Resistance": "T2"}
# Single drug (empty) or Substitutes; Combination and Sequential are out.
INTERACTIONS = {"", "Substitutes"}
MAX_DRUGS = 5
MAX_PER_GENE = 2
DEV_FRACTION = 1 / 3
COUNTS = {
    "dev": {"T1": 8, "T2": 8, "no_evidence": 4, "protein_form": 3},
    "test": {"T1": 17, "T2": 17, "no_evidence": 6, "protein_form": 7},
}

# Same two names as agent/civic.py (a test checks), kept here so this reader
# doesn't import the agent's code.
THERAPY_NAMES_WITH_COMMAS = ("HER2 Inhibitor CP-724,714", "2,4-pyrimidinediamine")
AMINO_ACIDS = {
    "Ala": "A", "Arg": "R", "Asn": "N", "Asp": "D", "Cys": "C", "Gln": "Q", "Glu": "E",
    "Gly": "G", "His": "H", "Ile": "I", "Leu": "L", "Lys": "K", "Met": "M", "Phe": "F",
    "Pro": "P", "Ser": "S", "Thr": "T", "Trp": "W", "Tyr": "Y", "Val": "V",
}
_PROTEIN_CHANGE = re.compile(r"p\.([A-Z][a-z]{2})(\d+)([A-Z][a-z]{2})")


@dataclass(frozen=True)
class Snapshot:
    release: str
    manifest_sha256: str
    features: list[dict]
    variants: list[dict]
    profiles: list[dict]
    evidence: list[dict]


@dataclass(frozen=True)
class Candidate:
    kind: str  # T1, T2 or no_evidence
    feature_id: str
    gene: str
    profile_id: str
    profile: str
    disease: str
    drugs: tuple[str, ...]  # lowercased, sorted; empty for no_evidence
    evidence_ids: tuple[int, ...]
    protein_form: str | None  # p.Val600Glu when the variant has one matching its name


def read_snapshot(directory: Path) -> Snapshot:
    manifest_bytes = (directory / "manifest.json").read_bytes()
    manifest = json.loads(manifest_bytes)
    by_suffix = {}
    for entry in manifest["files"]:
        path = directory / entry["name"]
        if hashlib.sha256(path.read_bytes()).hexdigest() != entry["sha256"]:
            raise ValueError(f"{path}: sha256 differs from manifest.json")
        by_suffix[entry["name"].removeprefix(f"{manifest['release']}-")] = path

    def table(name: str) -> list[dict]:
        if f"{name}.tsv" not in by_suffix:
            raise ValueError(f"{directory}: no {name}.tsv in manifest.json")
        with by_suffix[f"{name}.tsv"].open(newline="") as f:
            return list(csv.DictReader(f, delimiter="\t"))

    return Snapshot(
        release=manifest["release"],
        manifest_sha256=hashlib.sha256(manifest_bytes).hexdigest(),
        features=table("FeatureSummaries"),
        variants=table("VariantSummaries"),
        profiles=table("MolecularProfileSummaries"),
        evidence=table("AcceptedClinicalEvidenceSummaries"),
    )


def split_therapies(value: str) -> list[str]:
    """CIViC lists therapies comma-separated without escaping. Names that contain
    a comma are swapped for placeholders before splitting."""
    protected = value
    for i, name in enumerate(THERAPY_NAMES_WITH_COMMAS):
        protected = protected.replace(name, f"\x00{i}\x00")
    names = []
    for part in protected.split(","):
        part = part.strip()
        for i, name in enumerate(THERAPY_NAMES_WITH_COMMAS):
            part = part.replace(f"\x00{i}\x00", name)
        if part:
            names.append(part)
    return names


def protein_form(variant: dict) -> str | None:
    """The 3-letter p. form from hgvs_descriptions that spells the variant's own name
    (V600E -> p.Val600Glu, not the p.Val640Glu of another transcript)."""
    for description in variant["hgvs_descriptions"].split(","):
        if ":p." not in description:
            continue
        p = description.split(":", 1)[1]
        match = _PROTEIN_CHANGE.fullmatch(p)
        if match and match[1] in AMINO_ACIDS and match[3] in AMINO_ACIDS:
            if f"{AMINO_ACIDS[match[1]]}{match[2]}{AMINO_ACIDS[match[3]]}" == variant["variant"]:
                return p
    return None


def _is_flagged(row: dict) -> bool:
    if row["is_flagged"] not in ("true", "false"):
        raise ValueError(f"EID{row['evidence_id']}: is_flagged is {row['is_flagged']!r}")
    return row["is_flagged"] == "true"


def build_candidates(snapshot: Snapshot) -> tuple[list[Candidate], dict]:
    """Every possible question, plus counts to check against decisions.md."""
    # Single-variant means some variant names the profile as its own. Not "one
    # variant in variant_ids": NOT profiles have one too.
    variant_by_profile = {v["single_variant_molecular_profile_id"]: v
                          for v in snapshot.variants if v["single_variant_molecular_profile_id"]}
    profile_name = {p["molecular_profile_id"]: p["name"] for p in snapshot.profiles}
    gene_name = {f["feature_id"]: f["name"] for f in snapshot.features}

    def candidate(kind, profile_id, disease, rows):
        variant = variant_by_profile[profile_id]
        drugs = sorted({t.lower() for r in rows for t in split_therapies(r["therapies"])})
        return Candidate(kind=kind, feature_id=variant["feature_id"], gene=gene_name[variant["feature_id"]],
                         profile_id=profile_id, profile=profile_name[profile_id], disease=disease,
                         drugs=tuple(drugs), evidence_ids=tuple(sorted(int(r["evidence_id"]) for r in rows)),
                         protein_form=protein_form(variant))

    # Answerable: rows that pass every filter, grouped by template, profile and disease.
    groups = defaultdict(list)
    for row in snapshot.evidence:
        if (row["molecular_profile_id"] in variant_by_profile and not _is_flagged(row)
                and row["evidence_type"] == "Predictive" and row["evidence_direction"] == "Supports"
                and row["significance"] in SIGNIFICANCE and row["therapy_interaction_type"] in INTERACTIONS):
            groups[(SIGNIFICANCE[row["significance"]], row["molecular_profile_id"], row["disease"])].append(row)
    candidates, over_cap = [], 0
    for (kind, profile_id, disease), rows in groups.items():
        c = candidate(kind, profile_id, disease, rows)
        if not c.drugs:
            raise ValueError(f"no therapies in EIDs {c.evidence_ids}")
        if len(c.drugs) > MAX_DRUGS:
            over_cap += 1
            continue
        candidates.append(c)

    # No evidence: the variant has no row of any kind for the disease, on any profile
    # that contains it (flagged rows included, to be safe), while other variants of
    # the same gene have rows there.
    profiles_of_variant = defaultdict(set)
    for p in snapshot.profiles:
        for variant_id in p["variant_ids"].split(","):
            profiles_of_variant[variant_id.strip()].add(p["molecular_profile_id"])
    diseases_of_profile = defaultdict(set)
    for row in snapshot.evidence:
        diseases_of_profile[row["molecular_profile_id"]].add(row["disease"])
    gene_diseases = defaultdict(lambda: defaultdict(set))  # feature -> disease -> variant ids
    for row in snapshot.evidence:
        variant = variant_by_profile.get(row["molecular_profile_id"])
        if variant and not _is_flagged(row) and row["disease"]:
            gene_diseases[variant["feature_id"]][row["disease"]].add(variant["variant_id"])
    for profile_id, variant in variant_by_profile.items():
        if not diseases_of_profile[profile_id]:
            continue  # no evidence of its own: wouldn't look like a normal question
        own_diseases = {d for p in profiles_of_variant[variant["variant_id"]] | {profile_id}
                        for d in diseases_of_profile[p]}
        for disease, variant_ids in gene_diseases[variant["feature_id"]].items():
            if disease not in own_diseases and variant_ids - {variant["variant_id"]}:
                candidates.append(candidate("no_evidence", profile_id, disease, []))

    # A name shared by two features, or two variants of one feature (ERBB2
    # Y772_A775DUP / Y772_A775dup, two U2AF1 features) makes the right answer
    # "ambiguous", not the reference. Left out; ambiguity is tested by hand-made questions.
    feature_names = Counter(f["name"].casefold() for f in snapshot.features)
    variant_names = Counter((v["feature_id"], v["variant"].casefold()) for v in snapshot.variants)

    def unique_names(c: Candidate) -> bool:
        variant = variant_by_profile[c.profile_id]
        return (feature_names[c.gene.casefold()] == 1
                and variant_names[(variant["feature_id"], variant["variant"].casefold())] == 1)

    before = len(candidates)
    candidates = [c for c in candidates if unique_names(c)]

    stats = {
        "pairs_with_shared_names": before - len(candidates),
        "pool_rows": sum(len(rows) for rows in groups.values()),
        "T1_pairs": sum(c.kind == "T1" for c in candidates),
        "T2_pairs": sum(c.kind == "T2" for c in candidates),
        "pairs_over_drug_cap": over_cap,
        "no_evidence_pairs": sum(c.kind == "no_evidence" for c in candidates),
    }
    order = {"T1": 0, "T2": 1, "no_evidence": 2}
    candidates.sort(key=lambda c: (order[c.kind], int(c.profile_id), c.disease))  # input order can't matter
    return candidates, stats


def split_genes(feature_ids: set[str], rng: random.Random) -> tuple[set[str], set[str]]:
    ids = sorted(feature_ids, key=int)
    rng.shuffle(ids)
    n_dev = round(len(ids) * DEV_FRACTION)
    return set(ids[:n_dev]), set(ids[n_dev:])


def sample_side(candidates: list[Candidate], genes: set[str], counts: dict, rng: random.Random,
                side: str) -> list[tuple[Candidate, str, bool]]:
    """Returns (candidate, template, protein form or not). At most MAX_PER_GENE per gene.
    Smallest pools first so the cap can't starve them: no-evidence, then the
    protein-form questions, then the remaining T1 and T2 alternately."""
    pools = {}
    for kind in ("T1", "T2", "no_evidence"):
        pools[kind] = [c for c in candidates if c.kind == kind and c.feature_id in genes]
        rng.shuffle(pools[kind])
    per_gene = Counter()

    def take(kind, protein=False):
        for i, c in enumerate(pools[kind]):
            if per_gene[c.feature_id] < MAX_PER_GENE and (c.protein_form or not protein):
                per_gene[c.feature_id] += 1
                return pools[kind].pop(i)
        what = f"protein-form {kind}" if protein else kind
        raise ValueError(f"{side}: not enough {what} candidates under the {MAX_PER_GENE}-per-gene cap")

    # Half of each group in each wording, alternating.
    picked = [(take("no_evidence"), "T1" if i % 2 == 0 else "T2", False) for i in range(counts["no_evidence"])]
    for i in range(counts["protein_form"]):
        kind = "T1" if i % 2 == 0 else "T2"
        picked.append((take(kind, protein=True), kind, True))
    left = {kind: counts[kind] - sum(p[1] == kind and p[2] for p in picked) for kind in ("T1", "T2")}
    for i in range(max(left.values())):
        for kind in ("T1", "T2"):
            if i < left[kind]:
                picked.append((take(kind), kind, False))
    return picked


def question_record(qid: str, c: Candidate, template: str, use_protein: bool) -> dict:
    profile_text = f"{c.gene} {c.protein_form}" if use_protein else c.profile
    answerable = c.kind != "no_evidence"
    return {
        "id": qid,
        "question": TEMPLATES[template].format(disease=c.disease, profile=profile_text),
        "kind": c.kind,
        "template": template,
        "gene": c.gene,
        "profile": c.profile,
        "profile_id": int(c.profile_id),
        "disease": c.disease,
        "variant_form": "protein" if use_protein else "name",
        "reference_drugs": list(c.drugs),
        "reference_evidence_ids": [f"EID{i}" for i in c.evidence_ids],
        "expected_abstain": not answerable,
        "expected_abstain_reason": None if answerable else "no_evidence",
    }


def generate(snapshot: Snapshot, seed: int) -> tuple[dict[str, list[dict]], dict]:
    rng = random.Random(seed)
    candidates, stats = build_candidates(snapshot)
    dev_genes, test_genes = split_genes({c.feature_id for c in candidates}, rng)
    gene_name = {c.feature_id: c.gene for c in candidates}
    questions = {}
    for side, genes in (("dev", dev_genes), ("test", test_genes)):
        picked = sample_side(candidates, genes, COUNTS[side], rng, side)
        questions[side] = [question_record(f"{side}-{n:02d}", c, template, protein)
                           for n, (c, template, protein) in enumerate(picked, start=1)]
    split = {
        "seed": seed,
        "snapshot_release": snapshot.release,
        "manifest_sha256": snapshot.manifest_sha256,
        "dev_fraction": DEV_FRACTION,
        "max_per_gene": MAX_PER_GENE,
        "max_drugs": MAX_DRUGS,
        "counts": COUNTS,
        "candidates": stats,
        "genes": {"dev": sorted(gene_name[g] for g in dev_genes),
                  "test": sorted(gene_name[g] for g in test_genes)},
    }
    return questions, split


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--snapshot", required=True, type=Path)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args(argv)

    # The question set is written once; regenerating means deleting it on purpose.
    outputs = [args.out / name for name in ("dev.jsonl", "test.jsonl", "split.json")]
    if existing := [str(p) for p in outputs if p.exists()]:
        raise SystemExit(f"already exists, not overwriting: {', '.join(existing)}")
    questions, split = generate(read_snapshot(args.snapshot), args.seed)
    args.out.mkdir(parents=True, exist_ok=True)
    for side, records in questions.items():
        with (args.out / f"{side}.jsonl").open("x") as f:
            f.writelines(json.dumps(r) + "\n" for r in records)
    with (args.out / "split.json").open("x") as f:
        json.dump(split, f, indent=2)
    print(json.dumps(split["candidates"]))
    print(f"dev {len(questions['dev'])}, test {len(questions['test'])} questions in {args.out}")


if __name__ == "__main__":
    main()
