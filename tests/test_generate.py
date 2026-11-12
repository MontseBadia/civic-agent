import io
import json
import random
import tempfile
import unittest
from collections import Counter
from contextlib import redirect_stdout
from pathlib import Path

from agent import civic
from agent.civic import load_snapshot
from evaluation import generate as gen
from evaluation.generate import (Snapshot, build_candidates, protein_form, question_record, read_snapshot,
                                 sample_side, split_genes, split_therapies)
from tests.civic_fixture import SnapshotTestCase


def feature(fid, name):
    return {"feature_id": str(fid), "name": name}


def variant(vid, fid, name, profile_id, hgvs=""):
    return {"variant_id": str(vid), "feature_id": str(fid), "variant": name,
            "single_variant_molecular_profile_id": str(profile_id), "hgvs_descriptions": hgvs}


def profile(pid, name, variant_ids):
    return {"molecular_profile_id": str(pid), "name": name, "variant_ids": variant_ids}


def ev(eid, pid, disease, therapies="Druga", etype="Predictive", direction="Supports",
       significance="Sensitivity/Response", interaction="", flagged="false"):
    return {"evidence_id": str(eid), "molecular_profile_id": str(pid), "disease": disease, "therapies": therapies,
            "evidence_type": etype, "evidence_direction": direction, "significance": significance,
            "therapy_interaction_type": interaction, "is_flagged": flagged}


FEATURES = [feature(1, "GENEA"), feature(2, "GENEB")]
VARIANTS = [
    variant(101, 1, "V600E", 1001, hgvs="NM_1:c.1799T>A,NP_1:p.Val640Glu,NP_2:p.Val600Glu"),
    variant(102, 1, "K601E", 1002),
    variant(103, 2, "Amplification", 1003),
]
PROFILES = [
    profile(1001, "GENEA V600E", "101"), profile(1002, "GENEA K601E", "102"),
    profile(1003, "GENEB Amplification", "103"),
    profile(5000, "GENEA V600E AND GENEB Amplification", "101,103"),
    profile(5001, "NOT GENEA K601E", "102"),
]


def snapshot(evidence, features=FEATURES, variants=VARIANTS, profiles=PROFILES):
    return Snapshot(release="r", manifest_sha256="s", features=features, variants=variants,
                    profiles=profiles, evidence=evidence)


def answerable(evidence, **kw):
    candidates, _ = build_candidates(snapshot(evidence, **kw))
    return [c for c in candidates if c.kind != "no_evidence"]


def no_evidence(evidence, **kw):
    candidates, _ = build_candidates(snapshot(evidence, **kw))
    return {(c.profile, c.disease) for c in candidates if c.kind == "no_evidence"}


class ReferenceAnswerTest(unittest.TestCase):
    def test_drugs_lowercased_sorted_with_their_evidence_ids(self):
        (c,) = answerable([ev(2, 1001, "Disease X", "Drugb"), ev(1, 1001, "Disease X", "Druga,Drugb")])
        self.assertEqual((c.kind, c.profile, c.disease), ("T1", "GENEA V600E", "Disease X"))
        self.assertEqual((c.drugs, c.evidence_ids), (("druga", "drugb"), (1, 2)))

    def test_sensitivity_and_resistance_are_separate_questions(self):
        found = answerable([ev(1, 1001, "Disease X"), ev(2, 1001, "Disease X", significance="Resistance")])
        self.assertEqual(sorted(c.kind for c in found), ["T1", "T2"])

    def test_subtype_is_its_own_disease(self):
        found = answerable([ev(1, 1001, "Disease X"), ev(2, 1001, "Disease X Subtype", "Drugb")])
        self.assertEqual({c.disease: c.drugs for c in found},
                         {"Disease X": ("druga",), "Disease X Subtype": ("drugb",)})

    def test_rows_outside_the_filters_dont_count(self):
        for name, row in [
            ("flagged", ev(1, 1001, "Disease X", flagged="true")),
            ("does not support", ev(1, 1001, "Disease X", direction="Does Not Support")),
            ("not predictive", ev(1, 1001, "Disease X", etype="Prognostic")),
            ("reduced sensitivity", ev(1, 1001, "Disease X", significance="Reduced Sensitivity")),
            ("combination", ev(1, 1001, "Disease X", interaction="Combination")),
            ("sequential", ev(1, 1001, "Disease X", interaction="Sequential")),
            ("combined profile", ev(1, 5000, "Disease X")),
            ("NOT profile", ev(1, 5001, "Disease X")),
        ]:
            with self.subTest(name):
                self.assertEqual(answerable([row]), [])

    def test_substitutes_rows_count(self):
        (c,) = answerable([ev(1, 1001, "Disease X", "Druga,Drugb", interaction="Substitutes")])
        self.assertEqual(c.drugs, ("druga", "drugb"))

    def test_more_than_five_drugs_skipped(self):
        candidates, stats = build_candidates(snapshot([ev(1, 1001, "Disease X", "D1,D2,D3,D4,D5,D6")]))
        self.assertEqual([c for c in candidates if c.kind != "no_evidence"], [])
        self.assertEqual(stats["pairs_over_drug_cap"], 1)

    def test_unknown_flag_value_raises(self):
        with self.assertRaisesRegex(ValueError, "EID1: is_flagged"):
            build_candidates(snapshot([ev(1, 1001, "Disease X", flagged="yes")]))

    def test_shared_variant_name_left_out(self):
        variants = VARIANTS + [variant(104, 1, "v600e", 1004)]
        profiles = PROFILES + [profile(1004, "GENEA v600e", "104")]
        self.assertEqual(answerable([ev(1, 1001, "Disease X")], variants=variants, profiles=profiles), [])

    def test_shared_gene_name_left_out(self):
        features = FEATURES + [feature(3, "genea")]
        self.assertEqual(answerable([ev(1, 1001, "Disease X")], features=features), [])


class NoEvidenceTest(unittest.TestCase):
    # V600E has rows in Disease X, K601E in Disease Y: each is a candidate in the other's disease.
    BASE = [ev(1, 1001, "Disease X"), ev(2, 1002, "Disease Y", etype="Prognostic")]

    def test_disease_with_rows_for_another_variant_of_the_gene(self):
        self.assertEqual(no_evidence(self.BASE), {("GENEA V600E", "Disease Y"), ("GENEA K601E", "Disease X")})

    def test_row_on_a_combined_profile_with_the_variant_blocks_it(self):
        found = no_evidence(self.BASE + [ev(3, 5000, "Disease Y")])
        self.assertNotIn(("GENEA V600E", "Disease Y"), found)

    def test_flagged_row_of_the_variant_blocks_it(self):
        found = no_evidence(self.BASE + [ev(3, 1001, "Disease Y", flagged="true")])
        self.assertNotIn(("GENEA V600E", "Disease Y"), found)

    def test_variant_without_evidence_of_its_own_left_out(self):
        found = no_evidence([ev(1, 1001, "Disease X")])
        self.assertNotIn(("GENEA K601E", "Disease X"), found)

    def test_only_diseases_of_the_same_gene(self):
        found = no_evidence(self.BASE + [ev(3, 1003, "Disease Z")])
        self.assertFalse({d for _, d in found} & {"Disease Z"})


class ParsingTest(unittest.TestCase):
    def test_therapy_names_with_commas_kept_whole(self):
        self.assertEqual(split_therapies("Drugh,HER2 Inhibitor CP-724,714,2,4-pyrimidinediamine"),
                         ["Drugh", "HER2 Inhibitor CP-724,714", "2,4-pyrimidinediamine"])

    def test_comma_names_match_the_agent_reader(self):
        self.assertEqual(set(gen.THERAPY_NAMES_WITH_COMMAS), civic.THERAPY_NAMES_WITH_COMMAS)

    def test_protein_form_is_the_one_spelling_the_variant_name(self):
        self.assertEqual(protein_form(VARIANTS[0]), "p.Val600Glu")

    def test_no_protein_form_without_a_match(self):
        self.assertIsNone(protein_form(variant(1, 1, "Amplification", 1, hgvs="NP_1:p.Val600Glu")))


def world(n_genes):
    """Each gene: variant A with T1 and T2 rows in D1, variant B with a row in D2.
    So every gene has answerable and no-evidence candidates, all with a protein form."""
    features, variants, profiles, evidence = [], [], [], []
    for g in range(1, n_genes + 1):
        features.append(feature(g, f"G{g}"))
        for k, disease in ((1, "D1"), (2, "D2")):
            vid, pid, name = g * 10 + k, g * 100 + k, f"V{g * 10 + k}E"
            variants.append(variant(vid, g, name, pid, hgvs=f"NP_1:p.Val{g * 10 + k}Glu"))
            profiles.append(profile(pid, f"G{g} {name}", str(vid)))
            evidence.append(ev(pid, pid, disease))
        evidence.append(ev(g * 100 + 50, g * 100 + 1, "D1", significance="Resistance"))
    return snapshot(evidence, features=features, variants=variants, profiles=profiles)


COUNTS = {"T1": 4, "T2": 4, "no_evidence": 2, "protein_form": 2}


class SamplingTest(unittest.TestCase):
    def sample(self, n_genes=20, counts=COUNTS):
        candidates, _ = build_candidates(world(n_genes))
        genes = {c.feature_id for c in candidates}
        return sample_side(candidates, genes, counts, random.Random(7), "dev")

    def test_exact_counts(self):
        picked = self.sample()
        kinds = Counter(c.kind for c, _, _ in picked)
        self.assertEqual(kinds, Counter({"T1": 4, "T2": 4, "no_evidence": 2}))
        self.assertEqual(sum(protein for _, _, protein in picked), 2)

    def test_at_most_two_per_gene(self):
        picked = self.sample(n_genes=6)
        self.assertLessEqual(max(Counter(c.feature_id for c, _, _ in picked).values()), 2)

    def test_no_evidence_wording_alternates(self):
        templates = [t for c, t, _ in self.sample() if c.kind == "no_evidence"]
        self.assertEqual(templates, ["T1", "T2"])

    def test_protein_form_only_on_answerable_questions(self):
        self.assertTrue(all(c.kind != "no_evidence" for c, _, protein in self.sample() if protein))

    def test_too_few_genes_raises(self):
        with self.assertRaisesRegex(ValueError, "not enough"):
            self.sample(n_genes=4)

    def test_split_is_disjoint_and_complete(self):
        dev, test = split_genes({str(i) for i in range(1, 31)}, random.Random(7))
        self.assertEqual((len(dev), len(test)), (10, 20))
        self.assertEqual(dev | test, {str(i) for i in range(1, 31)})

    def test_same_seed_same_questions(self):
        self.assertEqual(self.sample(), self.sample())


class QuestionRecordTest(unittest.TestCase):
    def setUp(self):
        candidates, _ = build_candidates(snapshot([ev(86, 1001, "Disease X"),
                                                   ev(2, 1002, "Disease Y", etype="Prognostic")]))
        self.t1 = next(c for c in candidates if c.kind == "T1")
        self.none = next(c for c in candidates if c.kind == "no_evidence")

    def test_answerable_question(self):
        r = question_record("dev-01", self.t1, "T1", False)
        self.assertEqual(r["question"], "Which therapies have CIViC evidence items supporting sensitivity "
                                        "in Disease X with GENEA V600E?")
        self.assertEqual((r["reference_drugs"], r["reference_evidence_ids"]), (["druga"], ["EID86"]))
        self.assertEqual((r["expected_abstain"], r["expected_abstain_reason"]), (False, None))

    def test_protein_form_names_the_gene(self):
        r = question_record("dev-01", self.t1, "T1", True)
        self.assertIn("with GENEA p.Val600Glu?", r["question"])
        self.assertEqual(r["variant_form"], "protein")

    def test_no_evidence_question(self):
        r = question_record("dev-02", self.none, "T2", False)
        self.assertIn("supporting resistance", r["question"])
        self.assertEqual((r["reference_drugs"], r["expected_abstain"], r["expected_abstain_reason"]),
                         ([], True, "no_evidence"))


class ReadSnapshotTest(SnapshotTestCase):
    def test_reads_every_table(self):
        s = read_snapshot(self.snapshot_dir)
        self.assertEqual(s.release, "01-Jan-2000")
        self.assertEqual(len(s.profiles), 5)

    def test_changed_file_refused(self):
        path = next(self.snapshot_dir.glob("*VariantSummaries.tsv"))
        path.write_text(path.read_text() + "\n")
        with self.assertRaisesRegex(ValueError, "sha256"):
            read_snapshot(self.snapshot_dir)

    def test_references_agree_with_the_agent_reader(self):
        agent = load_snapshot(self.snapshot_dir)
        candidates, _ = build_candidates(read_snapshot(self.snapshot_dir))
        self.assertTrue([c for c in candidates if c.kind != "no_evidence"])
        for c in candidates:
            rows = {e.id: e for e in agent.evidence_for(int(c.profile_id), c.disease)}
            with self.subTest(profile=c.profile, disease=c.disease, kind=c.kind):
                if c.kind == "no_evidence":
                    self.assertEqual(rows, {})
                else:
                    drugs = {t.lower() for i in c.evidence_ids for t in rows[i].therapies}
                    self.assertEqual(tuple(sorted(drugs)), c.drugs)


class MainTest(unittest.TestCase):
    def test_existing_question_set_not_overwritten(self):
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "dev.jsonl").write_text("keep\n")
            with self.assertRaisesRegex(SystemExit, "not overwriting"), redirect_stdout(io.StringIO()):
                gen.main(["--snapshot", "unused", "--seed", "7", "--out", tmp])
            self.assertEqual((Path(tmp) / "dev.jsonl").read_text(), "keep\n")


if __name__ == "__main__":
    unittest.main()
