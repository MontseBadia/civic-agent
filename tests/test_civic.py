import unittest
from dataclasses import replace

from agent.civic import CivicSnapshot, SnapshotError, load_snapshot, normalize_variant, verify_manifest
from tests.civic_fixture import RELEASE, SnapshotTestCase


class NormalizeVariantTest(unittest.TestCase):
    def test_forms_of_the_same_change(self):
        for text in ("V600E", "v600e", "Val600Glu", "VAL600GLU", "p.Val600Glu", "p.V600E", " V600E "):
            with self.subTest(text=text):
                self.assertEqual(normalize_variant(text), "V600E")

    def test_non_amino_acid_three_letters_left_alone(self):
        self.assertEqual(normalize_variant("Abc600Xyz"), "ABC600XYZ")

    def test_other_names_only_uppercased(self):
        self.assertEqual(normalize_variant("Amplification"), "AMPLIFICATION")


class ManifestTest(SnapshotTestCase):
    def test_matching_snapshot_loads(self):
        verify_manifest(self.snapshot_dir)

    def test_changed_file_refused_and_named(self):
        path = self.snapshot_dir / f"{RELEASE}-VariantSummaries.tsv"
        path.write_text(path.read_text() + "\n")
        with self.assertRaisesRegex(SnapshotError, "VariantSummaries.tsv: sha256"):
            load_snapshot(self.snapshot_dir)

    def test_missing_file_refused(self):
        (self.snapshot_dir / f"{RELEASE}-FeatureSummaries.tsv").unlink()
        with self.assertRaisesRegex(SnapshotError, "FeatureSummaries.tsv: missing"):
            load_snapshot(self.snapshot_dir)


class DuplicateIdTest(SnapshotTestCase):
    def setUp(self):
        super().setUp()
        s = load_snapshot(self.snapshot_dir)
        self.features = list(s._features.values())
        self.variants = list(s._variants.values())
        self.evidence = [e for rows in s._evidence_by_profile.values() for e in rows]

    def test_duplicate_feature_id(self):
        with self.assertRaisesRegex(SnapshotError, "duplicate feature IDs: 1"):
            CivicSnapshot(self.features + [replace(self.features[0], name="OTHER")], self.variants, self.evidence)

    def test_two_variants_on_one_profile(self):
        clash = replace(self.variants[1], id=999)  # same profile_id as variant 102
        with self.assertRaisesRegex(SnapshotError, "duplicate single-variant profile IDs: 1002"):
            CivicSnapshot(self.features, self.variants + [clash], self.evidence)

    def test_duplicate_evidence_id(self):
        with self.assertRaisesRegex(SnapshotError, "duplicate evidence IDs: 9001"):
            CivicSnapshot(self.features, self.variants, self.evidence + [self.evidence[0]])


class LookupTest(SnapshotTestCase):
    def setUp(self):
        super().setUp()
        self.s = load_snapshot(self.snapshot_dir)

    def ids(self, items):
        return [i.id for i in items]

    def test_gene_by_name_and_alias_any_case(self):
        self.assertEqual(self.ids(self.s.find_genes("genea")), [1])
        self.assertEqual(self.ids(self.s.find_genes("GA1")), [1])

    def test_shared_gene_alias_returns_both(self):
        self.assertEqual(self.ids(self.s.find_genes("shared")), [1, 2])

    def test_unknown_gene_returns_nothing(self):
        self.assertEqual(self.s.find_genes("NOPE"), [])

    def test_variant_by_protein_forms_and_alias(self):
        for text in ("V600E", "p.Val600Glu", "VAL600GLU"):
            with self.subTest(text=text):
                self.assertEqual(self.ids(self.s.find_variants(1, text)), [101])

    def test_shared_variant_alias_returns_both(self):
        self.assertEqual(self.ids(self.s.find_variants(1, "rs1")), [101, 102])

    def test_variant_of_another_gene_not_matched(self):
        self.assertEqual(self.s.find_variants(2, "V600E"), [])

    def test_profile_name_is_gene_plus_variant(self):
        self.assertEqual(self.s.profile_variant(1001).profile_name, "GENEA V600E")

    def test_flagged_rows_dropped(self):
        self.assertNotIn(9004, self.ids(self.s.evidence_for(1001)))

    def test_all_types_and_directions_returned(self):
        self.assertEqual(self.ids(self.s.evidence_for(1001)), [9001, 9002, 9003, 9005, 9006])

    def test_disease_exact_any_case_no_subtypes(self):
        self.assertEqual(self.ids(self.s.evidence_for(1001, "disease x")), [9001, 9002, 9003])

    def test_combined_profile_refused(self):
        with self.assertRaisesRegex(LookupError, "no single-variant molecular profile 5000"):
            self.s.evidence_for(5000)

    def test_diseases_with_counts(self):
        self.assertEqual(self.s.diseases_for(1001), {"Disease X": 3, "Disease X Subtype": 1, "Disease Y": 1})

    def test_fields_parsed(self):
        e = {e.id: e for e in self.s.evidence_for(1001)}
        self.assertEqual(e[9003].therapies, ("Drugc", "Drugd"))
        self.assertEqual(e[9003].therapy_interaction_type, "Combination")
        self.assertIsNone(e[9001].therapy_interaction_type)
        self.assertEqual(e[9001].rating, 3)
        self.assertIsNone(e[9006].rating)

    def test_therapy_names_with_commas_kept_whole(self):
        self.assertEqual(
            self.s.evidence_for(1003)[0].therapies,
            ("Drugh", "HER2 Inhibitor CP-724,714", "2,4-pyrimidinediamine"),
        )

    def test_record_keeps_civic_values(self):
        record = self.s.evidence_for(1001, "Disease X")[1].to_record()
        self.assertEqual(record["evidence_id"], "EID9002")
        self.assertEqual(record["evidence_direction"], "Does Not Support")
        self.assertEqual(record["therapies"], ["Drugb"])


if __name__ == "__main__":
    unittest.main()
