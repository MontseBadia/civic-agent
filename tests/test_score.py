import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

from evaluation import score
from evaluation.generate import read_snapshot
from evaluation.score import (ScoreError, build_report, check_comparable, load_questions, load_run, mcnemar,
                              normalize_id, score_question, summarize)
from tests.civic_fixture import SnapshotTestCase

# Fixture: profile 1001 in Disease X has EID9001 (Druga, Supports, Sensitivity),
# EID9002 (Drugb, Does Not Support, Resistance), EID9003 (Drugc+Drugd, Combination).
T1 = {"id": "q1", "kind": "T1", "template": "T1", "profile_id": 1001, "disease": "Disease X",
      "reference_drugs": ["druga"], "reference_evidence_ids": ["EID9001"],
      "expected_abstain": False, "expected_abstain_reason": None}
NO_EVIDENCE = {"id": "q2", "kind": "no_evidence", "template": "T1", "profile_id": 1002, "disease": "Disease X",
               "reference_drugs": [], "reference_evidence_ids": [],
               "expected_abstain": True, "expected_abstain_reason": "no_evidence"}
AMBIGUOUS = {"id": "h1", "kind": "hand", "group": "ambiguous_gene", "template": "T1", "profile_id": None,
             "disease": "Disease X", "reference_drugs": [], "reference_evidence_ids": [],
             "expected_abstain": True, "expected_abstain_reason": "ambiguous"}


def record(qid="q1", condition="B", drugs=(), ids=(), abstain=False, reason=None, trace=(), retrieved=(),
           duration=10.0):
    answer = {"gene": "g", "variant": "v", "disease": "d", "explanation": "e", "abstain": abstain,
              "abstain_reason": reason, "drugs": list(drugs), "evidence_ids": list(ids)}
    return {"type": "question", "question_id": qid, "condition": condition, "duration_s": duration,
            "stop_reason": None, "answer": answer, "parse_error": None, "retrieved_ids": list(retrieved),
            "trace": list(trace), "error": None}


def model_step(prompt=100, completion=20):
    return {"kind": "ModelStep", "prompt_tokens": prompt, "completion_tokens": completion}


def evidence_call(profile_id=1001, disease="Disease X"):
    return {"kind": "ToolStep", "name": "get_evidence", "outcome": "ok",
            "raw_arguments": json.dumps({"molecular_profile_id": profile_id, "disease": disease})}


class ScoreTestCase(SnapshotTestCase):
    def setUp(self):
        super().setUp()
        self.evidence = {r["evidence_id"]: r for r in read_snapshot(self.snapshot_dir).evidence}

    def score(self, q, rec):
        return score_question(q, rec, self.evidence)


class CorrectnessTest(ScoreTestCase):
    def test_exact_drug_set_is_right(self):
        self.assertTrue(self.score(T1, record(drugs=[" Druga "]))["correct"])

    def test_extra_drug_is_wrong(self):
        self.assertFalse(self.score(T1, record(drugs=["druga", "drugb"]))["correct"])

    def test_missing_drug_is_wrong(self):
        self.assertFalse(self.score(T1, record(drugs=[]))["correct"])

    def test_abstaining_on_answerable_is_wrong(self):
        self.assertFalse(self.score(T1, record(abstain=True, reason="no_evidence"))["correct"])

    def test_expected_abstention_with_right_reason_is_right(self):
        self.assertTrue(self.score(NO_EVIDENCE, record("q2", abstain=True, reason="no_evidence"))["correct"])

    def test_wrong_abstain_reason_is_wrong(self):
        self.assertFalse(self.score(NO_EVIDENCE, record("q2", abstain=True, reason="ambiguous"))["correct"])

    def test_abstention_with_drugs_is_wrong(self):
        r = record("q2", abstain=True, reason="no_evidence", drugs=["druga"])
        self.assertFalse(self.score(NO_EVIDENCE, r)["correct"])

    def test_error_record_is_wrong(self):
        r = {"question_id": "q1", "condition": "B", "duration_s": 3.0,
             "error": {"type": "InternalServerError", "message": "500", "server_body": None}}
        s = self.score(T1, r)
        self.assertEqual((s["status"], s["correct"]), ("error", False))

    def test_unparseable_is_wrong_and_tagged(self):
        r = {**record(), "answer": None, "parse_error": "invalid JSON"}
        s = self.score(T1, r)
        self.assertEqual((s["status"], s["correct"], s["failure_tag"]), ("unparseable", False, "format_error"))


class CitationTest(ScoreTestCase):
    def test_prefix_forms_normalized(self):
        self.assertEqual([normalize_id(c) for c in ["EID86", "eid 86", "86", "PMID1"]],
                         ["EID86", "EID86", "EID86", "PMID1"])

    def test_cited_does_not_support_row_left_out_of_precision(self):
        s = self.score(T1, record(drugs=["druga"], ids=["EID9001", "EID9002"]))
        self.assertEqual((s["cited_counted"], s["cited_right"]), (["EID9001"], ["EID9001"]))

    def test_invented_id_is_invalid(self):
        s = self.score(T1, record(ids=["EID123456"]))
        self.assertEqual((s["cited_invented"], s["cited_valid"]), (["EID123456"], []))

    def test_row_of_another_disease_is_invalid(self):
        s = self.score(T1, record(ids=["EID9005"]))  # Disease X Subtype
        self.assertEqual((s["cited_valid"], s["cited_invented"]), ([], []))

    def test_any_citation_without_a_profile_is_invalid(self):
        s = self.score(AMBIGUOUS, record("h1", abstain=True, reason="ambiguous", ids=["EID9001"]))
        self.assertEqual(s["cited_valid"], [])

    def test_drug_backed_by_a_cited_supporting_row(self):
        s = self.score(T1, record(drugs=["druga", "drugc"], ids=["EID9001", "EID9003"]))
        self.assertEqual(s["drugs_supported"], ["druga"])  # Drugc only in a Combination row


class RetrievalAndLookupTest(ScoreTestCase):
    def test_reference_rows_retrieved(self):
        s = self.score(T1, record(retrieved=["EID9001", "EID9002"]))
        self.assertEqual(s["reference_retrieved"], ["EID9001"])

    def test_lookup_reaching_profile_and_disease(self):
        self.assertTrue(self.score(T1, record(trace=[evidence_call(disease="disease x")]))["lookup_ok"])

    def test_lookup_with_another_disease_fails(self):
        self.assertFalse(self.score(T1, record(trace=[evidence_call(disease="Disease Y")]))["lookup_ok"])

    def test_no_lookup_score_without_a_profile(self):
        self.assertIsNone(self.score(AMBIGUOUS, record("h1"))["lookup_ok"])


class SummaryTest(ScoreTestCase):
    def test_micro_precision_and_recall(self):
        rows = [self.score(T1, record(drugs=["druga", "drugb"])),           # 1 right of 2
                self.score(NO_EVIDENCE, record("q2", drugs=["drugz"]))]     # false positive
        s = summarize(rows)
        self.assertEqual((s["drug_precision"], s["drug_recall"]), (1 / 3, 1.0))

    def test_abstention_counts(self):
        rows = [self.score(T1, record(abstain=True, reason="no_evidence")),
                self.score(NO_EVIDENCE, record("q2")),
                self.score(AMBIGUOUS, record("h1", abstain=True, reason="no_evidence"))]
        s = summarize(rows)
        self.assertEqual((s["false_abstentions"], s["missed_abstentions"], s["wrong_abstain_reason"]), (1, 1, 1))

    def test_error_not_counted_as_missed_abstention(self):
        r = {"question_id": "q2", "condition": "B", "duration_s": 3.0,
             "error": {"type": "APITimeoutError", "message": "timeout", "server_body": None}}
        s = summarize([self.score(NO_EVIDENCE, r)])
        self.assertEqual((s["errors"], s["missed_abstentions"]), (1, 0))

    def test_cost_from_the_trace(self):
        s = summarize([self.score(T1, record(trace=[model_step(), evidence_call(), model_step(300, 40)]))])
        self.assertEqual((s["cost"]["mean_model_calls"], s["cost"]["mean_tool_calls"],
                          s["cost"]["mean_prompt_tokens"]), (2, 1, 400))

    def test_templated_and_hand_reported_apart(self):
        rows = [self.score(T1, record(condition="A", drugs=["druga"])),
                self.score(AMBIGUOUS, record("h1", condition="A"))]
        report = build_report(rows, ["A"])
        self.assertEqual({k: v["accuracy"] for k, v in report["by_condition"]["A"].items()},
                         {"templated": 1.0, "hand": 0.0})


class McNemarTest(unittest.TestCase):
    def test_no_discordant_pairs(self):
        self.assertEqual(mcnemar([True, False], [True, False])["p_value"], 1.0)

    def test_known_value(self):
        # 6 pairs only the second got right, 0 the other way: p = 2 * 0.5^6 = 0.03125
        m = mcnemar([False] * 6 + [True], [True] * 6 + [True])
        self.assertEqual((m["first_only"], m["second_only"]), (0, 6))
        self.assertAlmostEqual(m["p_value"], 0.03125)

    def test_symmetric(self):
        a, b = [True, True, False, True, False], [False, True, True, False, False]
        self.assertEqual(mcnemar(a, b)["p_value"], mcnemar(b, a)["p_value"])


class LoadingTest(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.dir = Path(tmp.name)

    def write(self, name, lines):
        path = self.dir / name
        path.write_text("".join(json.dumps(x) + "\n" for x in lines))
        return path

    def header(self, **changes):
        return {"type": "run", "label": "dev_B", "git": {"commit": "c", "dirty": False}, "model": "m",
                "settings": {"seed": 7}, "prompt_hash": "p", "snapshot": {"manifest_sha256": "s"}, **changes}

    def test_bad_abstain_reason_refused(self):
        path = self.write("q.jsonl", [{**NO_EVIDENCE, "expected_abstain_reason": "unkown_gene"}])
        with self.assertRaisesRegex(ScoreError, "abstain reason"):
            load_questions([path])

    def test_incomplete_run_refused(self):
        path = self.write("run.jsonl", [self.header(), record("q1")])
        with self.assertRaisesRegex(ScoreError, "incomplete run, no record for q2"):
            load_run(path, ["q1", "q2"])

    def test_record_for_unknown_question_refused(self):
        path = self.write("run.jsonl", [self.header(), record("q1"), record("zz")])
        with self.assertRaisesRegex(ScoreError, "not in the question files: zz"):
            load_run(path, ["q1"])

    def test_different_prompt_hash_refused(self):
        with self.assertRaisesRegex(ScoreError, "prompt_hash differs"):
            check_comparable({"A": self.header(), "B": self.header(prompt_hash="other")}, "s")

    def test_other_snapshot_refused(self):
        with self.assertRaisesRegex(ScoreError, "another snapshot"):
            check_comparable({"A": self.header()}, "different")


class MainTest(SnapshotTestCase):
    def test_end_to_end_writes_files_and_refuses_overwrite(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        d = Path(tmp.name)
        sha = read_snapshot(self.snapshot_dir).manifest_sha256
        (d / "q.jsonl").write_text(json.dumps(T1) + "\n")
        for condition, drugs in (("A", ["drugz"]), ("B", ["druga"])):
            header = {"type": "run", "label": f"dev_{condition}", "git": {"commit": "c", "dirty": False},
                      "model": "m", "settings": {}, "prompt_hash": "p", "snapshot": {"manifest_sha256": sha}}
            (d / f"{condition}.jsonl").write_text(
                json.dumps(header) + "\n" + json.dumps(record(condition=condition, drugs=drugs)) + "\n")
        argv = ["--questions", str(d / "q.jsonl"), "--runs", str(d / "A.jsonl"), str(d / "B.jsonl"),
                "--snapshot", str(self.snapshot_dir), "--out", str(d / "out")]
        with redirect_stdout(io.StringIO()):
            score.main(argv)
        summary = json.loads((d / "out" / "summary.json").read_text())
        self.assertEqual(summary["by_condition"]["B"]["templated"]["accuracy"], 1.0)
        self.assertIn("B vs A, templated", summary["mcnemar"])
        with self.assertRaisesRegex(SystemExit, "not overwriting"):
            score.main(argv)


if __name__ == "__main__":
    unittest.main()
