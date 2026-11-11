import json
from dataclasses import replace

from agent.civic import load_snapshot
from agent.civic_tools import EVIDENCE_CHAR_BUDGET, make_tools
from agent.tool_loop import MAX_RESULT_CHARS, ToolError, tools_hash
from tests.civic_fixture import SnapshotTestCase


class CivicToolsTest(SnapshotTestCase):
    def setUp(self):
        super().setUp()
        self.tools = {t.name: t.function for t in make_tools(load_snapshot(self.snapshot_dir))}

    def test_four_tools(self):
        self.assertEqual(sorted(self.tools), ["get_evidence", "list_diseases", "search_gene", "search_variant"])

    def test_unknown_gene_is_an_empty_result(self):
        self.assertEqual(self.tools["search_gene"]("NOPE")["matches"], [])

    def test_ambiguous_gene_returns_all_matches(self):
        matches = self.tools["search_gene"]("SHARED")["matches"]
        self.assertEqual([m["gene_id"] for m in matches], [1, 2])

    def test_empty_gene_is_a_tool_error(self):
        with self.assertRaisesRegex(ToolError, "gene must not be empty"):
            self.tools["search_gene"]("  ")

    def test_search_variant_gives_profile(self):
        match = self.tools["search_variant"](1, "p.Val600Glu")["matches"][0]
        self.assertEqual((match["molecular_profile_id"], match["molecular_profile_name"]), (1001, "GENEA V600E"))

    def test_search_variant_unknown_gene_id_is_a_tool_error(self):
        with self.assertRaisesRegex(ToolError, "no gene_id 99"):
            self.tools["search_variant"](99, "V600E")

    def test_list_diseases(self):
        diseases = self.tools["list_diseases"](1001)["diseases"]
        self.assertEqual(diseases[0], {"disease": "Disease X", "rows": 3})

    def test_combined_profile_is_a_tool_error(self):
        with self.assertRaisesRegex(ToolError, "5000"):
            self.tools["get_evidence"](5000)

    def test_unknown_disease_is_an_empty_result(self):
        result = self.tools["get_evidence"](1001, "Disease Q")
        self.assertEqual((result["total"], result["rows"], result["truncated"]), (0, [], False))

    def test_get_evidence_by_disease_returns_all(self):
        result = self.tools["get_evidence"](1001, "Disease X")
        self.assertEqual([r["evidence_id"] for r in result["rows"]], ["EID9001", "EID9002", "EID9003"])
        self.assertFalse(result["truncated"])

    def test_character_budget_truncates_and_reports_total(self):
        result = self.tools["get_evidence"](1002)
        self.assertEqual(result["total"], 40)
        self.assertTrue(result["truncated"])
        self.assertLess(result["returned"], 40)
        self.assertLessEqual(len(json.dumps(result["rows"])), EVIDENCE_CHAR_BUDGET)

    def test_truncated_result_stays_under_loop_guard(self):
        self.assertLess(len(json.dumps(self.tools["get_evidence"](1002))), MAX_RESULT_CHARS)


class ToolsHashTest(SnapshotTestCase):
    def test_stable_and_changes_with_description(self):
        tools = make_tools(load_snapshot(self.snapshot_dir))
        before = tools_hash(tools)
        self.assertEqual(tools_hash(make_tools(load_snapshot(self.snapshot_dir))), before)
        changed = [replace(tools[0], description="x"), *tools[1:]]
        self.assertNotEqual(tools_hash(changed), before)
