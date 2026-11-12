import json
import unittest

from agent.answer import ANSWER_SCHEMA
from agent.civic import load_snapshot
from agent.civic_tools import make_tools
from agent.conditions import EXTRACTION_SCHEMA, ExtractionParseError, parse_extraction, run_a, run_b
from agent.prompts import B_EXTRACTION_PROMPT, B_RECORDS_INTRO, system_prompt
from agent.tool_loop import ModelStep, ToolStep
from tests.civic_fixture import SnapshotTestCase
from tests.test_tool_loop import SETTINGS, ScriptedLLM, response

QUESTION = "Which therapies have CIViC evidence items supporting sensitivity in Disease X with GENEA V600E?"
ANSWER_JSON = '{"answer": true}'  # never parsed here; the run log parses it


def extracted(gene="GENEA", variant="V600E", disease="Disease X"):
    return response(content=json.dumps({"gene": gene, "variant": variant, "disease": disease}))


class ConditionATest(unittest.TestCase):
    def test_one_json_mode_call_with_the_a_prompt(self):
        llm = ScriptedLLM(response(content=ANSWER_JSON))
        result = run_a(llm, QUESTION, SETTINGS)
        (call,) = llm.calls
        self.assertEqual(call["messages"], [{"role": "system", "content": system_prompt("A")},
                                            {"role": "user", "content": QUESTION}])
        self.assertEqual((call["response_schema"], call["tools"]), (ANSWER_SCHEMA, None))
        self.assertEqual(result.final.content, ANSWER_JSON)
        self.assertEqual([type(s) for s in result.trace], [ModelStep])
        self.assertIsNone(result.stop_reason)


class ParseExtractionTest(unittest.TestCase):
    def test_valid(self):
        e = parse_extraction('{"gene": "BRAF", "variant": "V600E", "disease": "Melanoma"}')
        self.assertEqual((e.gene, e.variant, e.disease), ("BRAF", "V600E", "Melanoma"))

    def test_unusable_output_refused(self):
        for text in [None, '{"gene": "BRAF"', '["BRAF"]', '{"gene": "BRAF", "variant": "V600E"}',
                     '{"gene": "BRAF", "variant": "V600E", "disease": "Melanoma", "x": 1}',
                     '{"gene": " ", "variant": "V600E", "disease": "Melanoma"}',
                     '{"gene": "BRAF", "variant": 600, "disease": "Melanoma"}']:
            with self.subTest(text=text), self.assertRaises(ExtractionParseError):
                parse_extraction(text)


class ConditionBTest(SnapshotTestCase):
    def setUp(self):
        super().setUp()
        self.tools = make_tools(load_snapshot(self.snapshot_dir))

    def run_b(self, *responses):
        self.llm = ScriptedLLM(*responses)
        return run_b(self.llm, self.tools, QUESTION, SETTINGS)

    def assert_code_abstained(self, result, stop_reason, abstain_reason):
        self.assertEqual(len(self.llm.calls), 1)  # extraction only, no synthesis call
        self.assertIsNone(result.final)
        self.assertEqual(result.stop_reason, stop_reason)
        self.assertTrue(result.code_answer.abstain)
        self.assertEqual(result.code_answer.abstain_reason, abstain_reason)
        self.assertEqual((result.code_answer.drugs, result.code_answer.evidence_ids), ((), ()))

    def test_extraction_call_uses_its_own_prompt_and_schema(self):
        self.run_b(extracted(), response(content=ANSWER_JSON))
        call = self.llm.calls[0]
        self.assertEqual(call["messages"], [{"role": "system", "content": B_EXTRACTION_PROMPT},
                                            {"role": "user", "content": QUESTION}])
        self.assertEqual((call["response_schema"], call["tools"]), (EXTRACTION_SCHEMA, None))

    def test_answerable_question_synthesizes_from_the_disease_rows(self):
        result = self.run_b(extracted(), response(content=ANSWER_JSON))
        synthesis = self.llm.calls[1]
        self.assertEqual(synthesis["messages"][0], {"role": "system", "content": system_prompt("B")})
        user = synthesis["messages"][1]["content"]
        self.assertTrue(user.startswith(f"{QUESTION}\n\n{B_RECORDS_INTRO}\n"))
        rows = json.loads(user.split("\n", 3)[3])
        self.assertEqual([r["evidence_id"] for r in rows], ["EID9001", "EID9002", "EID9003"])  # no subtype row
        self.assertEqual(synthesis["response_schema"], ANSWER_SCHEMA)
        self.assertEqual((result.final.content, result.code_answer, result.stop_reason),
                         (ANSWER_JSON, None, "synthesis"))

    def test_lookups_go_through_the_tools_and_into_the_trace(self):
        result = self.run_b(extracted(gene="ga1", variant="p.Val600Glu", disease="disease x"),
                            response(content=ANSWER_JSON))
        steps = [s for s in result.trace if isinstance(s, ToolStep)]
        self.assertEqual([s.name for s in steps], ["search_gene", "search_variant", "get_evidence"])
        self.assertEqual({s.outcome for s in steps}, {"ok"})

    def test_unknown_gene(self):
        result = self.run_b(extracted(gene="FOO1"))
        self.assert_code_abstained(result, "unknown_gene", "unknown_gene")

    def test_several_genes_is_ambiguous(self):
        result = self.run_b(extracted(gene="SHARED"))
        self.assert_code_abstained(result, "ambiguous_gene", "ambiguous")

    def test_gene_name_wins_over_another_genes_alias(self):
        result = self.run_b(extracted(gene="geneb", variant="Amplification", disease="Disease W"),
                            response(content=ANSWER_JSON))
        self.assertEqual(result.stop_reason, "synthesis")

    def test_variant_name_wins_over_another_variants_alias(self):
        result = self.run_b(extracted(variant="k601e", disease="Disease X"))
        self.assertEqual(result.stop_reason, "no_rows")  # not ambiguous_variant
        self.assertIn("GENEA K601E", result.code_answer.explanation)

    def test_no_variant_is_no_evidence(self):
        result = self.run_b(extracted(variant="X1Y"))
        self.assert_code_abstained(result, "no_variant", "no_evidence")

    def test_several_variants_is_ambiguous(self):
        result = self.run_b(extracted(variant="RS1"))
        self.assert_code_abstained(result, "ambiguous_variant", "ambiguous")

    def test_no_rows_for_the_disease_is_no_evidence(self):
        result = self.run_b(extracted(disease="Disease Q"))
        self.assert_code_abstained(result, "no_rows", "no_evidence")

    def test_code_answer_keeps_the_extracted_fields(self):
        answer = self.run_b(extracted(gene="FOO1", variant="X1Y", disease="Disease Q")).code_answer
        self.assertEqual((answer.gene, answer.variant, answer.disease), ("FOO1", "X1Y", "Disease Q"))

    def test_unusable_extraction_stops_without_an_answer(self):
        result = self.run_b(response(content='{"gene": ""'))
        self.assertEqual(len(self.llm.calls), 1)
        self.assertEqual((result.final, result.code_answer, result.stop_reason), (None, None, "extraction_failed"))
        self.assertEqual(result.messages[-1], {"role": "assistant", "content": '{"gene": ""'})

    def test_truncated_evidence_raises(self):
        # GENEA K601E in Disease Z is over the character budget in the fixture.
        with self.assertRaisesRegex(RuntimeError, "truncated"):
            self.run_b(extracted(variant="K601E", disease="Disease Z"))


if __name__ == "__main__":
    unittest.main()
