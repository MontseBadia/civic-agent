import json
import unittest
from dataclasses import fields

from agent.answer import ANSWER_SCHEMA, MAX_LIST_ITEMS, Answer, AnswerParseError, parse_answer

VALID = {
    "gene": "BRAF",
    "variant": "V600E",
    "disease": "Melanoma",
    "explanation": "Two evidence items support sensitivity.",
    "abstain": False,
    "abstain_reason": None,
    "drugs": ["Dabrafenib", "Vemurafenib"],
    "evidence_ids": ["EID1", "EID2"],
}


def with_changes(**changes) -> str:
    return json.dumps({**VALID, **changes})


class ValidAnswerTest(unittest.TestCase):
    def test_parses_into_answer(self):
        a = parse_answer(json.dumps(VALID))
        self.assertEqual(a.drugs, ("Dabrafenib", "Vemurafenib"))
        self.assertEqual(a.evidence_ids, ("EID1", "EID2"))
        self.assertFalse(a.abstain)
        self.assertIsNone(a.abstain_reason)

    def test_abstention_parses(self):
        a = parse_answer(with_changes(abstain=True, abstain_reason="no_evidence", drugs=[], evidence_ids=[]))
        self.assertEqual(a.abstain_reason, "no_evidence")

    def test_abstain_with_drugs_still_parses(self):
        # Inconsistent but parseable; the scorer marks it wrong.
        a = parse_answer(with_changes(abstain=True, abstain_reason="no_evidence"))
        self.assertEqual(len(a.drugs), 2)


class UnparseableTest(unittest.TestCase):
    def assertUnparseable(self, text, reason):
        with self.assertRaisesRegex(AnswerParseError, reason):
            parse_answer(text)

    def test_no_content(self):
        self.assertUnparseable(None, "no content")

    def test_truncated_at_max_tokens(self):
        # Shape of the live check failure: looping list cut mid string.
        self.assertUnparseable('{"gene": "BRAF", "drugs": ["Dabrafenib", "Vemura', "invalid JSON")

    def test_not_an_object(self):
        self.assertUnparseable('["Dabrafenib"]', "expected an object")

    def test_missing_key(self):
        data = dict(VALID)
        del data["evidence_ids"]
        self.assertUnparseable(json.dumps(data), "missing keys: evidence_ids")

    def test_extra_key(self):
        self.assertUnparseable(with_changes(confidence=0.9), "unexpected keys: confidence")

    def test_string_field_wrong_type(self):
        self.assertUnparseable(with_changes(gene=None), "gene must be a string")

    def test_abstain_must_be_bool(self):
        self.assertUnparseable(with_changes(abstain="false"), "abstain must be true or false")

    def test_abstain_int_is_not_bool(self):
        self.assertUnparseable(with_changes(abstain=0), "abstain must be true or false")

    def test_unknown_abstain_reason(self):
        self.assertUnparseable(with_changes(abstain_reason="unsure"), "abstain_reason not allowed")

    def test_drugs_must_be_list(self):
        self.assertUnparseable(with_changes(drugs="Dabrafenib"), "drugs must be a list of strings")

    def test_drugs_items_must_be_strings(self):
        self.assertUnparseable(with_changes(drugs=["Dabrafenib", 3]), "drugs must be a list of strings")

    def test_too_many_drugs(self):
        too_many = [f"drug{i}" for i in range(MAX_LIST_ITEMS + 1)]
        self.assertUnparseable(with_changes(drugs=too_many), f"max {MAX_LIST_ITEMS}")


class SchemaAgreementTest(unittest.TestCase):
    def test_schema_parser_and_dataclass_have_same_fields(self):
        names = [f.name for f in fields(Answer)]
        self.assertEqual(list(ANSWER_SCHEMA["properties"]), names)
        self.assertEqual(ANSWER_SCHEMA["required"], names)

    def test_explanation_and_abstain_come_before_drugs(self):
        order = list(ANSWER_SCHEMA["properties"])
        self.assertLess(order.index("explanation"), order.index("drugs"))
        self.assertLess(order.index("abstain"), order.index("drugs"))


if __name__ == "__main__":
    unittest.main()
