import io
import json
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest import mock

from agent import cli
from agent.config import LLMConfig
from evaluation.run import SERVER_KEYS
from tests.civic_fixture import SnapshotTestCase
from tests.test_run import FailingLLM, server_error
from tests.test_tool_loop import ScriptedLLM, response

QUESTION = "Which therapies have CIViC evidence items supporting sensitivity in Disease X with GENEA V600E?"


def answer_json(**overrides):
    answer = {"gene": "GENEA", "variant": "V600E", "disease": "Disease X", "explanation": "Because.",
              "abstain": False, "abstain_reason": None, "drugs": ["Druga"], "evidence_ids": ["EID9001"]}
    return json.dumps({**answer, **overrides})


class CliTest(SnapshotTestCase):
    def setUp(self):
        super().setUp()
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.dir = Path(tmp.name)
        self.config = self.dir / "config.json"
        self.config.write_text(json.dumps({
            "snapshot": str(self.snapshot_dir),
            "settings": {"temperature": 0.0, "seed": 7, "max_tokens": 64, "top_p": 1.0, "top_k": 0},
            "server": {key: "v" for key in SERVER_KEYS},
        }))
        for target, value in [("RUNS_DIR", self.dir / "runs"),
                              ("load_config", lambda: LLMConfig("http://h/v1", "k", "m"))]:
            patcher = mock.patch.object(cli, target, value)
            patcher.start()
            self.addCleanup(patcher.stop)

    def run_cli(self, llm, argv=None):
        """Returns exit code, stdout, stderr."""
        out, err = io.StringIO(), io.StringIO()
        with mock.patch.object(cli, "LLMClient", lambda config: llm), redirect_stdout(out), redirect_stderr(err):
            code = cli.main(argv or ["--condition", "A", "--config", str(self.config), QUESTION])
        return code, out.getvalue(), err.getvalue()

    def answer(self, content):
        return self.run_cli(ScriptedLLM(response(content=content)))

    def test_answer_prints_drugs_and_exits_0(self):
        code, out, _ = self.answer(answer_json())
        self.assertEqual(code, 0)
        self.assertIn("Drugs: Druga", out)

    def test_disclaimer_printed(self):
        _, out, _ = self.answer(answer_json())
        self.assertTrue(out.startswith(cli.DISCLAIMER))

    def test_cited_row_shows_snapshot_fields(self):
        _, out, _ = self.answer(answer_json())
        row = next(line for line in out.splitlines() if line.startswith("EID9001"))
        for value in ("Disease X", "Predictive", "Sensitivity/Response", "Supports", "Made-up et al., 2000 (1)",
                      "https://civicdb.org/links/evidence_items/9001"):
            with self.subTest(value=value):
                self.assertIn(value, row)

    def test_cited_id_not_in_snapshot_shown(self):
        _, out, _ = self.answer(answer_json(evidence_ids=["EID42"]))
        self.assertIn("EID42  not in snapshot", out)

    def test_abstention_prints_reason_and_no_table(self):
        _, out, _ = self.answer(answer_json(abstain=True, abstain_reason="no_evidence", drugs=[], evidence_ids=[]))
        self.assertIn("Abstained: no_evidence", out)
        self.assertNotIn("Evidence cited", out)

    def test_unparseable_answer_shows_error_and_raw_output(self):
        code, out, _ = self.answer("not json")
        self.assertEqual(code, 1)
        self.assertIn("No usable answer: invalid JSON", out)
        self.assertIn("Raw model output:\nnot json", out)

    def test_question_logged_after_header(self):
        self.answer(answer_json())
        [path] = (self.dir / "runs").iterdir()
        header, record = [json.loads(line) for line in path.read_text().splitlines()]
        self.assertEqual((header["type"], header["label"]), ("run", "cli_A"))
        self.assertEqual((record["type"], record["question"]), ("question", QUESTION))

    def test_server_error_logged_and_exits_1(self):
        code, _, err = self.run_cli(FailingLLM(server_error()))
        self.assertEqual(code, 1)
        self.assertIn("Request failed: InternalServerError", err)
        [path] = (self.dir / "runs").iterdir()
        self.assertEqual(json.loads(path.read_text().splitlines()[1])["error"]["type"], "InternalServerError")

    def test_condition_required(self):
        with self.assertRaises(SystemExit) as caught:
            self.run_cli(ScriptedLLM(), ["--config", str(self.config), QUESTION])
        self.assertEqual(caught.exception.code, 2)

    def test_empty_question_refused(self):
        with self.assertRaises(SystemExit) as caught:
            self.run_cli(ScriptedLLM(), ["--condition", "A", "--config", str(self.config), "  "])
        self.assertEqual(caught.exception.code, 2)


if __name__ == "__main__":
    unittest.main()
