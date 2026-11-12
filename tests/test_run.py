import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

import httpx2
import openai

from agent.civic import load_snapshot
from agent.civic_tools import make_tools
from agent.llm import GenerationSettings
from evaluation.run import (SERVER_KEYS, Question, RunConfigError, answer_question, load_questions,
                            load_run_config, run_condition)
from tests.civic_fixture import SnapshotTestCase
from tests.test_tool_loop import SETTINGS, ScriptedLLM, response

ANSWER_JSON = '{"answer": true}'
REQUEST = httpx2.Request("POST", "http://h/v1/chat/completions")
CONFIG = {
    "snapshot": "data/raw/x",
    "settings": {"temperature": 0.0, "seed": 7, "max_tokens": 64, "top_p": 1.0, "top_k": 0},
    "server": {key: "v" for key in SERVER_KEYS},
}


def server_error():
    return openai.InternalServerError("Error code: 500", response=httpx2.Response(500, request=REQUEST),
                                      body={"detail": "Insufficient Memory"})


class FailingLLM:
    """Raises the given errors in order, then answers."""

    def __init__(self, *errors):
        self.errors = list(errors)

    def chat(self, messages, settings, tools=None, response_schema=None):
        if self.errors:
            raise self.errors.pop(0)
        return response(content=ANSWER_JSON)


class ListLog:
    def __init__(self):
        self.records = []

    def write(self, record):
        self.records.append(json.loads(json.dumps(record)))  # as it would be logged


class TempDirTestCase(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.dir = Path(tmp.name)

    def write(self, name, text):
        path = self.dir / name
        path.write_text(text)
        return path


class RunConfigTest(TempDirTestCase):
    def load(self, data):
        return load_run_config(self.write("config.json", json.dumps(data)))

    def test_valid(self):
        config = self.load(CONFIG)
        self.assertEqual(config.settings, GenerationSettings(**CONFIG["settings"]))
        self.assertEqual(config.snapshot, Path("data/raw/x"))

    def test_missing_setting_named(self):
        settings = {k: v for k, v in CONFIG["settings"].items() if k != "seed"}
        with self.assertRaisesRegex(RunConfigError, "missing seed"):
            self.load({**CONFIG, "settings": settings})

    def test_unexpected_key_named(self):
        with self.assertRaisesRegex(RunConfigError, "unexpected tempertaure"):
            self.load({**CONFIG, "settings": {**CONFIG["settings"], "tempertaure": 1.0}})

    def test_missing_server_fact_named(self):
        server = {k: v for k, v in CONFIG["server"].items() if k != "prefill_step_size"}
        with self.assertRaisesRegex(RunConfigError, "missing prefill_step_size"):
            self.load({**CONFIG, "server": server})

    def test_wrong_type_refused(self):
        for key, bad in [("seed", "7"), ("top_k", True), ("temperature", None)]:
            with self.subTest(key=key), self.assertRaisesRegex(RunConfigError, f"settings.{key}"):
                self.load({**CONFIG, "settings": {**CONFIG["settings"], key: bad}})

    def test_repo_config_loads(self):
        load_run_config(Path(__file__).parent.parent / "configs" / "greedy.json")


class LoadQuestionsTest(TempDirTestCase):
    def test_reads_id_and_question_and_skips_blank_lines(self):
        path = self.write("q.jsonl", '{"id": "q1", "question": "Q1?", "reference": ["Druga"]}\n\n'
                                     '{"id": "q2", "question": "Q2?"}\n')
        self.assertEqual(load_questions(path), [Question("q1", "Q1?"), Question("q2", "Q2?")])

    def test_duplicate_id_refused(self):
        path = self.write("q.jsonl", '{"id": "q1", "question": "a"}\n{"id": "q1", "question": "b"}\n')
        with self.assertRaisesRegex(ValueError, ":2: duplicate id"):
            load_questions(path)

    def test_missing_question_refused(self):
        with self.assertRaisesRegex(ValueError, ":1: needs"):
            load_questions(self.write("q.jsonl", '{"id": "q1"}\n'))

    def test_empty_file_refused(self):
        with self.assertRaisesRegex(ValueError, "no questions"):
            load_questions(self.write("q.jsonl", "\n"))


class AnswerQuestionTest(SnapshotTestCase):
    def test_c_gets_its_system_prompt_and_the_tools(self):
        llm = ScriptedLLM(response(content=None), response(content=ANSWER_JSON))
        tools = make_tools(load_snapshot(self.snapshot_dir))
        result = answer_question("C", llm, tools, "Q?", SETTINGS)
        self.assertEqual(result.stop_reason, "model_stopped")
        self.assertEqual(llm.calls[0]["messages"][1], {"role": "user", "content": "Q?"})
        self.assertEqual(len(llm.calls[0]["tools"]), len(tools))

    def test_unknown_condition_refused(self):
        with self.assertRaisesRegex(ValueError, "unknown condition"):
            answer_question("D", ScriptedLLM(), [], "Q?", SETTINGS)


class RunConditionTest(unittest.TestCase):
    QUESTIONS = [Question("q1", "Q1?"), Question("q2", "Q2?")]

    def run_a(self, llm):
        log = ListLog()
        with redirect_stdout(io.StringIO()):  # progress lines
            run_condition("A", self.QUESTIONS, llm, [], SETTINGS, log)
        return log.records

    def test_one_record_per_question_with_its_id(self):
        records = self.run_a(ScriptedLLM(response(content=ANSWER_JSON), response(content=ANSWER_JSON)))
        self.assertEqual([(r["question_id"], r["question"], r["condition"]) for r in records],
                         [("q1", "Q1?", "A"), ("q2", "Q2?", "A")])

    def test_server_error_recorded_and_run_continues(self):
        records = self.run_a(FailingLLM(server_error()))
        self.assertEqual(records[0]["error"]["server_body"], {"detail": "Insufficient Memory"})
        self.assertIsNone(records[1]["error"])

    def test_timeout_recorded_and_run_continues(self):
        records = self.run_a(FailingLLM(openai.APITimeoutError(request=REQUEST)))
        self.assertEqual(records[0]["error"]["type"], "APITimeoutError")
        self.assertEqual(len(records), 2)

    def test_server_down_stops_the_run(self):
        with self.assertRaises(openai.APIConnectionError):
            self.run_a(FailingLLM(openai.APIConnectionError(request=REQUEST)))

    def test_other_exception_stops_the_run(self):
        with self.assertRaises(KeyError):
            self.run_a(FailingLLM(KeyError("bug")))


if __name__ == "__main__":
    unittest.main()
