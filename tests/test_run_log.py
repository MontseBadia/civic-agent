import json
import subprocess
import tempfile
import unittest
from pathlib import Path

from agent.config import LLMConfig
from agent.llm import GenerationSettings, LLMResponse
from agent.run_log import RunLog, git_state, new_run_path, record_from_error, record_from_loop, run_header
from agent.tool_loop import LoopResult, ModelStep, ToolStep

VALID_ANSWER = json.dumps({
    "gene": "GENEA", "variant": "V600E", "disease": "Disease X", "explanation": "e",
    "abstain": False, "abstain_reason": None, "drugs": ["Druga"], "evidence_ids": ["EID9001"],
})


def final(content):
    return LLMResponse(content=content, tool_calls=(), finish_reason="stop", model="m",
                       prompt_tokens=10, completion_tokens=5, duration_s=1.0)


def tool_step(result, outcome="ok"):
    return ToolStep(call_id="c1", name="get_evidence", raw_arguments="{}", outcome=outcome,
                    result=result, content=json.dumps(result), duration_s=0.1)


def loop_result(content=VALID_ANSWER, steps=()):
    model = ModelStep(final=True, finish_reason="stop", prompt_tokens=10, completion_tokens=5,
                      duration_s=1.0, tool_call_count=0)
    return LoopResult(final=final(content), trace=(*steps, model),
                      messages=({"role": "user", "content": "q"},), stop_reason="model_stopped")


class TempDirTestCase(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.dir = Path(tmp.name)


class HeaderTest(TempDirTestCase):
    def header(self):
        manifest = self.dir / "manifest.json"
        manifest.write_text(json.dumps({"release": "01-Jan-2000", "files": []}))
        return run_header(
            label="check", llm_config=LLMConfig(base_url="http://h/v1", api_key="secret-key", model="m"),
            settings=GenerationSettings(temperature=0.0, seed=7, max_tokens=64),
            prompt_hash="p", tools_hash="t", manifest_path=manifest,
            server={"prefill_step_size": 512}, git={"commit": "abc", "dirty": False},
            loop_limits={"max_tool_calls": 6, "max_duplicate_calls": 2},
        )

    def test_api_key_never_logged(self):
        self.assertNotIn("secret-key", json.dumps(self.header()))

    def test_contains_what_runs_must_log(self):
        h = self.header()
        self.assertEqual(h["settings"], {"temperature": 0.0, "seed": 7, "max_tokens": 64})
        self.assertEqual((h["model"], h["prompt_hash"], h["tools_hash"]), ("m", "p", "t"))
        self.assertEqual(h["snapshot"]["release"], "01-Jan-2000")
        self.assertEqual(len(h["snapshot"]["manifest_sha256"]), 64)
        self.assertEqual(h["server"], {"prefill_step_size": 512})
        self.assertEqual(h["loop_limits"], {"max_tool_calls": 6, "max_duplicate_calls": 2})


class RecordTest(unittest.TestCase):
    def test_parsed_answer_kept(self):
        r = json.loads(json.dumps(record_from_loop("q", "C", loop_result(), duration_s=2.0)))  # as logged
        self.assertEqual(r["answer"]["drugs"], ["Druga"])
        self.assertIsNone(r["parse_error"])

    def test_parse_error_kept_with_raw_content(self):
        r = record_from_loop("q", "C", loop_result(content='{"cut'), duration_s=2.0)
        self.assertIsNone(r["answer"])
        self.assertIn("invalid JSON", r["parse_error"])
        self.assertEqual(r["final_content"], '{"cut')

    def test_retrieved_ids_unique_in_order(self):
        rows = {"rows": [{"evidence_id": "EID2"}, {"evidence_id": "EID1"}]}
        steps = (tool_step(rows), tool_step(rows), tool_step({"rows": [{"evidence_id": "EID9"}]}, outcome="tool_error"))
        r = record_from_loop("q", "C", loop_result(steps=steps), duration_s=2.0)
        self.assertEqual(r["retrieved_ids"], ["EID2", "EID1"])

    def test_trace_steps_keep_their_kind(self):
        r = record_from_loop("q", "C", loop_result(steps=(tool_step({"rows": []}),)), duration_s=2.0)
        self.assertEqual([s["kind"] for s in r["trace"]], ["ToolStep", "ModelStep"])

    def test_error_record_keeps_server_body(self):
        error = RuntimeError("Error code: 500")
        error.body = {"detail": "Insufficient Memory"}
        r = record_from_error("q", "C", error, duration_s=3.0)
        self.assertEqual(r["error"], {"type": "RuntimeError", "message": "Error code: 500",
                                      "server_body": {"detail": "Insufficient Memory"}})


class RunLogTest(TempDirTestCase):
    def test_header_then_records_one_json_per_line(self):
        path = self.dir / "runs" / "r.jsonl"
        with RunLog(path, {"type": "run"}) as log:
            log.write({"type": "question", "n": 1})
        lines = [json.loads(line) for line in path.read_text().splitlines()]
        self.assertEqual([l["type"] for l in lines], ["run", "question"])

    def test_never_overwrites(self):
        path = self.dir / "r.jsonl"
        RunLog(path, {"type": "run"}).close()
        with self.assertRaises(FileExistsError):
            RunLog(path, {"type": "run"})

    def test_unserializable_value_raises(self):
        with RunLog(self.dir / "r.jsonl", {"type": "run"}) as log:
            with self.assertRaises(TypeError):
                log.write({"bad": object()})

    def test_run_path_has_timestamp_and_label(self):
        self.assertRegex(new_run_path(self.dir, "check_c").name, r"^\d{8}-\d{6}_check_c\.jsonl$")


class GitStateTest(TempDirTestCase):
    def git(self, *args):
        subprocess.run(["git", "-c", "user.name=t", "-c", "user.email=t@t", *args],
                       cwd=self.dir, check=True, capture_output=True)

    def test_commit_and_dirty_flag(self):
        self.git("init")
        (self.dir / "f.txt").write_text("a")
        self.git("add", "f.txt")
        self.git("commit", "-m", "c")
        clean = git_state(self.dir)
        self.assertEqual((len(clean["commit"]), clean["dirty"]), (40, False))
        (self.dir / "f.txt").write_text("b")
        self.assertTrue(git_state(self.dir)["dirty"])


if __name__ == "__main__":
    unittest.main()
