import io
import json
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest import mock

from agent.civic import load_snapshot
from agent.mcp_client import main, run, server_params
from agent.mcp_server import build_server
from tests.civic_fixture import write_snapshot


class McpClientRunTest(unittest.IsolatedAsyncioTestCase):
    """run() against the real server, connected in-process instead of over stdio."""

    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.server = build_server(load_snapshot(write_snapshot(Path(tmp.name))))

    async def call(self, tool, arguments):
        out, err = io.StringIO(), io.StringIO()
        with redirect_stdout(out), redirect_stderr(err):
            code = await run(self.server, tool, arguments)
        return code, out.getvalue(), err.getvalue()

    async def test_prints_the_tool_list(self):
        _, out, _ = await self.call("search_gene", {"gene": "GENEA"})
        for name in ["search_gene", "search_variant", "list_diseases", "get_evidence"]:
            self.assertIn(f"  {name}: ", out)

    async def test_prints_the_result(self):
        code, out, _ = await self.call("search_gene", {"gene": "GENEA"})
        self.assertEqual(code, 0)
        result = json.loads(out.split("search_gene:\n", 1)[1])
        self.assertEqual(result["matches"][0]["gene_id"], 1)

    async def test_tool_error_goes_to_stderr(self):
        code, _, err = await self.call("search_variant", {"gene_id": 99, "variant": "V600E"})
        self.assertEqual(code, 1)
        self.assertIn("Tool error: no gene_id 99", err)

    async def test_unknown_tool_goes_to_stderr(self):
        code, _, err = await self.call("nope", {})
        self.assertEqual(code, 1)
        self.assertIn("Error: unknown tool 'nope'", err)


class McpClientMainTest(unittest.TestCase):
    def assert_usage_error(self, argv):
        with self.assertRaises(SystemExit), mock.patch("sys.stderr"):
            main(argv)

    def test_arguments_must_be_json(self):
        self.assert_usage_error(["--snapshot", "snap", "search_gene", "{gene: BRAF}"])

    def test_arguments_must_be_an_object(self):
        self.assert_usage_error(["--snapshot", "snap", "search_gene", '["BRAF"]'])

    def test_tool_and_arguments_are_required(self):
        self.assert_usage_error(["--snapshot", "snap"])

    def test_snapshot_must_have_a_manifest(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        with mock.patch("agent.mcp_client.asyncio.run") as run_server:
            self.assert_usage_error(["--snapshot", tmp.name, "search_gene", '{"gene": "BRAF"}'])
        run_server.assert_not_called()

    def test_server_runs_with_this_python(self):
        params = server_params(Path("data/raw/01-Nov-2026"))
        self.assertEqual(params.command, sys.executable)
        self.assertEqual(params.args, ["-m", "agent.mcp_server", "--snapshot", "data/raw/01-Nov-2026"])


if __name__ == "__main__":
    unittest.main()
