import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from mcp import Client
from mcp.shared.exceptions import MCPError

from agent.civic import load_snapshot
from agent.civic_tools import make_tools
from agent.mcp_server import build_server, main
from tests.civic_fixture import write_snapshot


class McpServerTest(unittest.IsolatedAsyncioTestCase):
    """Real protocol round trips, with the client connected to the server in-process."""

    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.snapshot = load_snapshot(write_snapshot(Path(tmp.name)))
        self.server = build_server(self.snapshot)

    async def test_lists_the_tools_with_their_schemas(self):
        async with Client(self.server) as client:
            listed = (await client.list_tools()).tools
        expected = make_tools(self.snapshot)
        self.assertEqual(
            [(t.name, t.description, t.input_schema) for t in listed],
            [(t.name, t.description, t.parameters) for t in expected],
        )

    async def test_call_returns_the_tool_result(self):
        async with Client(self.server) as client:
            result = await client.call_tool("search_gene", {"gene": "GENEA"})
        direct = {t.name: t.function for t in make_tools(self.snapshot)}["search_gene"](gene="GENEA")
        self.assertFalse(result.is_error)
        self.assertEqual(result.structured_content, direct)
        self.assertEqual(json.loads(result.content[0].text), direct)

    async def test_tool_error_is_an_error_result(self):
        async with Client(self.server) as client:
            result = await client.call_tool("search_variant", {"gene_id": 99, "variant": "V600E"})
        self.assertTrue(result.is_error)
        self.assertIn("no gene_id 99", result.content[0].text)

    async def test_invalid_arguments_are_an_error_result(self):
        async with Client(self.server) as client:
            result = await client.call_tool("search_gene", {})
        self.assertTrue(result.is_error)
        self.assertIn("missing arguments: gene", result.content[0].text)

    async def test_unknown_tool_is_a_protocol_error(self):
        async with Client(self.server) as client:
            with self.assertRaisesRegex(MCPError, "unknown tool 'nope'"):
                await client.call_tool("nope", {})


class McpServerMainTest(unittest.TestCase):
    def test_snapshot_is_required(self):
        with self.assertRaises(SystemExit), mock.patch("sys.stderr"):
            main([])


if __name__ == "__main__":
    unittest.main()
