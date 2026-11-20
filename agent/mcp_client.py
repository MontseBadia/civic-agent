"""MCP client: start the CIViC server over stdio, list its tools, call one.

Written for agent.mcp_server: assumes every tool has a description and
every result has text content and structured_content.

Example:
    python -m agent.mcp_client --snapshot data/raw/01-Nov-2026 search_gene '{"gene": "BRAF"}'
"""

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Any

from mcp import Client, StdioServerParameters
from mcp.shared.exceptions import MCPError


def server_params(snapshot: Path) -> StdioServerParameters:
    # The server runs with this client's Python and working directory, so
    # `-m agent.mcp_server` and a relative snapshot path resolve the same way.
    return StdioServerParameters(
        command=sys.executable,
        args=["-m", "agent.mcp_server", "--snapshot", str(snapshot)],
    )


async def run(target: Any, tool: str, arguments: dict) -> int:
    """target: anything Client accepts (StdioServerParameters here, a Server in tests)."""
    async with Client(target) as client:
        print("Tools:")
        for t in (await client.list_tools()).tools:
            print(f"  {t.name}: {t.description.splitlines()[0]}")
        try:
            result = await client.call_tool(tool, arguments)
        except MCPError as e:
            print(f"Error: {e.message}", file=sys.stderr)
            return 1
    if result.is_error:
        print(f"Tool error: {result.content[0].text}", file=sys.stderr)
        return 1
    print(f"\n{tool}:")
    print(json.dumps(result.structured_content, indent=2))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--snapshot", required=True, type=Path, help="snapshot directory, e.g. data/raw/01-Nov-2026")
    parser.add_argument("tool", help="tool name, e.g. search_gene")
    parser.add_argument("arguments", help='JSON object, e.g. \'{"gene": "BRAF"}\'')
    args = parser.parse_args(argv)
    try:
        arguments = json.loads(args.arguments)
    except json.JSONDecodeError as e:
        parser.error(f"arguments are not valid JSON: {e}")
    if not isinstance(arguments, dict):
        parser.error("arguments must be a JSON object")
    # The server checks too, but its startup traceback ends up buried under
    # the client's "Connection closed".
    if not (args.snapshot / "manifest.json").is_file():
        parser.error(f"no manifest.json in {args.snapshot}")
    return asyncio.run(run(server_params(args.snapshot), args.tool, arguments))


if __name__ == "__main__":
    sys.exit(main())
