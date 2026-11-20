"""MCP server for the four CIViC lookup tools, over stdio.

Wraps make_tools: same functions, names, descriptions and schemas as
condition C. The evaluation and the CLI call the functions directly.
"""

import argparse
import asyncio
import json
from pathlib import Path

from mcp import types
from mcp.server.lowlevel import Server
from mcp.server.stdio import stdio_server
from mcp.shared.exceptions import MCPError

from agent.civic import CivicSnapshot, load_snapshot
from agent.civic_tools import make_tools
from agent.tool_loop import InvalidArguments, ToolError, validate_arguments

SERVER_NAME = "civic"


def _error_result(message: str) -> types.CallToolResult:
    return types.CallToolResult(content=[types.TextContent(text=message)], is_error=True)


def build_server(snapshot: CivicSnapshot) -> Server:
    tools = {t.name: t for t in make_tools(snapshot)}

    async def list_tools(ctx, params) -> types.ListToolsResult:
        return types.ListToolsResult(tools=[
            types.Tool(name=t.name, description=t.description, input_schema=t.parameters)
            for t in tools.values()
        ])

    async def call_tool(ctx, params: types.CallToolRequestParams) -> types.CallToolResult:
        tool = tools.get(params.name)
        if tool is None:
            available = ", ".join(sorted(tools))
            raise MCPError(types.INVALID_PARAMS, f"unknown tool {params.name!r}, available: {available}")
        # MCP allows omitting arguments; validate_arguments then reports the missing ones.
        arguments = params.arguments if params.arguments is not None else {}
        try:
            validate_arguments(arguments, tool.parameters)
            result = tool.function(**arguments)
        except (InvalidArguments, ToolError) as e:
            return _error_result(str(e))
        # Any other exception propagates: the SDK returns it as a protocol error.
        return types.CallToolResult(content=[types.TextContent(text=json.dumps(result))], structured_content=result)

    return Server(SERVER_NAME, on_list_tools=list_tools, on_call_tool=call_tool)


async def _serve(server: Server) -> None:
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, server.create_initialization_options())


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--snapshot", required=True, type=Path, help="snapshot directory, e.g. data/raw/01-Nov-2026")
    args = parser.parse_args(argv)
    server = build_server(load_snapshot(args.snapshot))  # also checks the manifest
    asyncio.run(_serve(server))


if __name__ == "__main__":
    main()
