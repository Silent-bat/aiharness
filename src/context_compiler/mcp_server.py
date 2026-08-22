import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional

from .service import ContextService


PROTOCOL_VERSION = "2025-06-18"


TOOLS = [
    {
        "name": "index_repository",
        "description": "Incrementally update the repository code graph.",
        "inputSchema": {"type": "object", "properties": {}, "additionalProperties": False},
    },
    {
        "name": "search_code_graph",
        "description": "Find code using lexical search and task-conditioned graph expansion.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "task_type": {
                    "type": "string",
                    "enum": [
                        "general",
                        "debug",
                        "feature",
                        "refactor",
                        "security",
                        "performance",
                        "test",
                    ],
                },
                "limit": {"type": "integer", "minimum": 1, "maximum": 100},
            },
            "required": ["query"],
            "additionalProperties": False,
        },
    },
    {
        "name": "compile_repository_context",
        "description": "Compile the highest-value repository evidence under a token budget.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "task_type": {
                    "type": "string",
                    "enum": [
                        "general",
                        "debug",
                        "feature",
                        "refactor",
                        "security",
                        "performance",
                        "test",
                    ],
                },
                "token_budget": {
                    "type": "integer",
                    "minimum": 128,
                    "maximum": 100000,
                },
                "graph_depth": {"type": "integer", "minimum": 0, "maximum": 5},
                "max_items": {"type": "integer", "minimum": 1, "maximum": 100},
            },
            "required": ["query"],
            "additionalProperties": False,
        },
    },
    {
        "name": "analyze_symbol_impact",
        "description": "Find likely callers, references, imports, and dependents of a symbol.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "symbol": {"type": "string"},
                "depth": {"type": "integer", "minimum": 1, "maximum": 5},
                "limit": {"type": "integer", "minimum": 1, "maximum": 100},
            },
            "required": ["symbol"],
            "additionalProperties": False,
        },
    },
    {
        "name": "code_graph_stats",
        "description": "Return file, symbol, and edge counts for the current graph.",
        "inputSchema": {"type": "object", "properties": {}, "additionalProperties": False},
    },
]


class McpServer:
    def __init__(self, service: ContextService):
        self.service = service

    def handle(self, message: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        method = message.get("method")
        request_id = message.get("id")
        if request_id is None:
            return None
        try:
            if method == "initialize":
                return self._result(
                    request_id,
                    {
                        "protocolVersion": PROTOCOL_VERSION,
                        "capabilities": {"tools": {"listChanged": False}},
                        "serverInfo": {
                            "name": "graph-context-compiler",
                            "version": "0.1.0",
                        },
                    },
                )
            if method == "ping":
                return self._result(request_id, {})
            if method == "tools/list":
                return self._result(request_id, {"tools": TOOLS})
            if method == "tools/call":
                params = message.get("params") or {}
                value = self._call_tool(params.get("name"), params.get("arguments") or {})
                return self._result(
                    request_id,
                    {
                        "content": [
                            {
                                "type": "text",
                                "text": json.dumps(value, indent=2, sort_keys=True),
                            }
                        ],
                        "structuredContent": value,
                        "isError": False,
                    },
                )
            return self._error(request_id, -32601, "method not found: {}".format(method))
        except Exception as error:
            return self._error(request_id, -32000, str(error))

    def _call_tool(self, name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        if name == "index_repository":
            return self.service.index()
        if name == "search_code_graph":
            return {
                "results": self.service.search(
                    arguments["query"],
                    task_type=arguments.get("task_type"),
                    limit=int(arguments.get("limit", 20)),
                )
            }
        if name == "compile_repository_context":
            return self.service.compile_context(
                arguments["query"],
                token_budget=int(arguments.get("token_budget", 4000)),
                task_type=arguments.get("task_type"),
                graph_depth=int(arguments.get("graph_depth", 2)),
                max_items=int(arguments.get("max_items", 20)),
            )
        if name == "analyze_symbol_impact":
            return self.service.impact(
                arguments["symbol"],
                depth=int(arguments.get("depth", 2)),
                limit=int(arguments.get("limit", 40)),
            )
        if name == "code_graph_stats":
            return self.service.stats()
        raise ValueError("unknown tool: {}".format(name))

    @staticmethod
    def _result(request_id: Any, result: Any) -> Dict[str, Any]:
        return {"jsonrpc": "2.0", "id": request_id, "result": result}

    @staticmethod
    def _error(request_id: Any, code: int, message: str) -> Dict[str, Any]:
        return {
            "jsonrpc": "2.0",
            "id": request_id,
            "error": {"code": code, "message": message},
        }


def main() -> int:
    repository = Path(os.environ.get("CONTEXT_COMPILER_REPO", os.getcwd()))
    database_value = os.environ.get("CONTEXT_COMPILER_DATABASE")
    database = Path(database_value) if database_value else None
    with ContextService(repository, database) as service:
        server = McpServer(service)
        for line in sys.stdin:
            if not line.strip():
                continue
            try:
                message = json.loads(line)
                response = server.handle(message)
                if response is not None:
                    sys.stdout.write(json.dumps(response, separators=(",", ":")) + "\n")
                    sys.stdout.flush()
            except json.JSONDecodeError as error:
                response = McpServer._error(None, -32700, str(error))
                sys.stdout.write(json.dumps(response, separators=(",", ":")) + "\n")
                sys.stdout.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

