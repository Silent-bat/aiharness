import argparse
import json
import sys
from pathlib import Path
from typing import Optional, Sequence

from .service import ContextService
from .providers import ModelClient, ProviderConfig
from .retrieval import RETRIEVAL_MODES


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="context-compiler",
        description="Graph-guided repository context for coding agents.",
    )
    parser.add_argument(
        "--repo",
        type=Path,
        default=Path.cwd(),
        help="repository root (default: current directory)",
    )
    parser.add_argument(
        "--database",
        type=Path,
        help="SQLite graph path (default: REPO/.context-compiler/graph.sqlite3)",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("index", help="incrementally index the repository")

    search = subparsers.add_parser("search", help="retrieve ranked repository evidence")
    search.add_argument("query")
    search.add_argument("--task-type")
    search.add_argument("--limit", type=int, default=20)

    context = subparsers.add_parser(
        "context", help="compile a token-budgeted context package"
    )
    context.add_argument("query")
    context.add_argument("--task-type")
    context.add_argument("--token-budget", type=int, default=4000)
    context.add_argument("--graph-depth", type=int, default=2)
    context.add_argument("--max-items", type=int, default=20)
    context.add_argument(
        "--mode", choices=sorted(RETRIEVAL_MODES) + ["full_file"],
        help="retrieval ablation arm (default: lexical + graph)",
    )
    context.add_argument(
        "--json", action="store_true", help="emit metadata and context as JSON"
    )

    impact = subparsers.add_parser(
        "impact", help="find likely callers, references, and dependents"
    )
    impact.add_argument("symbol")
    impact.add_argument("--depth", type=int, default=2)
    impact.add_argument("--limit", type=int, default=40)

    subparsers.add_parser("stats", help="show graph statistics")

    ask = subparsers.add_parser(
        "ask", help="compile repository context and send one request through the configured AI provider"
    )
    ask.add_argument("query")
    ask.add_argument("--token-budget", type=int, default=4000)
    ask.add_argument("--graph-depth", type=int, default=2)
    ask.add_argument("--max-items", type=int, default=20)
    ask.add_argument("--max-tokens", type=int)
    ask.add_argument("--temperature", type=float)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    arguments = build_parser().parse_args(argv)
    try:
        with ContextService(arguments.repo, arguments.database) as service:
            if arguments.command == "index":
                _json(service.index())
            elif arguments.command == "search":
                _json(
                    service.search(
                        arguments.query,
                        task_type=arguments.task_type,
                        limit=arguments.limit,
                    )
                )
            elif arguments.command == "context":
                result = service.compile_context(
                    arguments.query,
                    token_budget=arguments.token_budget,
                    task_type=arguments.task_type,
                    graph_depth=arguments.graph_depth,
                    max_items=arguments.max_items,
                    mode=arguments.mode,
                )
                if arguments.json:
                    _json(result)
                else:
                    print(result["context"])
            elif arguments.command == "impact":
                _json(
                    service.impact(
                        arguments.symbol,
                        depth=arguments.depth,
                        limit=arguments.limit,
                    )
                )
            elif arguments.command == "stats":
                _json(service.stats())
            elif arguments.command == "ask":
                service.index()
                provider_config = ProviderConfig.from_env()
                compiled = service.compile_context(
                    arguments.query,
                    token_budget=arguments.token_budget,
                    graph_depth=arguments.graph_depth,
                    max_items=arguments.max_items,
                )
                response = ModelClient(provider_config).complete(
                    [{"role": "user", "content": arguments.query}],
                    system=compiled["context"],
                    max_tokens=arguments.max_tokens,
                    temperature=arguments.temperature,
                )
                _json({
                    "provider": provider_config.provider,
                    "model": response.model,
                    "answer": response.text,
                    "usage": {
                        "input_tokens": response.usage.input_tokens,
                        "output_tokens": response.usage.output_tokens,
                        "cache_read_tokens": response.usage.cache_read_tokens,
                        "cache_write_tokens": response.usage.cache_write_tokens,
                    },
                    "context": {
                        "estimated_tokens": compiled["estimated_tokens"],
                        "item_count": compiled["item_count"],
                    },
                })
            return 0
    except (OSError, ValueError, RuntimeError) as error:
        print("error: {}".format(error), file=sys.stderr)
        return 2


def _json(value: object) -> None:
    print(json.dumps(value, indent=2, sort_keys=True))


if __name__ == "__main__":
    raise SystemExit(main())
