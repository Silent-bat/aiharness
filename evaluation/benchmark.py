"""Compare retrieval arms on fixed cases with labeled relevant symbols.

`lexical` and `hybrid` are always reported so historical numbers stay
comparable. `--all-modes` additionally reports every retrieval ablation arm
declared in `context_compiler.retrieval.RETRIEVAL_MODES`.
"""

import argparse
import json
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set

from context_compiler.retrieval import RETRIEVAL_MODES
from context_compiler.service import ContextService


def _score(rows: Iterable[Dict[str, object]], expected: Set[str], limit: int) -> Dict[str, object]:
    results = list(rows)[:limit]
    names = [str(row["name"]) for row in results]
    matched = expected.intersection(names)
    first = next((index for index, name in enumerate(names, 1) if name in expected), None)
    return {
        "recall_at_k": len(matched) / len(expected) if expected else 1.0,
        "precision_at_k": len(matched) / len(set(names)) if names else 0.0,
        "reciprocal_rank": 1.0 / first if first else 0.0,
        "matched": sorted(matched),
    }


def benchmark(repository: Path, cases_path: Path, limit: int, token_budget: int,
              modes: Optional[Sequence[str]] = None) -> Dict[str, object]:
    cases = json.loads(cases_path.read_text(encoding="utf-8"))
    extra_modes = list(modes or [])
    methods: Dict[str, List[Dict[str, object]]] = {"lexical": [], "hybrid": []}
    for mode in extra_modes:
        methods.setdefault(mode, [])
    with ContextService(repository) as service:
        service.index()
        for case in cases:
            expected = set(case.get("expected_symbols", []))
            query = case["query"]
            started = time.perf_counter()
            lexical = service.store.search(query, limit=limit)
            lexical_ms = (time.perf_counter() - started) * 1000
            lexical_rows = [{"name": node.name} for node, _score in lexical]
            started = time.perf_counter()
            hybrid = service.search(query, task_type=case.get("task_type"), limit=limit)
            hybrid_ms = (time.perf_counter() - started) * 1000
            lexical_context = service.compile_context(query, token_budget=token_budget, graph_depth=0, max_items=limit)
            hybrid_context = service.compile_context(query, token_budget=token_budget, graph_depth=2, max_items=limit)
            methods["lexical"].append({"id": case["id"], "latency_ms": round(lexical_ms, 3), "estimated_context_tokens": lexical_context["estimated_tokens"], **_score(lexical_rows, expected, limit)})
            methods["hybrid"].append({"id": case["id"], "latency_ms": round(hybrid_ms, 3), "estimated_context_tokens": hybrid_context["estimated_tokens"], **_score(hybrid, expected, limit)})
            for mode in extra_modes:
                arm = RETRIEVAL_MODES[mode]
                started = time.perf_counter()
                hits = service.retriever.retrieve(
                    query, task_type=case.get("task_type"),
                    graph_depth=arm.graph_depth, result_limit=limit, mode=mode)
                elapsed = (time.perf_counter() - started) * 1000
                compiled = service.compile_context(
                    query, token_budget=token_budget, task_type=case.get("task_type"),
                    graph_depth=arm.graph_depth, max_items=limit, mode=mode)
                methods[mode].append({
                    "id": case["id"], "latency_ms": round(elapsed, 3),
                    "estimated_context_tokens": compiled["estimated_tokens"],
                    **_score([{"name": hit.node.name} for hit in hits], expected, limit)})

    summary: Dict[str, object] = {"case_count": len(cases), "limit": limit, "token_budget": token_budget, "methods": methods}
    for method, rows in methods.items():
        summary[method] = {
            "mean_recall_at_k": sum(float(row["recall_at_k"]) for row in rows) / len(rows),
            "mean_precision_at_k": sum(float(row["precision_at_k"]) for row in rows) / len(rows),
            "mean_reciprocal_rank": sum(float(row["reciprocal_rank"]) for row in rows) / len(rows),
            "mean_latency_ms": sum(float(row["latency_ms"]) for row in rows) / len(rows),
        }
        summary[method]["mean_estimated_context_tokens"] = sum(float(row["estimated_context_tokens"]) for row in rows) / len(rows)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--token-budget", type=int, default=1200)
    parser.add_argument("--modes", nargs="+", default=[], choices=sorted(RETRIEVAL_MODES),
                        metavar="MODE", help="additional retrieval ablation arms to report")
    parser.add_argument("--all-modes", action="store_true",
                        help="report every retrieval ablation arm")
    args = parser.parse_args()
    modes = sorted(RETRIEVAL_MODES) if args.all_modes else args.modes
    print(json.dumps(benchmark(args.repo, args.cases, args.limit, args.token_budget, modes),
                     indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
