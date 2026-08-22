import argparse
import json
from pathlib import Path
from typing import Dict, List, Sequence, Set

from .service import ContextService


def evaluate(repository: Path, cases_path: Path, limit: int = 20) -> Dict[str, object]:
    cases = json.loads(cases_path.read_text(encoding="utf-8"))
    rows: List[Dict[str, object]] = []
    with ContextService(repository) as service:
        service.index()
        for case in cases:
            results = service.search(
                case["query"], task_type=case.get("task_type"), limit=limit
            )
            expected = set(case.get("expected_symbols", []))
            returned = {result["name"] for result in results}
            matched = expected & returned
            recall = len(matched) / len(expected) if expected else 1.0
            precision = len(matched) / len(returned) if returned else 0.0
            reciprocal_rank = 0.0
            for index, result in enumerate(results, 1):
                if result["name"] in expected:
                    reciprocal_rank = 1.0 / index
                    break
            rows.append(
                {
                    "id": case.get("id", case["query"]),
                    "recall_at_k": recall,
                    "precision_at_k": precision,
                    "reciprocal_rank": reciprocal_rank,
                    "matched": sorted(matched),
                    "missing": sorted(expected - returned),
                }
            )
    count = len(rows) or 1
    return {
        "case_count": len(rows),
        "limit": limit,
        "mean_recall_at_k": sum(row["recall_at_k"] for row in rows) / count,
        "mean_precision_at_k": sum(row["precision_at_k"] for row in rows) / count,
        "mean_reciprocal_rank": sum(row["reciprocal_rank"] for row in rows) / count,
        "cases": rows,
    }


def main(argv: Sequence[str] = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate graph-context retrieval.")
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=20)
    arguments = parser.parse_args(argv)
    print(
        json.dumps(
            evaluate(arguments.repo, arguments.cases, arguments.limit),
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

