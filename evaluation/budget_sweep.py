"""Sweep the token budget across retrieval arms.

A single-budget comparison is not decidable: if the repository fits in the
budget, `full_file` is both the information ceiling and near the cost floor, so
retrieval can only lose. If the budget is far too small, every arm fails. The
informative measurement is success as a function of budget, which exposes where
each arm crosses over and how many tokens it needs to reach a target pass rate.

Per-task `token_budget` overrides are ignored here on purpose: an iso-token
comparison requires one budget in force for every arm and task.
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

from context_compiler.providers import load_env_file

from evaluation.task_benchmark import ARMS, DEFAULT_ARMS, benchmark

DEFAULT_BUDGETS = (128, 160, 200, 256, 320, 512, 1024, 4000)


def sweep(tasks: Sequence[Mapping[str, Any]], budgets: Sequence[int], *,
          arms: Sequence[str] = DEFAULT_ARMS, mock: bool = True,
          repetitions: int = 1, codex_cli: bool = False,
          input_rate: Optional[float] = None,
          output_rate: Optional[float] = None,
          max_items: Optional[int] = None) -> Dict[str, Any]:
    points: List[Dict[str, Any]] = []
    for budget in sorted(budgets):
        report = benchmark(tasks, mock=mock, token_budget=budget, max_tokens=None,
                           input_rate=input_rate, output_rate=output_rate,
                           repetitions=repetitions, codex_cli=codex_cli, arms=arms,
                           force_token_budget=True, max_items=max_items)
        if not report["token_budget_uniform"]:
            raise RuntimeError("budget was not uniform: {}".format(report["token_budgets_used"]))
        points.append({
            "token_budget": budget,
            "arms": {
                arm: {
                    "successes": report[arm]["successes"],
                    "attempts": report[arm]["attempts"],
                    "success_rate": report[arm]["success_rate"],
                    "mean_context_tokens": round(report[arm]["mean_context_tokens"], 2),
                    "mean_context_items": round(report[arm]["mean_context_items"], 2),
                }
                for arm in arms if arm in report
            },
        })
    return {"budgets": sorted(budgets), "arms": list(arms), "points": points,
            "max_items": max_items,
            "tokens_to_reach": _tokens_to_reach(points, arms),
            "note": "mock runs measure whether required evidence survives the budget, "
                    "not model competence"}


def _tokens_to_reach(points: Sequence[Mapping[str, Any]], arms: Sequence[str],
                     targets: Sequence[float] = (0.5, 1.0)) -> Dict[str, Dict[str, Any]]:
    """Smallest swept budget at which each arm first reaches each success rate."""
    output: Dict[str, Dict[str, Any]] = {}
    for arm in arms:
        reached: Dict[str, Any] = {}
        for target in targets:
            hit = next(
                (point["token_budget"] for point in points
                 if arm in point["arms"] and point["arms"][arm]["success_rate"] >= target),
                None,
            )
            reached["budget_for_success_rate_{:g}".format(target)] = hit
        output[arm] = reached
    return output


def render_table(result: Mapping[str, Any]) -> str:
    arms = [arm for arm in result["arms"]]
    lines = ["{:>7} ".format("budget") + " ".join("{:>15}".format(arm) for arm in arms)]
    for point in result["points"]:
        cells = []
        for arm in arms:
            row = point["arms"].get(arm)
            cells.append("-" if row is None else "{}/{} @{:.0f}".format(
                row["successes"], row["attempts"], row["mean_context_tokens"]))
        lines.append("{:>7} ".format(point["token_budget"])
                     + " ".join("{:>15}".format(cell) for cell in cells))
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", type=Path, required=True)
    parser.add_argument("--budgets", type=int, nargs="+", default=list(DEFAULT_BUDGETS))
    parser.add_argument("--arms", nargs="+", default=["full_file", "classic", "graphrag", "oracle"],
                        choices=sorted(ARMS), metavar="ARM")
    parser.add_argument("--mock", action="store_true",
                        help="use the deterministic client; measures retrieval, not the model")
    parser.add_argument("--codex-cli", action="store_true")
    parser.add_argument("--repetitions", type=int, default=1)
    parser.add_argument("--max-items", type=int,
                        help="uniform context item cap; raise it on large repositories "
                             "so the token budget is what binds")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--env-file", default=".env",
                        help="KEY=value file to load credentials from for live runs")

    parser.add_argument("--json", action="store_true", help="print JSON instead of a table")
    args = parser.parse_args()
    if args.mock and args.codex_cli:
        parser.error("--mock and --codex-cli are mutually exclusive")
    if not (args.mock or args.codex_cli):
        # See task_benchmark.main: a live run's credential has to come from a file,
        # because each sweep point is a fresh subprocess.
        loaded = load_env_file(args.env_file)
        if loaded:
            print("loaded from {}: {}".format(args.env_file, ", ".join(sorted(loaded))))
    tasks = json.loads(args.tasks.read_text(encoding="utf-8"))
    result = sweep(tasks, args.budgets, arms=args.arms, mock=args.mock,
                   repetitions=max(1, args.repetitions), codex_cli=args.codex_cli,
                   max_items=args.max_items)
    rendered = json.dumps(result, indent=2, sort_keys=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered if args.json else render_table(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
