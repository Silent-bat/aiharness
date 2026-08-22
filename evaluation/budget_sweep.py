"""Sweep the token budget across retrieval arms.

A single-budget comparison is not decidable: if the repository fits in the
budget, `full_file` is both the information ceiling and near the cost floor, so
retrieval can only lose. If the budget is far too small, every arm fails. The
informative measurement is success as a function of budget, which exposes where
each arm crosses over and how many tokens it needs to reach a target pass rate.

**The primary endpoint here is budget-at-100%-success, not success-at-a-budget.**
Two live runs of the same 8-arm comparison disagreed on which arms passed --
`classic` moved 3/6 to 6/6 with nothing changed -- so success at a fixed budget
is dominated by run-to-run noise on task sets this size. The budget an arm needs
to reach full success is a continuous quantity measured over the whole sweep, and
it is bracketed: `oracle` is the information floor and `full_file` the ceiling.
Success rate is reported only as a precondition, so a reader can see that an arm
reached the target before its budget is compared. Output tokens and latency are
co-primary for the same reason: they reproduced across both live runs.

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
                    # Co-primary endpoints. These reproduced across two live runs
                    # at a separation success never achieved, so they are carried
                    # at every sweep point rather than reported once at the end.
                    "mean_output_tokens": _safe_mean(report[arm], "total_output_tokens"),
                    "mean_latency_ms": round(report[arm]["mean_latency_ms"], 2),
                    "output_tokens_sd": _metric_sd(report, arm, "output_tokens"),
                    "latency_ms_sd": _metric_sd(report, arm, "latency_ms"),
                    "success_rate_swing": _swing(report, arm),
                }
                for arm in arms if arm in report
            },
            "variance": report.get("variance", {}).get("success_deltas_interpretable"),
        })
    reach = _tokens_to_reach(points, arms)
    return {"budgets": sorted(budgets), "arms": list(arms), "points": points,
            "max_items": max_items,
            "repetitions": repetitions,
            "tokens_to_reach": reach,
            # The headline. Success at a fixed budget is a precondition here, not
            # a contrast: it is reported only so a reader can see that an arm
            # reached the target at all before its budget is compared.
            "primary_endpoint": _primary_endpoint(points, arms, reach),
            "co_primary_endpoints": _co_primary(points, arms),
            "note": "mock runs measure whether required evidence survives the budget, "
                    "not model competence"}


def _safe_mean(arm_report: Mapping[str, Any], total_key: str) -> Optional[float]:
    attempts = arm_report.get("attempts") or 0
    if not attempts:
        return None
    return round((arm_report.get(total_key) or 0) / attempts, 2)


def _metric_sd(report: Mapping[str, Any], arm: str, metric: str) -> Optional[float]:
    entry = report.get("variance", {}).get("arms", {}).get(arm, {})
    return ((entry.get("metrics") or {}).get(metric) or {}).get("sd")


def _swing(report: Mapping[str, Any], arm: str) -> Optional[float]:
    entry = report.get("variance", {}).get("arms", {}).get(arm, {})
    return entry.get("success_rate_swing")


def _primary_endpoint(points: Sequence[Mapping[str, Any]], arms: Sequence[str],
                      reach: Mapping[str, Mapping[str, Any]],
                      target: float = 1.0) -> Dict[str, Any]:
    """Budget needed for full success, normalized against floor and ceiling.

    `oracle` is the information floor: it is handed the labeled fix site, so its
    budget is the fewest tokens under which the task is solvable at all. Any
    deployable arm's ratio to it is the size of the selection problem that
    remains. `full_file` is the ceiling: pack the whole repository and rank
    nothing. An arm above the ceiling ratio is worse than not retrieving.
    """
    key = "budget_for_success_rate_{:g}".format(target)
    budgets = {arm: reach.get(arm, {}).get(key) for arm in arms}
    floor = budgets.get("oracle")
    ceiling = budgets.get("full_file")
    rows: Dict[str, Any] = {}
    for arm in arms:
        budget = budgets.get(arm)
        rows[arm] = {
            "budget_to_reach_target": budget,
            # Null means the arm never reached the target inside the swept range,
            # which is a censored observation, not a large value: report it as
            # unreached rather than substituting the largest budget.
            "reached_target": budget is not None,
            "ratio_to_oracle_floor": (round(budget / floor, 2)
                                      if budget and floor else None),
            "ratio_to_full_file_ceiling": (round(budget / ceiling, 2)
                                           if budget and ceiling else None),
        }
    return {
        "target_success_rate": target,
        "metric": "smallest swept token budget at which the arm reaches the target",
        "floor_arm": "oracle" if floor else None,
        "ceiling_arm": "full_file" if ceiling else None,
        "arms": rows,
        "unreached_arms": [arm for arm in arms if budgets.get(arm) is None],
        "note": ("success rate is a precondition, not the contrast: an arm that "
                 "never reaches the target has no budget to compare, and one that "
                 "does is compared on tokens, not on wins"),
    }


def _co_primary(points: Sequence[Mapping[str, Any]],
                arms: Sequence[str]) -> Dict[str, Any]:
    """Output tokens and latency at the largest swept budget.

    Measured where every arm has the most room, so a difference cannot be
    explained by one arm being starved. These are per-trial means with their
    within-arm sd alongside, because a mean without dispersion is what made the
    original success comparison unreadable.
    """
    if not points:
        return {}
    largest = max(points, key=lambda point: point["token_budget"])
    return {
        "measured_at_token_budget": largest["token_budget"],
        "arms": {
            arm: {
                "mean_output_tokens": row.get("mean_output_tokens"),
                "output_tokens_sd": row.get("output_tokens_sd"),
                "mean_latency_ms": row.get("mean_latency_ms"),
                "latency_ms_sd": row.get("latency_ms_sd"),
                "success_rate": row.get("success_rate"),
            }
            for arm, row in largest["arms"].items() if arm in arms
        },
        "note": ("output tokens track context precision and reproduced across "
                 "two live runs; success at this budget did not"),
    }


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

    primary = result.get("primary_endpoint") or {}
    if primary:
        lines.append("")
        lines.append("PRIMARY: budget to reach {:.0%} success (oracle = floor, "
                     "full_file = ceiling)".format(primary["target_success_rate"]))
        lines.append("{:<22} {:>10} {:>14} {:>16}".format(
            "arm", "budget", "x oracle", "x full_file"))
        for arm in arms:
            row = (primary.get("arms") or {}).get(arm)
            if row is None:
                continue
            lines.append("{:<22} {:>10} {:>14} {:>16}".format(
                arm,
                "unreached" if row["budget_to_reach_target"] is None
                else str(row["budget_to_reach_target"]),
                "-" if row["ratio_to_oracle_floor"] is None
                else "{:.2f}".format(row["ratio_to_oracle_floor"]),
                "-" if row["ratio_to_full_file_ceiling"] is None
                else "{:.2f}".format(row["ratio_to_full_file_ceiling"]),
            ))
        if primary.get("unreached_arms"):
            lines.append("never reached the target: {}".format(
                ", ".join(primary["unreached_arms"])))

    co_primary = result.get("co_primary_endpoints") or {}
    if co_primary:
        lines.append("")
        lines.append("CO-PRIMARY at budget {}: output tokens and latency".format(
            co_primary["measured_at_token_budget"]))
        lines.append("{:<22} {:>18} {:>18}".format("arm", "output tokens", "latency ms"))
        for arm in arms:
            row = (co_primary.get("arms") or {}).get(arm)
            if row is None:
                continue
            lines.append("{:<22} {:>18} {:>18}".format(
                arm,
                "{}+-{}".format(_cell(row["mean_output_tokens"]), _cell(row["output_tokens_sd"])),
                "{}+-{}".format(_cell(row["mean_latency_ms"]), _cell(row["latency_ms_sd"])),
            ))
    if result.get("repetitions", 1) < 2:
        lines.append("")
        lines.append("WARNING: repetitions=1, so no within-arm variance was measured. "
                     "No delta in this table is interpretable; rerun with "
                     "--repetitions 5.")
    return "\n".join(lines)


def _cell(value: Optional[float]) -> str:
    if value is None:
        return "-"
    return "{:.0f}".format(value) if abs(value) >= 100 else "{:.2f}".format(value)


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
