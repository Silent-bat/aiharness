"""Variance estimation for arm comparisons.

This module exists because of a specific failure. The first live 8-arm run read
`graphrag` 6/6 against `classic` 3/6, which was reported as an underpowered but
real effect. Re-running it unchanged moved `classic` to 6/6, taking the paired
exact p from 0.125 to 1.0. One arm's run-to-run swing was larger than the
between-arm effect the benchmark was built to detect, so the delta was never
interpretable in the first place.

The rule that follows is mechanical: no delta is reportable until the within-arm
swing has been measured and shown to be smaller than it. `variance_report`
computes both quantities and returns a verdict, so the gate is a value in the
output rather than a claim in prose.

It runs two ways: inline, as the `variance` block of a benchmark summary, and
post hoc as `python3 -m evaluation.variance results.json`, which re-derives the
same block from any run that recorded per-repetition rows.
"""

import argparse
import json
import math
import random
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

# Continuous per-trial measurements worth a dispersion estimate. Output tokens
# and latency are here because they are co-primary endpoints: they reproduced
# across two live runs at ~2.8x separation while success was pure noise.
CONTINUOUS_METRICS = (
    "output_tokens",
    "latency_ms",
    "context_tokens",
    "input_tokens",
    "total_prompt_tokens",
)


def dispersion(values: Sequence[Optional[float]]) -> Optional[Dict[str, Any]]:
    """Sample mean, sd, range, and coefficient of variation.

    Sample (n-1) rather than population sd: these repetitions are a sample of
    the run-to-run distribution, not the distribution itself.
    """
    present = [float(value) for value in values if value is not None]
    if not present:
        return None
    count = len(present)
    mean = sum(present) / count
    if count > 1:
        variance = sum((value - mean) ** 2 for value in present) / (count - 1)
        deviation = math.sqrt(variance)
    else:
        deviation = None
    return {
        "n": count,
        "mean": round(mean, 4),
        "sd": None if deviation is None else round(deviation, 4),
        "min": round(min(present), 4),
        "max": round(max(present), 4),
        # Scale-free, so output tokens and latency are comparable at a glance.
        "cv": (None if deviation is None or mean == 0
               else round(deviation / abs(mean), 4)),
    }


def per_repetition_success(rows: Sequence[Mapping[str, Any]]) -> Dict[int, Dict[str, Any]]:
    """Success rate within each repetition index, over that repetition's tasks.

    A repetition is one sweep over every task, so its success rate is the
    quantity a single run of the benchmark would have reported. The spread
    across repetitions is exactly the noise that produced the 3/6-to-6/6 swing.
    """
    grouped: Dict[int, List[Mapping[str, Any]]] = {}
    for row in rows:
        if row.get("void"):
            # A void trial measured nothing, so it belongs in neither numerator
            # nor denominator; counting it as a loss is how provider flakiness
            # gets attributed to whichever arm happened to draw it.
            continue
        grouped.setdefault(int(row["repetition"]), []).append(row)
    output: Dict[int, Dict[str, Any]] = {}
    for repetition, group in sorted(grouped.items()):
        successes = sum(1 for row in group if row["success"])
        output[repetition] = {
            "successes": successes,
            "graded_attempts": len(group),
            "success_rate": round(successes / len(group), 4) if group else None,
        }
    return output


def paired_bootstrap(deltas: Sequence[float], *, iterations: int = 4000,
                     seed: int = 20260822) -> Optional[Dict[str, Any]]:
    """Percentile bootstrap CI for a mean paired difference.

    Paired, so the resample unit is the (task, repetition) difference rather
    than either arm's absolute value: repository size dominates the absolute
    numbers and would swamp the effect. Seeded, so a reported interval is
    reproducible from the same rows.
    """
    sample = [float(delta) for delta in deltas]
    if len(sample) < 2:
        return None
    generator = random.Random(seed)
    count = len(sample)
    means: List[float] = []
    for _ in range(iterations):
        means.append(
            sum(sample[generator.randrange(count)] for _ in range(count)) / count
        )
    means.sort()
    low = means[int(0.025 * (iterations - 1))]
    high = means[int(0.975 * (iterations - 1))]
    return {
        "n_pairs": count,
        "mean_delta": round(sum(sample) / count, 4),
        "ci95": [round(low, 4), round(high, 4)],
        # The interval straddling zero is the whole point of computing it: it
        # says the sign of this delta is not established by these pairs.
        "excludes_zero": bool(low > 0 or high < 0),
        "iterations": iterations,
        "seed": seed,
    }


def arm_variance(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    """Within-arm dispersion for one arm's rows across repetitions."""
    repetitions = per_repetition_success(rows)
    rates = [entry["success_rate"] for entry in repetitions.values()
             if entry["success_rate"] is not None]
    swing = (round(max(rates) - min(rates), 4) if len(rates) > 1 else None)
    return {
        "repetitions": {str(index): entry for index, entry in repetitions.items()},
        "success_rate_by_repetition": rates,
        "success_rate_sd": (dispersion(rates) or {}).get("sd"),
        # The headline number: how far a single run of this arm can move without
        # anything changing. Any between-arm delta smaller than this is noise.
        "success_rate_swing": swing,
        "metrics": {
            metric: dispersion([row.get(metric) for row in rows
                                if not row.get("void")])
            for metric in CONTINUOUS_METRICS
        },
        # `metrics` pools trials, so its sd mixes two sources: how much tasks
        # differ from each other, and how much one task moves between runs. Only
        # the second is run-to-run noise, and it is the one a delta has to clear.
        "metrics_within_task": {
            metric: _within_task_dispersion(rows, metric)
            for metric in CONTINUOUS_METRICS
        },
    }


def _within_task_dispersion(rows: Sequence[Mapping[str, Any]],
                            metric: str) -> Optional[Dict[str, Any]]:
    """Repetition-to-repetition dispersion of one metric, averaged over tasks.

    Task difficulty is held fixed by grouping on `task_id` first, so what is left
    is the same task measured again. Returns null with a single repetition,
    because then there is nothing to compare a task against.
    """
    grouped: Dict[Any, List[float]] = {}
    for row in rows:
        if row.get("void") or row.get(metric) is None:
            continue
        grouped.setdefault(row["task_id"], []).append(float(row[metric]))
    per_task = [dispersion(values) for values in grouped.values() if len(values) > 1]
    deviations = [entry["sd"] for entry in per_task if entry and entry["sd"] is not None]
    if not deviations:
        return None
    relative = [entry["cv"] for entry in per_task if entry and entry["cv"] is not None]
    return {
        "tasks_with_repeats": len(per_task),
        "mean_within_task_sd": round(sum(deviations) / len(deviations), 4),
        "max_within_task_sd": round(max(deviations), 4),
        "mean_within_task_cv": (round(sum(relative) / len(relative), 4)
                                if relative else None),
    }


def variance_report(rows: Sequence[Mapping[str, Any]], *,
                    arms: Optional[Sequence[str]] = None,
                    baseline_arm: str = "classic",
                    treatment_arm: str = "graphrag") -> Dict[str, Any]:
    """Within-arm variance, between-arm spread, and the gate verdict."""
    present = list(dict.fromkeys(row["arm"] for row in rows))
    selected = [arm for arm in (arms or present) if arm in present]
    per_arm = {arm: arm_variance([row for row in rows if row["arm"] == arm])
               for arm in selected}
    repetition_count = len({int(row["repetition"]) for row in rows})

    swings = [entry["success_rate_swing"] for entry in per_arm.values()
              if entry["success_rate_swing"] is not None]
    means = [rate for entry in per_arm.values()
             for rate in [_mean(entry["success_rate_by_repetition"])] if rate is not None]
    within = max(swings) if swings else None
    between = round(max(means) - min(means), 4) if len(means) > 1 else None

    report: Dict[str, Any] = {
        "repetitions": repetition_count,
        "arms": per_arm,
        "max_within_arm_success_swing": within,
        "between_arm_success_spread": between,
        # The gate. With one repetition there is no variance estimate at all, so
        # the answer is not "yes" but "unmeasured" -- which is the state every
        # single-run comparison in this project was published in.
        "success_deltas_interpretable": (
            None if within is None or between is None else bool(between > within)
        ),
        "note": (
            "A success delta is interpretable only when it exceeds the largest "
            "within-arm swing. Continuous endpoints (output tokens, latency) "
            "carry their own dispersion under arms.<arm>.metrics and should be "
            "preferred when this verdict is false or null."
        ),
    }
    report.update(_paired_metric_intervals(rows, baseline_arm, treatment_arm))
    return report


def _paired_metric_intervals(rows: Sequence[Mapping[str, Any]], baseline_arm: str,
                             treatment_arm: str) -> Dict[str, Any]:
    """Bootstrap CIs for treatment-minus-baseline on every paired metric."""
    by_pair: Dict[Any, Dict[str, Mapping[str, Any]]] = {}
    for row in rows:
        by_pair.setdefault((row["task_id"], row["repetition"]), {})[row["arm"]] = row
    pairs = [entry for _key, entry in sorted(by_pair.items())
             if baseline_arm in entry and treatment_arm in entry
             and not entry[baseline_arm].get("void")
             and not entry[treatment_arm].get("void")]
    if not pairs:
        return {}
    intervals: Dict[str, Any] = {}
    for metric in ("success",) + CONTINUOUS_METRICS:
        deltas = [
            float(entry[treatment_arm][metric] or 0) - float(entry[baseline_arm][metric] or 0)
            for entry in pairs
            if entry[treatment_arm].get(metric) is not None
            and entry[baseline_arm].get(metric) is not None
        ]
        interval = paired_bootstrap(deltas)
        if interval is not None:
            intervals[metric] = interval
    return {"paired_intervals": {
        "baseline_arm": baseline_arm,
        "treatment_arm": treatment_arm,
        "metrics": intervals,
    }}


def _mean(values: Sequence[Optional[float]]) -> Optional[float]:
    present = [float(value) for value in values if value is not None]
    return sum(present) / len(present) if present else None


def render_table(report: Mapping[str, Any]) -> str:
    """Human-readable summary: swing first, because it gates everything else."""
    lines = [
        "repetitions: {}".format(report["repetitions"]),
        "{:<22} {:>18} {:>10} {:>18} {:>16}".format(
            "arm", "success by rep", "swing", "output tokens", "latency ms"),
    ]
    for arm, entry in report["arms"].items():
        rates = "/".join("{:.2f}".format(rate)
                         for rate in entry["success_rate_by_repetition"])
        output = entry["metrics"].get("output_tokens") or {}
        latency = entry["metrics"].get("latency_ms") or {}
        lines.append("{:<22} {:>18} {:>10} {:>18} {:>16}".format(
            arm,
            rates or "-",
            "-" if entry["success_rate_swing"] is None else "{:.2f}".format(entry["success_rate_swing"]),
            "{}+-{}".format(_short(output.get("mean")), _short(output.get("sd"))),
            "{}+-{}".format(_short(latency.get("mean")), _short(latency.get("sd"))),
        ))
    lines.append("")
    lines.append("max within-arm success swing:  {}".format(report["max_within_arm_success_swing"]))
    lines.append("between-arm success spread:    {}".format(report["between_arm_success_spread"]))
    verdict = report["success_deltas_interpretable"]
    lines.append("success deltas interpretable:  {}".format(
        "UNMEASURED (need --repetitions > 1)" if verdict is None
        else ("yes" if verdict else "NO -- within-arm noise dominates")))
    paired = report.get("paired_intervals")
    if paired:
        lines.append("")
        lines.append("paired {} minus {} (bootstrap 95% CI):".format(
            paired["treatment_arm"], paired["baseline_arm"]))
        for metric, interval in paired["metrics"].items():
            lines.append("  {:<22} {:>12}  [{}, {}]{}".format(
                metric, _short(interval["mean_delta"]),
                _short(interval["ci95"][0]), _short(interval["ci95"][1]),
                "" if interval["excludes_zero"] else "   (straddles zero)"))
    return "\n".join(lines)


def _short(value: Optional[float]) -> str:
    if value is None:
        return "-"
    return "{:.0f}".format(value) if abs(value) >= 100 else "{:.2f}".format(value)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result", type=Path,
                        help="a task_benchmark.py output JSON with per-trial results")
    parser.add_argument("--arms", nargs="+")
    parser.add_argument("--baseline-arm", default="classic")
    parser.add_argument("--treatment-arm", default="graphrag")
    parser.add_argument("--json", action="store_true")
    arguments = parser.parse_args()
    payload = json.loads(arguments.result.read_text(encoding="utf-8"))
    rows = payload["results"] if isinstance(payload, dict) else payload
    report = variance_report(rows, arms=arguments.arms,
                             baseline_arm=arguments.baseline_arm,
                             treatment_arm=arguments.treatment_arm)
    print(json.dumps(report, indent=2, sort_keys=True) if arguments.json
          else render_table(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
