import json
import unittest
from pathlib import Path

from evaluation.variance import (
    arm_variance,
    dispersion,
    paired_bootstrap,
    per_repetition_success,
    variance_report,
)


def _row(task, arm, repetition, success, **extra):
    row = {"task_id": task, "arm": arm, "repetition": repetition, "success": success,
           "void": False, "output_tokens": 100, "latency_ms": 10.0,
           "context_tokens": 500, "input_tokens": 600, "total_prompt_tokens": 600}
    row.update(extra)
    return row


class DispersionTest(unittest.TestCase):
    def test_single_observation_has_no_standard_deviation(self):
        # One repetition is the state every published comparison in this project
        # was in; the answer must be "unmeasured", never 0.
        self.assertIsNone(dispersion([5.0])["sd"])

    def test_reports_mean_range_and_coefficient_of_variation(self):
        result = dispersion([10.0, 20.0, 30.0])
        self.assertEqual(result["mean"], 20.0)
        self.assertEqual(result["min"], 10.0)
        self.assertEqual(result["max"], 30.0)
        self.assertAlmostEqual(result["sd"], 10.0)
        self.assertAlmostEqual(result["cv"], 0.5)

    def test_missing_measurements_are_skipped_not_zeroed(self):
        # A void trial records no usage; treating None as 0 would drag the mean
        # down and understate the arm's cost.
        self.assertEqual(dispersion([100.0, None, 100.0])["mean"], 100.0)
        self.assertIsNone(dispersion([None, None]))


class PerRepetitionTest(unittest.TestCase):
    def test_void_trials_leave_both_numerator_and_denominator(self):
        rows = [_row("a", "classic", 1, True), _row("b", "classic", 1, False, void=True)]
        result = per_repetition_success(rows)
        self.assertEqual(result[1]["graded_attempts"], 1)
        self.assertEqual(result[1]["success_rate"], 1.0)

    def test_swing_is_the_spread_across_repetitions(self):
        rows = [_row("a", "classic", 1, True), _row("b", "classic", 1, True),
                _row("a", "classic", 2, True), _row("b", "classic", 2, False)]
        result = arm_variance(rows)
        self.assertEqual(result["success_rate_by_repetition"], [1.0, 0.5])
        self.assertEqual(result["success_rate_swing"], 0.5)


class BootstrapTest(unittest.TestCase):
    def test_interval_straddles_zero_for_a_sign_flipping_delta(self):
        interval = paired_bootstrap([-100.0, 100.0, -90.0, 95.0, -5.0, 3.0])
        self.assertFalse(interval["excludes_zero"])
        self.assertLess(interval["ci95"][0], 0)
        self.assertGreater(interval["ci95"][1], 0)

    def test_interval_excludes_zero_for_a_consistent_delta(self):
        interval = paired_bootstrap([-131.0, -54.0, -152.0, -96.0, -80.0, -26.0])
        # These are the real v2 input-token deltas: negative on all six pairs, so
        # the iso-token half of the claim should survive its own interval.
        self.assertTrue(interval["excludes_zero"])
        self.assertLess(interval["ci95"][1], 0)

    def test_is_reproducible_from_the_same_rows(self):
        deltas = [1.0, -2.0, 3.0, -4.0, 5.0]
        self.assertEqual(paired_bootstrap(deltas), paired_bootstrap(deltas))

    def test_one_pair_gives_no_interval(self):
        self.assertIsNone(paired_bootstrap([5.0]))


class VarianceGateTest(unittest.TestCase):
    def test_single_repetition_leaves_the_verdict_unmeasured(self):
        rows = [_row("a", "classic", 1, False), _row("a", "graphrag", 1, True),
                _row("b", "classic", 1, False), _row("b", "graphrag", 1, True)]
        report = variance_report(rows)
        # A 0-versus-1 success rate looks decisive and is not: with one run per
        # arm there is nothing to compare the gap against.
        self.assertIsNone(report["success_deltas_interpretable"])
        self.assertEqual(report["between_arm_success_spread"], 1.0)

    def test_gate_fails_when_within_arm_swing_dominates(self):
        rows = []
        for repetition, classic_wins in enumerate([0, 2], start=1):
            for index in range(2):
                rows.append(_row("t{}".format(index), "classic", repetition,
                                 index < classic_wins))
                rows.append(_row("t{}".format(index), "graphrag", repetition, True))
        report = variance_report(rows)
        # This is the live v1-versus-v2 shape: classic swings 0.0 -> 1.0 on its
        # own, which is larger than its 0.5 mean gap to graphrag.
        self.assertEqual(report["arms"]["classic"]["success_rate_swing"], 1.0)
        self.assertFalse(report["success_deltas_interpretable"])

    def test_gate_passes_when_arms_are_stable_and_separated(self):
        rows = []
        for repetition in (1, 2, 3):
            for index in range(2):
                rows.append(_row("t{}".format(index), "classic", repetition, False))
                rows.append(_row("t{}".format(index), "graphrag", repetition, True))
        report = variance_report(rows)
        self.assertEqual(report["max_within_arm_success_swing"], 0.0)
        self.assertTrue(report["success_deltas_interpretable"])

    def test_paired_intervals_cover_the_co_primary_endpoints(self):
        rows = []
        for repetition in (1, 2):
            for index in range(3):
                rows.append(_row("t{}".format(index), "classic", repetition, True,
                                 output_tokens=10000, latency_ms=90.0))
                rows.append(_row("t{}".format(index), "graphrag", repetition, True,
                                 output_tokens=3500, latency_ms=40.0))
        metrics = variance_report(rows)["paired_intervals"]["metrics"]
        self.assertLess(metrics["output_tokens"]["mean_delta"], 0)
        self.assertTrue(metrics["output_tokens"]["excludes_zero"])
        self.assertTrue(metrics["latency_ms"]["excludes_zero"])
        # Success is flat here, so its interval must not claim a direction.
        self.assertFalse(metrics["success"]["excludes_zero"])


class LiveRunRegressionTest(unittest.TestCase):
    """The two live runs are the reason this module exists; pin what they show."""

    def _rows(self, name):
        path = Path(__file__).parents[1] / "evaluation" / "results" / name
        return json.loads(path.read_text(encoding="utf-8"))["results"]

    def test_neither_live_run_alone_can_establish_a_success_delta(self):
        for name in ("live_haystack_ablation_4000.json",
                     "live_haystack_ablation_4000_v2.json"):
            report = variance_report(self._rows(name))
            self.assertIsNone(report["success_deltas_interpretable"],
                              msg="{} reported one repetition".format(name))

    def test_pooling_both_live_runs_shows_the_swing_exceeds_the_gap(self):
        rows = []
        for repetition, name in enumerate(("live_haystack_ablation_4000.json",
                                           "live_haystack_ablation_4000_v2.json"), start=1):
            for row in self._rows(name):
                pooled = dict(row)
                pooled["repetition"] = repetition
                rows.append(pooled)
        report = variance_report(rows)
        classic = report["arms"]["classic"]
        # 3/6 in run one against 6/6 in run two, voids excluded from both.
        self.assertGreaterEqual(classic["success_rate_swing"], 0.4)
        self.assertFalse(report["success_deltas_interpretable"])
        # What did hold: graph_only emitted far fewer output tokens in both runs.
        graph_only = report["arms"]["graph_only"]["metrics"]["output_tokens"]
        classic_output = classic["metrics"]["output_tokens"]
        self.assertLess(graph_only["mean"], classic_output["mean"] * 0.6)


class WithinTaskDispersionTest(unittest.TestCase):
    """Pooled sd mixes task difficulty with run noise; only the latter gates."""

    def test_pooled_sd_can_be_large_while_run_to_run_noise_is_zero(self):
        rows = []
        for repetition in (1, 2, 3):
            rows.append(_row("cheap", "classic", repetition, True, output_tokens=100))
            rows.append(_row("dear", "classic", repetition, True, output_tokens=9000))
        result = arm_variance(rows)
        # Tasks differ by 90x, so the pooled sd is huge...
        self.assertGreater(result["metrics"]["output_tokens"]["sd"], 4000)
        # ...but each task is perfectly reproducible, which is what matters.
        self.assertEqual(result["metrics_within_task"]["output_tokens"]["mean_within_task_sd"], 0.0)
        self.assertEqual(result["metrics_within_task"]["output_tokens"]["tasks_with_repeats"], 2)

    def test_single_repetition_gives_no_within_task_estimate(self):
        rows = [_row("a", "classic", 1, True), _row("b", "classic", 1, True)]
        self.assertIsNone(arm_variance(rows)["metrics_within_task"]["output_tokens"])


if __name__ == "__main__":
    unittest.main()
