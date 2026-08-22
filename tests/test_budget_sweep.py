"""Tests for the sweep's durability and for the endpoint it reports.

A live sweep is hours of paid calls. The first attempt at one wrote nothing until
the final budget point finished, so a single dropped socket on the first point
discarded every call already made. These pin the checkpoint that fixes that, and
the invariant that a finished run leaves no partial marker behind.
"""

import json
import tempfile
import unittest
from pathlib import Path

from evaluation import budget_sweep

TASKS = [
    {"task_id": "t1", "prompt": "submit_order accepts an empty collection; reject it.",
     "task_type": "debug", "required_symbols": ["submit_order"],
     "repository": "evaluation/fixtures/empty_cart_bug",
     "oracle_context": ["checkout.py"]},
]


class CheckpointTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.output = Path(self.temporary.name) / "nested" / "sweep.json"

    def tearDown(self):
        self.temporary.cleanup()

    def _stub_benchmark(self, fail_at=None):
        """Stand in for the real arms: the durability question is about the loop."""
        real = budget_sweep.benchmark
        seen = []

        def stub(tasks, **kwargs):
            budget = kwargs["token_budget"]
            seen.append(budget)
            if fail_at is not None and budget == fail_at:
                raise RuntimeError("the read operation timed out")
            report = {"token_budget_uniform": True, "token_budgets_used": [budget],
                      "arms": list(kwargs["arms"])}
            for arm in kwargs["arms"]:
                report[arm] = {"successes": 1, "attempts": 1, "success_rate": 1.0,
                               "mean_context_tokens": float(budget) / 2,
                               "mean_context_items": 3.0, "total_output_tokens": 10,
                               "mean_latency_ms": 1.0}
            return report

        budget_sweep.benchmark = stub
        self.addCleanup(lambda: setattr(budget_sweep, "benchmark", real))
        return seen

    def test_a_crash_leaves_the_completed_points_on_disk(self):
        self._stub_benchmark(fail_at=1024)
        with self.assertRaises(RuntimeError):
            budget_sweep.sweep(TASKS, [512, 1024, 2048], arms=["oracle"],
                               checkpoint=self.output)
        payload = json.loads(self.output.read_text(encoding="utf-8"))
        # The marker is what stops a partial file being read as a finished run.
        self.assertTrue(payload["partial"])
        self.assertEqual(payload["budgets_completed"], [512])
        self.assertEqual([point["token_budget"] for point in payload["points"]], [512])

    def test_a_finished_sweep_replaces_the_checkpoint_with_a_complete_payload(self):
        budgets = self._stub_benchmark()
        result = budget_sweep.sweep(TASKS, [512, 1024], arms=["oracle"],
                                    checkpoint=self.output)
        self.assertEqual(budgets, [512, 1024])
        payload = json.loads(self.output.read_text(encoding="utf-8"))
        self.assertNotIn("partial", payload)
        self.assertNotIn("budgets_completed", payload)
        self.assertEqual(payload["budgets"], result["budgets"])
        self.assertIn("tokens_to_reach", payload)

    def test_a_sweep_without_a_checkpoint_writes_nothing(self):
        self._stub_benchmark()
        budget_sweep.sweep(TASKS, [512], arms=["oracle"])
        self.assertFalse(self.output.exists())

    def test_the_checkpoint_records_the_stopping_rule_it_ran_under(self):
        # A stopping-on table must never be readable as a stopping-off one.
        self._stub_benchmark(fail_at=1024)
        rule = budget_sweep.StoppingRule()
        with self.assertRaises(RuntimeError):
            budget_sweep.sweep(TASKS, [512, 1024], arms=["oracle"],
                               stopping=rule, checkpoint=self.output)
        payload = json.loads(self.output.read_text(encoding="utf-8"))
        self.assertEqual(payload["stopping_rule"]["patience"], rule.patience)


if __name__ == "__main__":
    unittest.main()
