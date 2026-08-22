import json
import tempfile
import unittest
from pathlib import Path

from evaluation.task_benchmark import (
    ARMS,
    _apply_files,
    _paired_exact_test,
    _response_json,
    benchmark,
    run_task,
)


class _EmptyResponseClient:
    """A provider that returns no content, as OpenRouter occasionally does."""

    def complete(self, messages, *, system=None, max_tokens=None, temperature=None):
        from context_compiler.providers import ChatResponse

        return ChatResponse(text="")


class VoidTrialTest(unittest.TestCase):
    """An empty completion is a harness failure and must not be graded as a loss."""

    def test_empty_response_is_marked_void(self):
        fixture = Path(__file__).parents[1] / "evaluation" / "fixtures" / "empty_cart_bug"
        task = {"id": "empty-cart", "repository": str(fixture),
                "prompt": "reject an empty cart"}
        result = run_task(task, "classic", provider=_EmptyResponseClient(), mock=False,
                          codex_cli=False, token_budget=2000, max_tokens=None,
                          input_rate=None, output_rate=None)
        self.assertTrue(result.void)
        self.assertFalse(result.success)
        # Scoring it as a wrong answer would credit every other arm with a win it
        # did not earn: in the first live 8-arm run this cost `classic` one of its
        # three recorded failures.
        self.assertEqual(result.changed_files, 0)


class TaskBenchmarkTest(unittest.TestCase):
    def test_parses_plain_and_fenced_json(self):
        expected = {"files": {"a.py": "x = 1\n"}}
        self.assertEqual(_response_json(json.dumps(expected)), expected)
        self.assertEqual(_response_json("```json\n{}\n```".format(json.dumps(expected))), expected)

    def test_rejects_paths_outside_repository(self):
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaises(ValueError):
                _apply_files(Path(temporary), {"../escape.py": "bad"})

    def test_paired_exact_test_counts_graph_wins(self):
        result = _paired_exact_test([
            {"classic": False, "graphrag": True},
            {"classic": False, "graphrag": True},
            {"classic": True, "graphrag": True},
        ])
        self.assertEqual(result["graphrag_only_wins"], 2)
        self.assertEqual(result["classic_only_wins"], 0)
        self.assertEqual(result["two_sided_exact_p_value"], 0.5)

    def test_mock_benchmark_runs_both_arms_and_tests_edits(self):
        fixture = Path(__file__).parents[1] / "evaluation" / "fixtures" / "empty_cart_bug"
        task = {
            "id": "empty-cart",
            "repository": str(fixture),
            "prompt": "Fix empty cart behavior in submit_order and validate_amount",
            "task_type": "debug",
            "test_command": "python3 -m unittest discover -s tests -v",
            "mock_requires": ["submit_order", "validate_amount"],
            "mock_files": {
                "payments.py": (
                    "class PaymentGateway:\n"
                    "    def capture(self, amount: int) -> str:\n"
                    "        return \"payment-accepted\"\n\n\n"
                    "def validate_amount(amount: int) -> None:\n"
                    "    if amount <= 0:\n"
                    "        raise ValueError(\"amount must be positive\")\n"
                )
            },
        }
        result = benchmark([task], mock=True, token_budget=1200, max_tokens=None,
                           input_rate=1.0, output_rate=2.0)
        self.assertEqual(len(result["results"]), 2)
        self.assertTrue(all(row["success"] for row in result["results"]))
        self.assertTrue(all(row["changed_files"] == 1 for row in result["results"]))
        self.assertGreater(result["classic"]["total_input_tokens"], 0)

    def test_mock_benchmark_runs_every_ablation_arm(self):
        tasks = json.loads(
            (Path(__file__).parents[1] / "evaluation" / "task_cases.json").read_text()
        )
        task = dict(tasks[0])
        task["repository"] = str(Path(__file__).parents[1] / task["repository"])
        arms = sorted(ARMS)
        result = benchmark([task], mock=True, token_budget=4000, max_tokens=None,
                           input_rate=1.0, output_rate=2.0, arms=arms)
        self.assertEqual(result["arms"], arms)
        for arm in arms:
            self.assertIn(arm, result, msg="arm {} produced no summary".format(arm))
            self.assertEqual(result[arm]["attempts"], 1)
            self.assertEqual(result[arm]["retrieval_mode"], ARMS[arm]["mode"])
        # The oracle arm gives the model exactly the labeled fix site, so it must
        # succeed; the lexical-only arm cannot see the callee and must fail.
        self.assertEqual(result["oracle"]["successes"], 1)
        self.assertEqual(result["classic"]["successes"], 0)
        self.assertEqual(result["graphrag"]["successes"], 1)
        self.assertEqual(
            result["marginal_graph_contribution"]["classic_to_graphrag"]["success_rate_delta"], 1.0)

    def test_unknown_arm_is_rejected(self):
        with self.assertRaises(ValueError):
            benchmark([], mock=True, token_budget=1200, max_tokens=None,
                      input_rate=None, output_rate=None, arms=["telepathy"])

    def test_oracle_arm_is_skipped_when_task_has_no_labels(self):
        fixture = Path(__file__).parents[1] / "evaluation" / "fixtures" / "empty_cart_bug"
        task = {"id": "unlabeled", "repository": str(fixture), "prompt": "fix it",
                "task_type": "debug", "mock_requires": [], "mock_files": {}}
        result = benchmark([task], mock=True, token_budget=1200, max_tokens=None,
                           input_rate=None, output_rate=None, arms=["oracle", "classic"])
        self.assertNotIn("oracle", result)
        self.assertIn("classic", result)


if __name__ == "__main__":
    unittest.main()
