import json
import shutil
import tempfile
import unittest
from pathlib import Path

from context_compiler.compiler import ALL_CONTEXT_MODES
from context_compiler.retrieval import RETRIEVAL_MODES, HybridRetriever
from context_compiler.service import ContextService
from context_compiler.vectors import VectorIndex, split_identifier, tokenize
from evaluation.benchmark import benchmark as retrieval_benchmark
from evaluation.budget_sweep import render_table, sweep
from evaluation.task_benchmark import ARMS

FIXTURE = Path(__file__).parents[1] / "evaluation" / "fixtures" / "empty_cart_bug"
PROMPT = "submit_order accepts an empty collection; reject it."
ORACLE = ["checkout.py", "payments.py", "tests/test_checkout.py"]


class VectorTest(unittest.TestCase):
    def test_split_identifier_handles_snake_and_camel_case(self):
        self.assertEqual(split_identifier("sanitize_payload"), ["sanitize", "payload"])
        self.assertEqual(split_identifier("sanitizePayload"), ["sanitize", "payload"])
        self.assertEqual(split_identifier("tests/test_api.py"), ["tests", "test", "api", "py"])

    def test_tokenize_emits_whole_and_subword_units(self):
        tokens = set(tokenize("def sanitize_payload(request):"))
        self.assertIn("sanitize_payload", tokens)
        self.assertIn("sanitize", tokens)
        self.assertIn("payload", tokens)

    def test_empty_index_returns_no_hits(self):
        self.assertEqual(VectorIndex([]).search("anything"), [])


class AblationModeTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name) / FIXTURE.name
        shutil.copytree(FIXTURE, self.root)
        self.service = ContextService(self.root)
        self.service.index()

    def tearDown(self):
        self.service.close()
        self.temporary.cleanup()

    def compile(self, mode, **kwargs):
        return self.service.compile_context(
            PROMPT, token_budget=4000, task_type="debug",
            graph_depth=2 if mode.endswith("graph") else 0,
            mode=mode, oracle_targets=ORACLE, **kwargs)

    def test_every_declared_mode_compiles_non_empty_context(self):
        for mode in ALL_CONTEXT_MODES:
            with self.subTest(mode=mode):
                compiled = self.compile(mode)
                self.assertGreater(compiled["item_count"], 0)
                self.assertEqual(compiled["retrieval_mode"], mode)
                self.assertLessEqual(compiled["estimated_tokens"], compiled["token_budget"])

    def test_benchmark_arms_cover_every_retrieval_mode(self):
        declared = {arm["mode"] for arm in ARMS.values()}
        self.assertTrue(set(RETRIEVAL_MODES).issubset(declared))
        self.assertIn("oracle", declared)
        self.assertIn("full_file", declared)

    def test_graph_only_mode_keeps_only_graph_reachable_evidence(self):
        compiled = self.compile("graph")
        self.assertGreater(compiled["item_count"], 0)
        for item in compiled["items"]:
            self.assertTrue(
                any(reason.startswith("graph ") for reason in item["reasons"]),
                msg="graph-only arm leaked a non-traversal hit: {}".format(item["reasons"]),
            )

    def test_graph_traversal_reaches_the_callee_lexical_search_misses(self):
        lexical_paths = {item["path"] for item in self.compile("lexical")["items"]}
        graph_paths = {item["path"] for item in self.compile("lexical_graph")["items"]}
        self.assertNotIn("payments.py", lexical_paths)
        self.assertIn("payments.py", graph_paths)

    def test_oracle_mode_selects_exactly_the_labeled_files(self):
        paths = {item["path"] for item in self.compile("oracle")["items"]}
        self.assertEqual(paths, set(ORACLE))

    def test_oracle_mode_rejects_unknown_targets(self):
        with self.assertRaises(ValueError):
            self.service.compile_context(
                PROMPT, task_type="debug", mode="oracle",
                oracle_targets=["does_not_exist.py"])

    def test_oracle_mode_requires_targets(self):
        with self.assertRaises(ValueError):
            self.service.compile_context(PROMPT, task_type="debug", mode="oracle")

    def test_full_file_mode_includes_every_source_file(self):
        paths = {item["path"] for item in self.compile("full_file")["items"]}
        self.assertIn("checkout.py", paths)
        self.assertIn("payments.py", paths)

    def test_unknown_mode_is_rejected(self):
        with self.assertRaises(ValueError):
            self.compile("telepathy")

    def test_resolve_mode_preserves_legacy_graph_depth_semantics(self):
        retriever = HybridRetriever(self.service.store)
        self.assertEqual(retriever.resolve_mode(None, 0).name, "lexical")
        legacy = retriever.resolve_mode(None, 3)
        self.assertEqual(legacy.name, "lexical_graph")
        self.assertEqual(legacy.graph_depth, 3)


class RetrievalBenchmarkAblationTest(unittest.TestCase):
    def test_all_modes_are_scored_and_hybrid_matches_lexical_graph(self):
        root = Path(__file__).parents[1]
        report = retrieval_benchmark(
            root / "examples" / "sample_repo",
            root / "evaluation" / "sample_cases.json",
            limit=10, token_budget=1200, modes=sorted(RETRIEVAL_MODES))
        for mode in RETRIEVAL_MODES:
            self.assertIn(mode, report, msg="mode {} was not scored".format(mode))
            self.assertIn("mean_recall_at_k", report[mode])
        # `hybrid` is the legacy name for the lexical+graph arm; the refactor must
        # not have changed what it measures.
        self.assertAlmostEqual(report["hybrid"]["mean_recall_at_k"],
                               report["lexical_graph"]["mean_recall_at_k"])
        self.assertGreater(report["lexical_graph"]["mean_recall_at_k"],
                           report["lexical"]["mean_recall_at_k"])


class BudgetSweepTest(unittest.TestCase):
    """The budget sweep is the arm comparison that is actually decidable."""

    @classmethod
    def setUpClass(cls):
        root = Path(__file__).parents[1]
        tasks = json.loads((root / "evaluation" / "task_cases.json").read_text())
        for task in tasks:
            task["repository"] = str(root / task["repository"])
        cls.result = sweep(tasks, (256, 512, 1024),
                           arms=["full_file", "classic", "graphrag", "oracle"], mock=True)

    def test_sweep_reports_one_point_per_budget(self):
        self.assertEqual([point["token_budget"] for point in self.result["points"]],
                         [256, 512, 1024])

    def test_per_task_budget_overrides_are_ignored(self):
        # Every point must observe the swept budget, otherwise the comparison is
        # not iso-token and the sweep is meaningless.
        for point in self.result["points"]:
            for arm, row in point["arms"].items():
                self.assertLessEqual(row["mean_context_tokens"], point["token_budget"],
                                     msg="{} exceeded the swept budget".format(arm))

    def test_lexical_only_never_reaches_the_fix_site(self):
        reached = self.result["tokens_to_reach"]["classic"]
        self.assertIsNone(reached["budget_for_success_rate_0.5"])

    def test_oracle_reaches_full_success_at_or_below_graphrag_budget(self):
        oracle = self.result["tokens_to_reach"]["oracle"]["budget_for_success_rate_1"]
        graphrag = self.result["tokens_to_reach"]["graphrag"]["budget_for_success_rate_1"]
        self.assertIsNotNone(oracle)
        self.assertIsNotNone(graphrag)
        # Headroom left in evidence selection: the graph reaches the right symbol
        # but spends more budget than a human selection needs.
        self.assertLessEqual(oracle, graphrag)

    def test_render_table_lists_every_arm(self):
        table = render_table(self.result)
        for arm in self.result["arms"]:
            self.assertIn(arm, table)


if __name__ == "__main__":
    unittest.main()
