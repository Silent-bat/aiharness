"""Tests for the haystack generator and the retrieval defects it exposed.

The six seeded fixtures are 129-281 tokens against a 4,000-token budget, so the
budget never binds there and `full_file` wins by brute force. The haystack exists
to make the budget bind. Nothing about a haystack result is reportable unless the
generated repository is valid Python and the seeded test still fails before the
patch, so both are asserted here.
"""

import ast
import collections
import compileall
import contextlib
import io
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from context_compiler.service import ContextService

from evaluation.haystack import build_cases, generate

ROOT = Path(__file__).parents[1]
FIXTURE = ROOT / "evaluation" / "fixtures" / "empty_cart_bug"
PROMPT = "submit_order accepts an empty collection; reject it."


class HaystackGeneratorTest(unittest.TestCase):
    """A generator that emits invalid Python would silently void every result."""

    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temporary.name) / "haystack_empty_cart_bug"
        # 40 modules is enough to exercise decoys and cross-module edges while
        # keeping the test fast; the reported runs use 120.
        cls.report = generate(FIXTURE, cls.root, modules=40, seed=7, prompt=PROMPT)

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def test_generation_is_deterministic_for_a_seed(self):
        with tempfile.TemporaryDirectory() as other:
            twin = Path(other) / self.root.name
            generate(FIXTURE, twin, modules=40, seed=7, prompt=PROMPT)
            for path in sorted(self.root.rglob("*.py")):
                relative = path.relative_to(self.root)
                self.assertEqual(path.read_text(encoding="utf-8"),
                                 (twin / relative).read_text(encoding="utf-8"),
                                 msg="{} differs across runs at the same seed".format(relative))

    def test_every_generated_module_compiles(self):
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            ok = compileall.compile_dir(str(self.root), quiet=2, force=True)
        self.assertTrue(ok, msg=buffer.getvalue())

    def test_every_generated_module_imports(self):
        # Import is stricter than compile: a decoy that dropped a planned name
        # breaks the importer of that name, and only import catches it.
        script = (
            "import importlib, sys\n"
            "sys.path.insert(0, {!r})\n"
            "for index in range({}):\n"
            "    importlib.import_module('modules.mod_%03d' % index)\n"
        ).format(str(self.root), self.report["generated_modules"])
        completed = subprocess.run([sys.executable, "-c", script],
                                   capture_output=True, text=True)
        self.assertEqual(completed.returncode, 0, msg=completed.stderr)

    def test_no_module_defines_one_name_twice(self):
        # Two defs of one name is legal Python but collapses the module's exported
        # surface, and it used to abort indexing with a UNIQUE constraint failure.
        for path in sorted(self.root.rglob("*.py")):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            names = [node.name for node in tree.body
                     if isinstance(node, (ast.FunctionDef, ast.ClassDef))]
            duplicates = [name for name, count
                          in collections.Counter(names).items() if count > 1]
            self.assertEqual(duplicates, [], msg="{} redefines {}".format(path.name, duplicates))

    def test_repository_is_far_larger_than_any_reported_budget(self):
        # If the repository fits the budget, `full_file` is both the information
        # ceiling and near the cost floor, and retrieval cannot win by design.
        # 40 modules reach roughly 12k tokens; the reported 120-module runs reach
        # roughly 39k against a 4,000-token budget.
        self.assertGreater(self.report["approximate_tokens"], 8 * 1024)

    def test_decoy_modules_outrank_the_real_fix_site_lexically(self):
        service = ContextService(self.root)
        try:
            service.index()
            ranked = service.store.search(PROMPT, limit=12)
        finally:
            service.close()
        paths = [node.path for node, _ in ranked]
        self.assertTrue(any(path.startswith("modules/") for path in paths),
                        msg="the haystack is not competing with the fix site at all")

    def test_graph_reaches_the_callee_when_decoys_dominate_lexical_seeds(self):
        # The defect the haystack exposed: seeds were taken in raw bm25 row order,
        # which the decoy modules monopolize, so the real fix site was never a seed
        # and traversal never reached its callee. Seeds now come from lexical score.
        service = ContextService(self.root)
        try:
            service.index()
            compiled = {
                mode: service.compile_context(
                    PROMPT, token_budget=4000, task_type="debug",
                    graph_depth=2 if mode.endswith("graph") else 0,
                    max_items=200, mode=mode)
                for mode in ("lexical", "lexical_graph")
            }
        finally:
            service.close()
        lexical_paths = {item["path"] for item in compiled["lexical"]["items"]}
        graph_paths = {item["path"] for item in compiled["lexical_graph"]["items"]}
        self.assertNotIn("payments.py", lexical_paths)
        self.assertIn("payments.py", graph_paths)

    def test_seeded_test_still_fails_before_any_patch(self):
        completed = subprocess.run(
            [sys.executable, "-m", "unittest", "discover", "-s", "tests"],
            cwd=str(self.root), capture_output=True, text=True)
        self.assertNotEqual(completed.returncode, 0,
                            msg="the seeded bug did not survive generation")

    def test_build_cases_drops_per_task_budget_overrides(self):
        tasks = [{"id": "debug-empty-cart", "repository": str(FIXTURE),
                  "prompt": PROMPT, "token_budget": 700}]
        with tempfile.TemporaryDirectory() as output:
            built = build_cases(tasks, Path(output), modules=4, seed=1)
        self.assertNotIn("token_budget", built["cases"][0],
                         msg="a per-task budget would defeat the point of the haystack")


class LexicalScoreOrientationTest(unittest.TestCase):
    """bm25() is negative and more negative is better."""

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        (self.root / "target.py").write_text(
            "def submit_order(items):\n    return len(items)\n", encoding="utf-8")
        (self.root / "other.py").write_text(
            "def unrelated_helper(value):\n    return value\n", encoding="utf-8")
        self.service = ContextService(self.root)
        self.service.index()

    def tearDown(self):
        self.service.close()
        self.temporary.cleanup()

    def test_score_decreases_with_row_position(self):
        ranked = self.service.store.search("submit_order items", limit=20)
        scores = [score for _, score in ranked]
        self.assertEqual(scores, sorted(scores, reverse=True),
                         msg="row order is best-first, so score must not increase down the list")


class DuplicateNodeKeyTest(unittest.TestCase):
    """One file legally defining a name twice must not abort the index run."""

    def test_duplicate_keys_are_collapsed_instead_of_raising(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "twice.py").write_text(
                "def handler(value):\n    return value\n\n\n"
                "def handler(value):\n    return value + 1\n", encoding="utf-8")
            service = ContextService(root)
            try:
                report = service.index()
                names = [node.name for node in service.store.all_nodes(["function"])]
            finally:
                service.close()
        self.assertEqual(report["errors"], [])
        self.assertEqual(names.count("handler"), 1)


if __name__ == "__main__":
    unittest.main()
