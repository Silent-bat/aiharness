"""Tests for the SWE-bench Lite loader.

The two properties that decide whether a real-repository run means anything are
that test names never reach the prompt and that the sample is stratified and
reproducible. Both are checked here against a synthetic dataset, so the suite
needs no network and no dataset download.
"""

import json
import tempfile
import unittest
from pathlib import Path

from evaluation import swebench

PROBLEM = """`slugify` crashes on unicode input.

Reproduce with tests/test_slug.py::TestSlug::test_unicode_slug -- it raises
UnicodeDecodeError inside utils/text.py. See also test_slug_ascii, which passes.
"""


def _instance(instance_id, files, *, problem=PROBLEM, tests=None):
    patch = "".join(
        "diff --git a/{path} b/{path}\n--- a/{path}\n+++ b/{path}\n@@ -1 +1 @@\n-x\n+y\n".format(path=path)
        for path in files)
    return {
        "instance_id": instance_id,
        "repo": "example/{}".format(instance_id.split("-")[0]),
        "base_commit": "0" * 40,
        "problem_statement": problem,
        "patch": patch,
        "test_patch": "diff --git a/tests/test_slug.py b/tests/test_slug.py\n",
        "FAIL_TO_PASS": tests if tests is not None else ["tests/test_slug.py::TestSlug::test_unicode_slug"],
        "PASS_TO_PASS": ["tests/test_slug.py::TestSlug::test_slug_ascii"],
    }


class WithholdingTest(unittest.TestCase):
    def test_node_ids_class_names_and_function_names_are_all_removed(self):
        redacted = swebench.withhold_test_names(PROBLEM)
        for leak in ("test_unicode_slug", "TestSlug", "test_slug_ascii",
                     "tests/test_slug.py"):
            with self.subTest(leak=leak):
                self.assertNotIn(leak, redacted)

    def test_the_non_test_evidence_survives_redaction(self):
        # Over-broad redaction is the safe direction, but a prompt stripped of the
        # symbol under discussion would make every arm fail for the wrong reason.
        redacted = swebench.withhold_test_names(PROBLEM)
        self.assertIn("slugify", redacted)
        self.assertIn("utils/text.py", redacted)
        self.assertIn("UnicodeDecodeError", redacted)

    def test_the_leak_detector_agrees_with_the_redactor(self):
        node_ids = ["tests/test_slug.py::TestSlug::test_unicode_slug"]
        self.assertTrue(swebench.leaks_test_names(PROBLEM, node_ids))
        self.assertEqual(
            swebench.leaks_test_names(swebench.withhold_test_names(PROBLEM), node_ids), [])

    def test_the_detector_ignores_short_and_non_test_tokens(self):
        # "py" and "slug" appear everywhere; flagging them would block every case.
        self.assertEqual(swebench.leaks_test_names("a slug in a py file",
                                                   ["tests/test_slug.py::x"]), [])


class StratificationTest(unittest.TestCase):
    def test_test_files_do_not_count_toward_the_gold_patch_size(self):
        patch = ("diff --git a/pkg/core.py b/pkg/core.py\n"
                 "diff --git a/tests/test_core.py b/tests/test_core.py\n"
                 "diff --git a/pkg/conftest.py b/pkg/conftest.py\n")
        self.assertEqual(swebench.patch_files(patch), ["pkg/core.py"])

    def test_strata_boundaries(self):
        self.assertEqual(
            [swebench.stratum_of(n) for n in (0, 1, 2, 3, 9)],
            ["single_file", "single_file", "two_file", "multi_file", "multi_file"])

    def test_stratify_places_every_instance_in_exactly_one_stratum(self):
        instances = [_instance("a-1", ["one.py"]), _instance("b-2", ["one.py", "two.py"]),
                     _instance("c-3", ["one.py", "two.py", "three.py"])]
        buckets = swebench.stratify(instances)
        self.assertEqual([len(buckets[name]) for name in swebench.STRATA], [1, 1, 1])

    def test_sampling_is_reproducible_and_independent_of_file_order(self):
        instances = [_instance("i-{}".format(i), ["one.py"]) for i in range(20)]
        first = [row["instance_id"] for row in swebench.sample(swebench.stratify(instances), 5)]
        shuffled = list(reversed(instances))
        second = [row["instance_id"] for row in swebench.sample(swebench.stratify(shuffled), 5)]
        self.assertEqual(first, second)
        self.assertEqual(len(first), 5)

    def test_sampling_draws_from_every_stratum(self):
        instances = ([_instance("a-{}".format(i), ["one.py"]) for i in range(4)]
                     + [_instance("b-{}".format(i), ["one.py", "two.py"]) for i in range(4)]
                     + [_instance("c-{}".format(i), ["1.py", "2.py", "3.py"]) for i in range(4)])
        drawn = swebench.sample(swebench.stratify(instances), 2)
        strata = {swebench.stratum_of(len(row["_gold_files"])) for row in drawn}
        self.assertEqual(strata, set(swebench.STRATA))
        self.assertEqual(len(drawn), 6)


class CaseBuildingTest(unittest.TestCase):
    def setUp(self):
        instance = _instance("django-11", ["utils/text.py", "utils/encoding.py"])
        instance["_gold_files"] = swebench.patch_files(instance["patch"])
        self.case = swebench.build_case(instance, Path("/tmp/workspaces/django-11"))

    def test_the_prompt_is_redacted_and_the_leak_list_is_empty(self):
        self.assertNotIn("test_unicode_slug", self.case["prompt"])
        self.assertEqual(self.case["swebench"]["prompt_test_name_leaks"], [])

    def test_the_withheld_tests_are_kept_for_grading_only(self):
        self.assertIn("test_unicode_slug", self.case["test_command"])
        self.assertEqual(self.case["swebench"]["fail_to_pass"],
                         ["tests/test_slug.py::TestSlug::test_unicode_slug"])
        # The prompt is the only field rendered into the model's context.
        self.assertNotIn("test_unicode_slug", self.case["prompt"])

    def test_oracle_context_is_the_gold_patch_file_list(self):
        self.assertEqual(self.case["oracle_context"], ["utils/text.py", "utils/encoding.py"])
        self.assertEqual(self.case["labels"]["stratum"], "two_file")
        self.assertEqual(self.case["labels"]["gold_patch_file_count"], 2)

    def test_no_mock_fields_are_emitted(self):
        # A mock run here would grade the oracle labels against themselves.
        self.assertNotIn("mock_requires", self.case)
        self.assertNotIn("mock_files", self.case)

    def test_an_ungradeable_instance_is_rejected_rather_than_scored(self):
        instance = _instance("x-1", ["one.py"], tests=[])
        with self.assertRaises(ValueError):
            swebench.build_case(instance, Path("/tmp/x"))

    def test_the_case_uses_the_fields_the_harness_reads(self):
        for field in ("id", "repository", "prompt", "task_type", "test_command",
                      "oracle_context"):
            self.assertIn(field, self.case)


class PlanTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.dataset = Path(self.temporary.name) / "lite.json"
        instances = ([_instance("a-{}".format(i), ["one.py"]) for i in range(3)]
                     + [_instance("b-{}".format(i), ["one.py", "two.py"]) for i in range(3)])
        self.dataset.write_text(json.dumps(instances), encoding="utf-8")

    def tearDown(self):
        self.temporary.cleanup()

    def test_plan_fingerprints_the_dataset_so_a_sample_can_be_registered(self):
        report = swebench.plan_report(self.dataset, 2)
        self.assertTrue(report["dataset_fingerprint"].startswith("sha256:"))
        self.assertEqual(report["sampling_seed"], swebench.SAMPLING_SEED)
        self.assertEqual(report["prompt_test_name_leaks"], {})
        self.assertEqual(report["sampled_by_stratum"], {"single_file": 2, "two_file": 2})
        self.assertEqual(len(report["sampled_instance_ids"]), 4)

    def test_jsonl_and_json_datasets_load_identically(self):
        rows = json.loads(self.dataset.read_text(encoding="utf-8"))
        lines = Path(self.temporary.name) / "lite.jsonl"
        lines.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
        self.assertEqual(swebench.load_dataset(self.dataset), swebench.load_dataset(lines))

    def test_a_dataset_missing_required_fields_is_rejected_at_load(self):
        broken = Path(self.temporary.name) / "broken.json"
        broken.write_text(json.dumps([{"instance_id": "a-1"}]), encoding="utf-8")
        with self.assertRaises(ValueError):
            swebench.load_dataset(broken)

    def test_prepare_writes_cases_without_cloning(self):
        cases = swebench.prepare(self.dataset, Path(self.temporary.name) / "repos", 1,
                                 clone_repos=False)
        self.assertEqual(len(cases), 2)
        for case in cases:
            self.assertEqual(case["swebench"]["prompt_test_name_leaks"], [])


if __name__ == "__main__":
    unittest.main()
