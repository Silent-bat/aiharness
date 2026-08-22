"""Tests for the two agent arms and the token accounting they are measured with.

The arm numbers are only meaningful if three things hold: the repository map is
task-independent (or it cannot sit in a cached prefix), the transcript cost model
actually charges for re-sent context (or the whole comparison collapses to final
window size), and the success rule is body-level (or the map arm passes for free
on the signatures it emits by construction). Each is asserted here.
"""

import json
import tempfile
import unittest
from pathlib import Path

from context_compiler.repo_map import build_repo_map, callees
from context_compiler.service import ContextService

from evaluation.agent_arm import (Repository, Transcript, _identifiers,
                                 _required_evidence, run_grep_loop, run_map_expand,
                                 run_map_query, summarize)
from evaluation.haystack import generate

ROOT = Path(__file__).parents[1]
FIXTURE = ROOT / "evaluation" / "fixtures" / "empty_cart_bug"
CASES = json.loads((ROOT / "evaluation" / "task_cases.json").read_text(encoding="utf-8"))
CASE = (CASES["cases"] if isinstance(CASES, dict) else CASES)[0]


class RepoMapTest(unittest.TestCase):
    def setUp(self):
        self.service = ContextService(FIXTURE)
        self.service.index()

    def tearDown(self):
        self.service.close()

    def test_map_is_byte_identical_across_builds(self):
        # Nothing task-conditioned may leak in: a prefix that changes per task
        # cannot be read from cache, which is the map's entire economic argument.
        self.assertEqual(build_repo_map(self.service.store),
                         build_repo_map(self.service.store))

    def test_map_is_much_smaller_than_the_source_it_describes(self):
        source = sum(len(path.read_text(encoding="utf-8"))
                     for path in FIXTURE.rglob("*.py"))
        self.assertLess(len(build_repo_map(self.service.store)), source)

    def test_map_omits_function_bodies(self):
        # `validate_amount`'s body is the fix site. If the map carried it, the map
        # would be the evidence and progressive disclosure would be pointless.
        self.assertNotIn("amount must be positive", build_repo_map(self.service.store))

    def test_callees_walks_outbound_edges_only(self):
        reached = {node.name for node in callees(self.service.store, "submit_order")}
        self.assertIn("validate_amount", reached)


class TranscriptCostModelTest(unittest.TestCase):
    """Total input must grow quadratically in turns, because turn t re-sends 1..t-1."""

    def test_no_cache_cost_is_the_sum_of_prefixes(self):
        transcript = Transcript("system")
        base = transcript.final_tokens()
        transcript.act("call", "observation")
        first = transcript.no_cache_input
        transcript.act("call", "observation")
        second = transcript.no_cache_input - first
        self.assertEqual(first, base)
        # The second turn pays for the first turn's output as well as the system
        # prompt; a model that charged only for new bytes would report `first`.
        self.assertGreater(second, first)

    def test_cache_discounts_only_the_stable_prefix(self):
        transcript = Transcript("system", cacheable_prefix="x" * 3000)
        for _ in range(4):
            transcript.act("call", "observation")
        self.assertLess(transcript.cached_input, transcript.no_cache_input)

    def test_cache_credits_the_accumulated_transcript_not_only_the_header(self):
        """An arm with no static block still gets a discount as its transcript grows.

        Caching keys on a byte-identical prefix, and turn `t`'s input opens with turn
        `t-1`'s input verbatim -- so a growing transcript is cacheable too. Charging
        only a static header at the read rate would hand a large discount to an arm
        that carries a repository map and almost none to one that accumulates tool
        output, which measures where an arm's tokens sit rather than what caching
        does. This is the bug that made cached and uncached nearly equal for
        `grep_loop` and `map_query`.
        """
        transcript = Transcript("system")
        for _ in range(8):
            transcript.act("read_file a.py", "x" * 400)
        # Only the newest turn's bytes are ever billed fresh, so the discount has to
        # be large -- not the rounding error a header-only model produces.
        self.assertLess(transcript.cached_input, transcript.no_cache_input * 0.5)

    def test_warm_prefix_discounts_the_static_block_from_the_first_turn(self):
        """A map already in cache from an earlier case is a read, not a write."""
        prefix = "x" * 4000
        cold = Transcript("system", cacheable_prefix=prefix)
        warm = Transcript("system", cacheable_prefix=prefix, warm_prefix=True)
        for transcript in (cold, warm):
            transcript.act("call", "observation")
        self.assertLess(warm.cached_input, cold.cached_input)
        # The uncached price is a property of the transcript, not of the cache state.
        self.assertEqual(warm.no_cache_input, cold.no_cache_input)

    def test_identifiers_are_longest_first(self):
        terms = _identifiers("submit_order accepts an empty collection; reject it.")
        self.assertEqual(terms[0], "submit_order")


class SuccessRuleTest(unittest.TestCase):
    """The rule must require bodies, not the definition headers a map already has."""

    def setUp(self):
        self.service = ContextService(FIXTURE)
        self.service.index()

    def tearDown(self):
        self.service.close()

    def test_required_evidence_is_not_satisfied_by_the_map_alone(self):
        required = _required_evidence(CASE, self.service)
        self.assertTrue(required, msg="the case labels no symbols to require")
        map_only = Transcript("system", cacheable_prefix=build_repo_map(self.service.store))
        self.assertFalse(map_only.satisfies(required),
                         msg="signatures alone count as solved; the arms are not comparable")

    def test_required_evidence_is_scoped_to_labelled_files(self):
        required = _required_evidence(CASE, self.service)
        labels = CASE["labels"]
        self.assertTrue(set(label for label, _ in required)
                        <= set(labels["relevant_symbols"]) | set(labels["test_symbols"]))


class ArmBehaviourTest(unittest.TestCase):
    def setUp(self):
        self.repository = Repository(FIXTURE)
        self.service = ContextService(FIXTURE)
        self.service.index()
        self.required = _required_evidence(CASE, self.service)

    def tearDown(self):
        self.service.close()

    def test_grep_loop_reads_the_failing_test_first_when_told_which_it_is(self):
        transcript = run_grep_loop(self.repository, CASE, self.required)
        self.assertTrue(transcript.entries[0][0].startswith("read_file tests/"))

    def test_withholding_the_test_path_makes_the_grep_loop_more_expensive(self):
        hinted = run_grep_loop(self.repository, CASE, self.required, test_hint=True)
        blind = run_grep_loop(self.repository, CASE, self.required, test_hint=False)
        self.assertGreater(blind.no_cache_input, hinted.no_cache_input)

    def test_map_query_costs_less_than_carrying_the_map_in_the_prefix(self):
        # This is the finding the arms exist to produce: the map's value is
        # entry-point discovery, and a lookup buys that far cheaper than a dump.
        # It is a large-repository claim and must be tested on one: on the 6-file
        # fixture the map is so small that carrying it beats paying for a lookup
        # turn, and the ordering reverses.
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "haystack"
            generate(FIXTURE, root, modules=20, seed=3, prompt=CASE["prompt"])
            case = dict(CASE, repository=str(root))
            repository = Repository(root)
            service = ContextService(root)
            try:
                service.index()
                required = _required_evidence(case, service)
                queried = run_map_query(repository, case, service, required)
                carried = run_map_expand(repository, case, service, required)
            finally:
                service.close()
        self.assertTrue(queried.satisfies(required))
        self.assertTrue(carried.satisfies(required))
        self.assertLess(queried.no_cache_input, carried.no_cache_input)

    def test_carrying_the_map_wins_on_a_repository_small_enough_to_hold_it(self):
        queried = run_map_query(self.repository, CASE, self.service, self.required)
        carried = run_map_expand(self.repository, CASE, self.service, self.required)
        self.assertLess(carried.no_cache_input, queried.no_cache_input)

    def test_summary_averages_cost_over_solved_cases_only(self):
        rows = [{"arm": "a", "solved": True, "turns": 2, "no_cache_input_tokens": 100,
                 "cached_input_tokens": 50, "final_transcript_tokens": 60},
                {"arm": "a", "solved": False, "turns": 9, "no_cache_input_tokens": 9000,
                 "cached_input_tokens": 900, "final_transcript_tokens": 800}]
        summary = summarize(rows)
        self.assertEqual(summary["a"]["solved"], 1)
        self.assertEqual(summary["a"]["mean_no_cache_input_tokens"], 100)


if __name__ == "__main__":
    unittest.main()
