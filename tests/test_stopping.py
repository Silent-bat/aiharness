"""Tests for the sufficiency stopping rule.

Every arm except `oracle` and `graph_only` fills whatever budget it is handed,
which is why raising the budget raised cost without raising success. The rule
makes stopping a decision rather than an accident. These tests pin the decision
itself -- which condition fired, and that budget was left unspent -- rather than
only the downstream success counts, because success on six tasks cannot separate
a rule that works from one that got lucky.
"""

import shutil
import tempfile
import unittest
from pathlib import Path

from context_compiler.compiler import (CLOSURE_EDGES, ContextCompiler, StoppingRule,
                                       StoppingDiagnostics)
from context_compiler.service import ContextService

FIXTURE = Path(__file__).parents[1] / "evaluation" / "fixtures" / "empty_cart_bug"
PROMPT = "submit_order accepts an empty collection; reject it."


class StoppingRuleDefaultsTest(unittest.TestCase):
    def test_closure_edges_are_outbound_dependency_kinds(self):
        # Inbound callers matter for impact analysis, not for reading the
        # evidence in hand, so the closure set is deliberately narrow.
        self.assertEqual(CLOSURE_EDGES,
                         ("resolved_calls", "resolved_imports", "resolved_inherits"))

    def test_rule_is_off_unless_a_caller_asks_for_it(self):
        compiler = ContextCompiler.__new__(ContextCompiler)
        ContextCompiler.__init__(compiler, retriever=None)
        self.assertFalse(compiler.stopping.enabled)

    def test_diagnostics_default_to_exhaustion_not_to_a_rule(self):
        # A run with no rule active must not claim a rule stopped it.
        self.assertEqual(StoppingDiagnostics().as_dict()["reason"],
                         "candidates_exhausted")


class ConditionTest(unittest.TestCase):
    """Each condition in isolation, since end-to-end runs fire several at once."""

    class _Hit:
        def __init__(self, score):
            self.final_score = score

    def _stop(self, rule, score, open_refs, flat, best=1.0):
        return ContextCompiler._sufficiency_stop(rule, self._Hit(score), best,
                                                 open_refs, flat)

    def test_closure_fires_when_nothing_is_outstanding(self):
        rule = StoppingRule()
        self.assertEqual(self._stop(rule, 0.9, set(), 0), "closure")

    def test_closure_can_be_disabled_without_disabling_the_rule(self):
        rule = StoppingRule(require_closure=False)
        self.assertIsNone(self._stop(rule, 0.9, set(), 0))

    def test_cliff_fires_below_the_ratio_and_not_above_it(self):
        rule = StoppingRule(cliff_ratio=0.15)
        self.assertEqual(self._stop(rule, 0.10, {"open"}, 0), "score_cliff")
        self.assertIsNone(self._stop(rule, 0.20, {"open"}, 0))

    def test_cliff_is_inert_when_the_best_score_is_zero(self):
        # Otherwise a run where nothing scored would stop on the first candidate.
        rule = StoppingRule()
        self.assertIsNone(self._stop(rule, 0.0, {"open"}, 0, best=0.0))

    def test_patience_fires_on_a_flat_tail(self):
        rule = StoppingRule(patience=4)
        self.assertIsNone(self._stop(rule, 0.9, {"open"}, 3))
        self.assertEqual(self._stop(rule, 0.9, {"open"}, 4), "no_new_information")

    def test_closure_outranks_the_other_conditions(self):
        # Reported reasons are used to explain runs, so precedence must be fixed.
        rule = StoppingRule()
        self.assertEqual(self._stop(rule, 0.01, set(), 99), "closure")


class EndToEndStoppingTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temporary.name) / FIXTURE.name
        shutil.copytree(FIXTURE, cls.root)
        cls.off = ContextService(cls.root)
        cls.off.index()
        cls.on = ContextService(cls.root, database=cls.off.database,
                                stopping=StoppingRule())

    @classmethod
    def tearDownClass(cls):
        cls.on.close()
        cls.off.close()
        cls.temporary.cleanup()

    def _compile(self, service, **kwargs):
        return service.compile_context(PROMPT, token_budget=4000, task_type="debug",
                                       mode="graph_escalate", **kwargs)

    def test_without_the_rule_selection_is_not_attributed_to_one(self):
        stopping = self._compile(self.off)["stopping"]
        self.assertIn(stopping["reason"],
                      ("candidates_exhausted", "max_items", "token_budget"))

    def test_with_the_rule_a_named_condition_stops_packing(self):
        stopping = self._compile(self.on)["stopping"]
        self.assertIn(stopping["reason"],
                      ("closure", "score_cliff", "no_new_information"))

    def test_the_rule_leaves_budget_unspent(self):
        # The headline property: cost is decided by sufficiency, not by the budget.
        on = self._compile(self.on)
        off = self._compile(self.off)
        self.assertGreater(on["stopping"]["tokens_left_unused"],
                           off["stopping"]["tokens_left_unused"])
        self.assertLessEqual(on["estimated_tokens"], off["estimated_tokens"])
        self.assertLessEqual(on["item_count"], off["item_count"])

    def test_the_rule_still_returns_evidence(self):
        on = self._compile(self.on)
        self.assertGreaterEqual(on["item_count"], StoppingRule().min_items)
        self.assertGreater(on["item_count"], 0)

    def test_min_items_is_a_floor_the_rule_cannot_undercut(self):
        service = ContextService(self.root, database=self.off.database,
                                 stopping=StoppingRule(min_items=7))
        try:
            compiled = self._compile(service)
        finally:
            service.close()
        self.assertGreaterEqual(compiled["item_count"], 7)

    def test_the_rule_does_not_touch_the_non_retrieval_arms(self):
        # `oracle` is the floor by construction and `full_file` is the ceiling;
        # a rule that trimmed either would change what the bounds mean.
        for mode in ("oracle", "full_file"):
            with self.subTest(mode=mode):
                kwargs = {"oracle_targets": ["checkout.py"]} if mode == "oracle" else {}
                on = self.on.compile_context(PROMPT, token_budget=4000,
                                             task_type="debug", mode=mode, **kwargs)
                off = self.off.compile_context(PROMPT, token_budget=4000,
                                               task_type="debug", mode=mode, **kwargs)
                self.assertEqual(on["item_count"], off["item_count"])
                self.assertEqual(on["stopping"]["reason"], off["stopping"]["reason"])


if __name__ == "__main__":
    unittest.main()
