"""Tests for the fusion schemes that replace the weighted sum.

The weighted sum adds bm25 relevance, cosine similarity, and graph path decay
together, which are three different units. On the haystacks that shows up as the
hybrid arms losing to the graph alone. These tests pin the two replacements at
the level where the difference is decidable -- ordering under a single confident
decoy -- rather than only through end-to-end success counts.
"""

import shutil
import tempfile
import unittest
from pathlib import Path

from context_compiler.models import Node, SearchHit
from context_compiler.retrieval import (
    FUSIONS,
    LEGACY_WEIGHTED_MODES,
    RECOMMENDED_MODE,
    RETRIEVAL_MODES,
    RRF_K,
    HybridRetriever,
)
from context_compiler.service import ContextService

FIXTURE = Path(__file__).parents[1] / "evaluation" / "fixtures" / "empty_cart_bug"
PROMPT = "submit_order accepts an empty collection; reject it."


def _hit(key, *, lexical=0.0, vector=0.0, graph=0.0):
    node = Node(key=key, path="{}.py".format(key), name=key, kind="function",
                signature="def {}()".format(key), content="def {}():\n    pass\n",
                start_line=1, end_line=2)
    return SearchHit(node=node, lexical_score=lexical, vector_score=vector,
                     graph_score=graph)


class FusionRegistryTest(unittest.TestCase):
    def test_every_mode_declares_a_known_fusion(self):
        for name, mode in RETRIEVAL_MODES.items():
            with self.subTest(mode=name):
                self.assertIn(mode.fusion, FUSIONS)

    def test_recommended_mode_is_not_a_weighted_hybrid(self):
        # The weighted hybrids stay in the registry as ablation baselines, but a
        # caller with no opinion must not be handed the scheme three results say
        # is defective.
        self.assertNotIn(RECOMMENDED_MODE, LEGACY_WEIGHTED_MODES)
        self.assertEqual(RETRIEVAL_MODES[RECOMMENDED_MODE].fusion, "escalate")
        for name in LEGACY_WEIGHTED_MODES:
            self.assertEqual(RETRIEVAL_MODES[name].fusion, "weighted")

    def test_rrf_variants_keep_the_weights_of_the_arms_they_replace(self):
        # Same sources and same weights: the only manipulated variable is whether
        # the weights apply to scores or to ranks.
        for weighted, ranked in (("lexical_graph", "lexical_graph_rrf"),
                                 ("lexical_vector_graph", "lexical_vector_graph_rrf")):
            first, second = RETRIEVAL_MODES[weighted], RETRIEVAL_MODES[ranked]
            self.assertEqual(
                (first.lexical_weight, first.vector_weight, first.graph_weight),
                (second.lexical_weight, second.vector_weight, second.graph_weight))


class WeightedSumDefectTest(unittest.TestCase):
    """The failure mode the replacements exist to remove."""

    def setUp(self):
        self.retriever = HybridRetriever.__new__(HybridRetriever)

    def _scores(self, mode_name, hits):
        mode = RETRIEVAL_MODES[mode_name]
        return self.retriever._fuse({hit.node.key: hit for hit in hits}, mode)

    def test_weighted_sum_lets_one_large_lexical_score_outrank_agreement(self):
        # A decoy with an outsized bm25 score against a hit both other sources
        # found. bm25 is unbounded, so the decoy wins on units alone.
        decoy = _hit("decoy", lexical=40.0)
        real = _hit("real", lexical=0.4, graph=0.9)
        scores = self._scores("lexical_graph", [decoy, real])
        self.assertGreater(scores["decoy"], scores["real"])

    def test_rank_fusion_puts_the_agreed_hit_first(self):
        decoy = _hit("decoy", lexical=40.0)
        real = _hit("real", lexical=0.4, graph=0.9)
        scores = self._scores("lexical_graph_rrf", [decoy, real])
        # The decoy is still rank 1 lexically, but it is unranked by the graph,
        # so it cannot outscore a hit that both sources ranked.
        self.assertGreater(scores["real"], scores["decoy"])

    def test_rank_fusion_ignores_score_magnitude_entirely(self):
        modest = self._scores("lexical_graph_rrf",
                              [_hit("a", lexical=0.9), _hit("b", lexical=0.1)])
        extreme = self._scores("lexical_graph_rrf",
                               [_hit("a", lexical=9000.0), _hit("b", lexical=0.001)])
        self.assertEqual(modest, extreme)

    def test_a_source_that_did_not_retrieve_a_hit_contributes_nothing(self):
        # Zero means "not retrieved", not "retrieved and scored zero"; giving it
        # a rank would let every source vote on every candidate.
        scores = self._scores("lexical_graph_rrf",
                              [_hit("only_lexical", lexical=1.0),
                               _hit("only_graph", graph=1.0)])
        # Each hit is rank 1 in exactly one source, so their fused scores are
        # 0.62/(k+1) and 0.38/(k+1); after normalizing by the larger, the graph
        # hit sits at exactly the weight ratio.
        self.assertAlmostEqual(scores["only_lexical"], 1.0)
        self.assertAlmostEqual(scores["only_graph"], 0.38 / 0.62, places=6)


class EscalationTest(unittest.TestCase):
    def setUp(self):
        self.retriever = HybridRetriever.__new__(HybridRetriever)
        self.mode = RETRIEVAL_MODES["graph_escalate"]

    def test_non_empty_frontier_discards_lexical_evidence(self):
        hits = {hit.node.key: hit for hit in
                [_hit("decoy", lexical=40.0), _hit("real", lexical=0.1, graph=0.7)]}
        kept = self.retriever._escalate(hits, self.mode)
        self.assertEqual(set(kept), {"real"})
        self.assertTrue(any("frontier was non-empty" in reason
                            for reason in kept["real"].reasons))

    def test_empty_frontier_falls_back_instead_of_returning_nothing(self):
        # This is the case `graph_only` cannot handle: no traversal result means
        # no context at all, so the arm fails a task lexical search would solve.
        hits = {hit.node.key: hit for hit in [_hit("only_lexical", lexical=3.0)]}
        kept = self.retriever._escalate(hits, self.mode)
        self.assertEqual(set(kept), {"only_lexical"})
        self.assertTrue(any("frontier was empty" in reason
                            for reason in kept["only_lexical"].reasons))

    def test_scores_follow_whichever_source_survived(self):
        graph_hits = {hit.node.key: hit for hit in [_hit("a", lexical=9.0, graph=0.3),
                                                    _hit("b", lexical=0.1, graph=0.8)]}
        kept = self.retriever._escalate(graph_hits, self.mode)
        scores = self.retriever._fuse(kept, self.mode)
        self.assertGreater(scores["b"], scores["a"])

        lexical_hits = {hit.node.key: hit for hit in [_hit("a", lexical=1.0),
                                                      _hit("b", lexical=5.0)]}
        kept = self.retriever._escalate(lexical_hits, self.mode)
        scores = self.retriever._fuse(kept, self.mode)
        self.assertGreater(scores["b"], scores["a"])


class EndToEndFusionTest(unittest.TestCase):
    """The new modes must compile real context, not just score synthetic hits."""

    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temporary.name) / FIXTURE.name
        shutil.copytree(FIXTURE, cls.root)
        cls.service = ContextService(cls.root)
        cls.service.index()

    @classmethod
    def tearDownClass(cls):
        cls.service.close()
        cls.temporary.cleanup()

    def test_new_modes_compile_non_empty_context_within_budget(self):
        for mode in ("lexical_graph_rrf", "lexical_vector_graph_rrf",
                     "graph_escalate", "graph_escalate_vector"):
            with self.subTest(mode=mode):
                compiled = self.service.compile_context(
                    PROMPT, token_budget=4000, task_type="debug", mode=mode)
                self.assertGreater(compiled["item_count"], 0)
                self.assertEqual(compiled["retrieval_mode"], mode)
                self.assertLessEqual(compiled["estimated_tokens"], compiled["token_budget"])

    def test_escalation_reaches_the_callee_the_lexical_arm_misses(self):
        lexical = self.service.compile_context(
            PROMPT, token_budget=700, task_type="debug", mode="lexical")
        escalated = self.service.compile_context(
            PROMPT, token_budget=700, task_type="debug", mode="graph_escalate")
        lexical_paths = {item["path"] for item in lexical["items"]}
        escalated_paths = {item["path"] for item in escalated["items"]}
        # `submit_order` calls into payments.py, which lexical search does not
        # surface from this prompt.
        self.assertIn("payments.py", escalated_paths)
        self.assertNotIn("payments.py", lexical_paths)

    def test_escalation_records_which_branch_it_took(self):
        compiled = self.service.compile_context(
            PROMPT, token_budget=4000, task_type="debug", mode="graph_escalate")
        reasons = " ".join(reason for item in compiled["items"]
                           for reason in item.get("reasons", []))
        self.assertIn("escalation:", reasons)


if __name__ == "__main__":
    unittest.main()
