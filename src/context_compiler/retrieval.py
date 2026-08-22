import math
from collections import deque
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

from .models import SearchHit
from .store import GraphStore
from .vectors import VectorIndex


@dataclass(frozen=True)
class RetrievalMode:
    """One ablation arm: which evidence sources contribute to ranking."""

    name: str
    lexical_weight: float
    vector_weight: float
    graph_weight: float
    graph_depth: int
    label: str
    graph_seeds_only: bool = False

    @property
    def uses_lexical(self) -> bool:
        return self.lexical_weight > 0

    @property
    def uses_vector(self) -> bool:
        return self.vector_weight > 0

    @property
    def uses_graph(self) -> bool:
        return self.graph_weight > 0 and self.graph_depth > 0


RETRIEVAL_MODES: Dict[str, RetrievalMode] = {
    # Lexical BM25 only. This is the "classic CLI" baseline.
    "lexical": RetrievalMode("lexical", 1.0, 0.0, 0.0, 0, "lexical bm25 only"),
    # Sparse TF-IDF vector similarity only.
    "vector": RetrievalMode("vector", 0.0, 1.0, 0.0, 0, "vector (sparse tf-idf) only"),
    # Graph expansion only: lexical hits seed traversal but do not score.
    "graph": RetrievalMode(
        "graph", 0.0, 0.0, 1.0, 2, "graph traversal only (lexical seeds discarded)",
        graph_seeds_only=True,
    ),
    "lexical_graph": RetrievalMode(
        "lexical_graph", 0.62, 0.0, 0.38, 2, "lexical + graph traversal"
    ),
    "lexical_vector": RetrievalMode(
        "lexical_vector", 0.6, 0.4, 0.0, 0, "lexical + vector"
    ),
    "lexical_vector_graph": RetrievalMode(
        "lexical_vector_graph", 0.45, 0.25, 0.30, 2, "lexical + vector + graph traversal"
    ),
}


TASK_KEYWORDS = {
    "debug": {
        "bug",
        "debug",
        "error",
        "exception",
        "fail",
        "failing",
        "fix",
        "incorrect",
        "regression",
        "stack",
        "trace",
    },
    "feature": {
        "add",
        "build",
        "create",
        "feature",
        "implement",
        "introduce",
        "support",
    },
    "refactor": {
        "cleanup",
        "extract",
        "move",
        "rename",
        "refactor",
        "reorganize",
        "replace",
    },
    "security": {
        "auth",
        "permission",
        "sanitize",
        "secret",
        "security",
        "threat",
        "validate",
        "vulnerability",
    },
    "performance": {
        "benchmark",
        "cache",
        "latency",
        "memory",
        "optimize",
        "performance",
        "slow",
        "throughput",
    },
    "test": {"coverage", "spec", "test", "tests", "verify"},
}


TASK_EDGE_WEIGHTS = {
    "debug": {
        "resolved_calls": 1.0,
        "resolved_references": 0.8,
        "contains": 0.45,
        "resolved_imports": 0.5,
    },
    "feature": {
        "resolved_calls": 0.75,
        "resolved_imports": 0.65,
        "resolved_inherits": 0.9,
        "contains": 0.55,
        "resolved_references": 0.55,
    },
    "refactor": {
        "resolved_references": 1.0,
        "resolved_calls": 0.9,
        "resolved_imports": 0.85,
        "resolved_inherits": 0.9,
        "contains": 0.5,
    },
    "security": {
        "resolved_calls": 1.0,
        "resolved_references": 0.9,
        "resolved_imports": 0.5,
        "contains": 0.4,
    },
    "performance": {
        "resolved_calls": 1.0,
        "resolved_references": 0.6,
        "contains": 0.4,
    },
    "test": {
        "resolved_calls": 0.85,
        "resolved_references": 0.9,
        "resolved_imports": 0.75,
        "contains": 0.55,
    },
    "general": {
        "resolved_calls": 0.8,
        "resolved_references": 0.65,
        "resolved_imports": 0.6,
        "resolved_inherits": 0.75,
        "contains": 0.45,
    },
}


def classify_task(query: str) -> str:
    words = set(_words(query))
    scored = [
        (len(words & keywords), task) for task, keywords in TASK_KEYWORDS.items()
    ]
    score, task = max(scored, default=(0, "general"))
    return task if score else "general"


class HybridRetriever:
    def __init__(self, store: GraphStore):
        self.store = store
        self._vector_index: Optional[VectorIndex] = None

    def vector_index(self) -> VectorIndex:
        if self._vector_index is None:
            self._vector_index = VectorIndex(self.store.all_nodes())
        return self._vector_index

    def invalidate_caches(self) -> None:
        self._vector_index = None

    def resolve_mode(self, mode: Optional[str], graph_depth: int) -> RetrievalMode:
        """Pick an ablation arm, falling back to legacy graph_depth semantics."""
        if mode is not None:
            if mode not in RETRIEVAL_MODES:
                raise ValueError(
                    "unknown retrieval mode {!r}; expected one of {}".format(
                        mode, ", ".join(sorted(RETRIEVAL_MODES))
                    )
                )
            base = RETRIEVAL_MODES[mode]
            if base.uses_graph and graph_depth and graph_depth != base.graph_depth:
                return RetrievalMode(
                    base.name, base.lexical_weight, base.vector_weight,
                    base.graph_weight, graph_depth, base.label, base.graph_seeds_only,
                )
            return base
        if graph_depth <= 0:
            return RETRIEVAL_MODES["lexical"]
        legacy = RETRIEVAL_MODES["lexical_graph"]
        return RetrievalMode(
            legacy.name, legacy.lexical_weight, legacy.vector_weight,
            legacy.graph_weight, graph_depth, legacy.label, legacy.graph_seeds_only,
        )

    def retrieve(
        self,
        query: str,
        task_type: Optional[str] = None,
        seed_limit: int = 12,
        graph_depth: int = 2,
        result_limit: int = 40,
        mode: Optional[str] = None,
    ) -> List[SearchHit]:
        task = task_type or classify_task(query)
        active = self.resolve_mode(mode, graph_depth)
        hits: Dict[str, SearchHit] = {}

        # Lexical hits are always computed: even a graph-only or vector-only arm
        # needs seeds, but only scoring weights decide whether they survive.
        lexical = self.store.search(query, limit=max(seed_limit, result_limit))
        for node, score in lexical:
            # No positional discount: `store.search` now returns a score that is
            # monotone in bm25 relevance, so discounting by row position would
            # re-impose raw bm25 order over the exact-symbol-name bonus.
            hit = SearchHit(node=node, lexical_score=score)
            hit.reasons.append("lexical match for task query")
            hits[node.key] = hit

        if active.uses_vector:
            for index, (node, similarity) in enumerate(
                self.vector_index().search(query, limit=max(seed_limit, result_limit))
            ):
                hit = hits.setdefault(node.key, SearchHit(node=node))
                hit.vector_score = max(hit.vector_score, similarity)
                reason = "vector (sparse tf-idf) similarity {:.3f}".format(similarity)
                if reason not in hit.reasons:
                    hit.reasons.append(reason)

        if active.uses_graph:
            # Seed from the strongest lexical *scores*, not from raw bm25 row
            # order. On a repository large enough to hold near-miss distractors,
            # bm25 row order is dominated by files that repeat the prompt's
            # vocabulary, so row-order seeding never puts the real fix site in the
            # frontier and traversal cannot reach its callees.
            seed_keys = [
                node.key for node, _ in sorted(
                    lexical, key=lambda pair: (-pair[1], pair[0].path, pair[0].start_line)
                )[:seed_limit]
            ]
            if active.uses_vector:
                seed_keys.extend(
                    node.key
                    for node, _ in self.vector_index().search(query, limit=seed_limit)
                )
            graph_scores = self._expand_graph(
                list(dict.fromkeys(seed_keys)), task, active.graph_depth
            )
            graph_nodes = self.store.get_nodes(graph_scores.keys())
            for key, (score, distance, reasons) in graph_scores.items():
                node = graph_nodes.get(key)
                if not node:
                    continue
                hit = hits.setdefault(key, SearchHit(node=node))
                hit.graph_score = max(hit.graph_score, score)
                hit.distance = distance if hit.distance is None else min(hit.distance, distance)
                hit.reasons.extend(reason for reason in reasons if reason not in hit.reasons)

        if active.graph_seeds_only:
            # Graph-only arm: keep nodes reached by traversal, drop pure lexical seeds.
            hits = {key: hit for key, hit in hits.items() if hit.graph_score > 0}

        query_words = set(_words(query))
        weight_total = (
            active.lexical_weight + active.vector_weight + active.graph_weight
        ) or 1.0
        for hit in hits.values():
            hit.task_score = self._task_affinity(hit, task, query_words)
            evidence = (
                active.lexical_weight * hit.lexical_score
                + active.vector_weight * hit.vector_score
                + active.graph_weight * hit.graph_score
            ) / weight_total
            hit.final_score = 0.9 * evidence + 0.10 * hit.task_score
            if hit.node.kind == "file":
                hit.final_score *= 0.72
            if hit.node.path.startswith("tests/") or "/tests/" in hit.node.path:
                if task in {"debug", "test", "refactor"}:
                    hit.final_score *= 1.18
                    hit.reasons.append("test evidence is relevant to {} tasks".format(task))

        return sorted(
            hits.values(),
            key=lambda hit: (-hit.final_score, hit.node.path, hit.node.start_line),
        )[:result_limit]

    def _expand_graph(
        self, seeds: Sequence[str], task: str, max_depth: int
    ) -> Dict[str, Tuple[float, int, List[str]]]:
        weights = TASK_EDGE_WEIGHTS.get(task, TASK_EDGE_WEIGHTS["general"])
        best: Dict[str, Tuple[float, int, List[str]]] = {}
        queue = deque((seed, 0, 1.0) for seed in seeds)
        visited: Dict[str, float] = {seed: 1.0 for seed in seeds}

        while queue:
            key, depth, path_score = queue.popleft()
            if depth >= max_depth:
                continue
            for other, edge_kind, confidence, direction in self.store.neighbors(
                key, list(weights.keys())
            ):
                edge_weight = weights.get(edge_kind, 0.2)
                direction_weight = _direction_weight(task, edge_kind, direction)
                next_score = path_score * edge_weight * confidence * direction_weight
                next_score *= 0.72 ** depth
                if next_score < 0.08:
                    continue
                reason = "graph {} {} at distance {}".format(
                    edge_kind.replace("resolved_", ""), direction, depth + 1
                )
                previous = best.get(other)
                if previous is None or next_score > previous[0]:
                    best[other] = (next_score, depth + 1, [reason])
                elif reason not in previous[2]:
                    previous[2].append(reason)
                if next_score > visited.get(other, 0.0):
                    visited[other] = next_score
                    queue.append((other, depth + 1, next_score))
        return best

    @staticmethod
    def _task_affinity(hit: SearchHit, task: str, query_words: Set[str]) -> float:
        node_words = set(_words("{} {} {}".format(
            hit.node.name, hit.node.signature, hit.node.path
        )))
        overlap = len(node_words & query_words)
        score = min(1.0, overlap / max(1.0, math.sqrt(len(query_words))))
        lowered = hit.node.path.lower()
        if task == "test" and ("test" in lowered or "spec" in lowered):
            score += 0.7
        if task == "security" and any(
            term in node_words for term in ("auth", "permission", "validate", "sanitize")
        ):
            score += 0.5
        if task == "performance" and any(
            term in node_words for term in ("cache", "query", "batch", "async")
        ):
            score += 0.4
        return min(1.5, score)


def _direction_weight(task: str, edge_kind: str, direction: str) -> float:
    if task in {"debug", "refactor"} and edge_kind in {
        "resolved_calls",
        "resolved_references",
    }:
        return 1.12 if direction == "in" else 1.0
    if task == "feature" and edge_kind == "resolved_inherits":
        return 1.1
    return 1.0


def _words(text: str) -> Iterable[str]:
    current = []
    for character in text.lower():
        if character.isalnum() or character == "_":
            current.append(character)
        elif current:
            yield "".join(current)
            current = []
    if current:
        yield "".join(current)

