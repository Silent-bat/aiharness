"""Sparse TF-IDF vector retrieval baseline.

This is deliberately a *sparse* vector space (TF-IDF over subword-split
identifiers), not a neural embedding model. It runs offline and
deterministically, which makes it usable as a reproducible ablation arm. It is
reported as `vector (sparse tf-idf)` so results are never mistaken for a dense
embedding baseline. A dense provider can be substituted by supplying an
alternative `embed` callable to :class:`VectorIndex`.
"""

import math
from collections import Counter
from typing import Dict, Iterable, List, Sequence, Tuple

from .models import Node


def split_identifier(token: str) -> List[str]:
    """Split ``sanitize_payload``/``sanitizePayload`` into subword units."""
    parts: List[str] = []
    for chunk in token.replace("-", "_").replace(".", "_").replace("/", "_").split("_"):
        if not chunk:
            continue
        current = [chunk[0]]
        for previous, character in zip(chunk, chunk[1:]):
            if character.isupper() and not previous.isupper():
                parts.append("".join(current))
                current = [character]
            else:
                current.append(character)
        parts.append("".join(current))
    return [part.lower() for part in parts if part]


def tokenize(text: str) -> List[str]:
    raw: List[str] = []
    current: List[str] = []
    for character in text:
        if character.isalnum() or character in "_-./":
            current.append(character)
        elif current:
            raw.append("".join(current))
            current = []
    if current:
        raw.append("".join(current))

    tokens: List[str] = []
    for token in raw:
        lowered = token.lower()
        tokens.append(lowered)
        subwords = split_identifier(token)
        if len(subwords) > 1:
            tokens.extend(subwords)
    return [token for token in tokens if len(token) > 1]


def node_document(node: Node) -> str:
    return "{} {} {} {}".format(node.name, node.signature, node.path, node.content)


class VectorIndex:
    """TF-IDF index over graph nodes with cosine similarity search."""

    def __init__(self, nodes: Sequence[Node]):
        self.nodes = list(nodes)
        self.document_frequency: Counter = Counter()
        self._vectors: List[Dict[str, float]] = []

        term_frequencies = []
        for node in self.nodes:
            counts = Counter(tokenize(node_document(node)))
            term_frequencies.append(counts)
            self.document_frequency.update(counts.keys())

        total = max(1, len(self.nodes))
        for counts in term_frequencies:
            self._vectors.append(self._weight(counts, total))

    def _weight(self, counts: Counter, total: int) -> Dict[str, float]:
        vector: Dict[str, float] = {}
        for term, count in counts.items():
            frequency = self.document_frequency.get(term, 0)
            if frequency == 0:
                continue
            idf = math.log((1 + total) / (1 + frequency)) + 1.0
            vector[term] = (1.0 + math.log(count)) * idf
        norm = math.sqrt(sum(value * value for value in vector.values()))
        if norm == 0:
            return {}
        return {term: value / norm for term, value in vector.items()}

    def search(self, query: str, limit: int = 40) -> List[Tuple[Node, float]]:
        total = max(1, len(self.nodes))
        query_vector = self._weight(Counter(tokenize(query)), total)
        if not query_vector:
            return []
        scored: List[Tuple[Node, float]] = []
        for node, vector in zip(self.nodes, self._vectors):
            if not vector:
                continue
            shared = query_vector.keys() & vector.keys()
            if not shared:
                continue
            similarity = sum(query_vector[term] * vector[term] for term in shared)
            if similarity > 0:
                scored.append((node, similarity))
        scored.sort(key=lambda pair: (-pair[1], pair[0].path, pair[0].start_line))
        return scored[:limit]


def build_index(nodes: Iterable[Node]) -> VectorIndex:
    return VectorIndex(list(nodes))
