from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass(frozen=True)
class Node:
    key: str
    kind: str
    name: str
    path: str
    start_line: int = 1
    end_line: int = 1
    signature: str = ""
    content: str = ""
    metadata: Dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class Edge:
    source: str
    target: str
    kind: str
    confidence: float = 1.0
    metadata: Dict[str, str] = field(default_factory=dict)


@dataclass
class SearchHit:
    node: Node
    lexical_score: float = 0.0
    graph_score: float = 0.0
    vector_score: float = 0.0
    task_score: float = 0.0
    final_score: float = 0.0
    reasons: List[str] = field(default_factory=list)
    distance: Optional[int] = None


@dataclass(frozen=True)
class ContextItem:
    node: Node
    detail: str
    text: str
    estimated_tokens: int
    score: float
    reasons: List[str]


@dataclass(frozen=True)
class CompiledContext:
    query: str
    task_type: str
    token_budget: int
    estimated_tokens: int
    items: List[ContextItem]
    rendered: str
    # Why evidence selection stopped: the sufficiency rule's reason and counters,
    # or the default "candidates_exhausted" when no rule was active.
    stopping: Dict[str, object] = field(default_factory=dict)

