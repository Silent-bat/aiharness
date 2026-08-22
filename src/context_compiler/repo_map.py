"""The tier-one repository map: names and call edges, no function bodies.

The context compiler's existing output is task-conditioned, so every new task
produces a different prompt prefix and nothing caches. This map is deliberately
*task-independent*: it depends only on the indexed commit, so it is byte-identical
across tasks and turns and can sit in a cached prefix.

It answers "where does this live and what does it call" without answering "what
does this code say", which is the expensive half. On the generated haystack the
map is roughly a tenth of the source it describes.
"""

from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from .models import Node
from .store import GraphStore

# Only edge families that carry "which symbol does this reach" belong in the map.
# Containment is implied by the outline, and reference edges are too noisy to pay
# for at map scale.
MAP_EDGE_KINDS = ("resolved_calls", "resolved_imports")


def _signature(node: Node) -> str:
    return (node.signature or node.name).strip()


def build_repo_map(store: GraphStore, *, include_calls: bool = True,
                   max_calls_per_symbol: int = 6) -> str:
    """Render the whole repository as an outline of signatures plus call edges."""
    nodes = store.all_nodes()
    by_path: Dict[str, List[Node]] = {}
    for node in nodes:
        if node.kind == "file":
            by_path.setdefault(node.path, [])
            continue
        by_path.setdefault(node.path, []).append(node)

    calls: Dict[str, List[str]] = {}
    if include_calls:
        for source, target in _map_edges(store):
            calls.setdefault(source, []).append(target)

    lines: List[str] = []
    for path in sorted(by_path):
        lines.append(path)
        for node in sorted(by_path[path], key=lambda item: item.start_line):
            indent = "    " if node.kind != "class" else "  "
            entry = "{}{}".format(indent, _signature(node))
            reached = calls.get(node.key, [])
            if reached:
                shown = sorted(set(reached))[:max_calls_per_symbol]
                entry += "  -> {}".format(", ".join(shown))
            lines.append(entry)
    return "\n".join(lines) + "\n"


def _map_edges(store: GraphStore) -> List[Tuple[str, str]]:
    placeholders = ",".join("?" for _ in MAP_EDGE_KINDS)
    rows = store.connection.execute(
        "SELECT source, target FROM edges WHERE kind IN ({}) ORDER BY source, target".format(
            placeholders),
        MAP_EDGE_KINDS,
    ).fetchall()
    resolved: List[Tuple[str, str]] = []
    for row in rows:
        target = store.get_node(row["target"])
        if target is None or target.kind == "file":
            continue
        resolved.append((row["source"], "{}:{}".format(target.path, target.name)))
    return resolved


def symbol_bodies(store: GraphStore, names: Sequence[str]) -> List[Node]:
    """Return the definitions for `names` — bodies only, not their whole files."""
    wanted = {name for name in names}
    found: List[Node] = []
    for node in store.all_nodes(["function", "class"]):
        if node.name in wanted:
            found.append(node)
    return found


def callees(store: GraphStore, name: str, *, depth: int = 2) -> List[Node]:
    """Symbols reachable from `name` through call and import edges."""
    frontier = [node.key for node in store.all_nodes(["function", "class"])
                if node.name == name]
    seen = set(frontier)
    reached: List[Node] = []
    for _ in range(max(1, depth)):
        next_frontier: List[str] = []
        for key in frontier:
            for target_key, _kind, _confidence, direction in store.neighbors(
                    key, MAP_EDGE_KINDS):
                # `neighbors` returns both endpoints; only outbound edges are
                # callees. Inbound ones are callers, which this walk must not pull
                # in or the frontier expands into the whole component.
                if direction != "out" or target_key in seen:
                    continue
                seen.add(target_key)
                node = store.get_node(target_key)
                if node is None or node.kind == "file":
                    continue
                reached.append(node)
                next_frontier.append(target_key)
        frontier = next_frontier
        if not frontier:
            break
    return reached
