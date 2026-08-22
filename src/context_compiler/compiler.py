from dataclasses import dataclass
from typing import Dict, List, Mapping, Optional, Sequence, Set, Tuple

from .models import CompiledContext, ContextItem, Node, SearchHit
from .retrieval import RETRIEVAL_MODES, HybridRetriever, classify_task


DETAIL_ORDER = ("full", "snippet", "signature")

# Arms that bypass ranked retrieval entirely.
FULL_FILE_MODE = "full_file"
ORACLE_MODE = "oracle"
NON_RETRIEVAL_MODES = (FULL_FILE_MODE, ORACLE_MODE)
ALL_CONTEXT_MODES = tuple(sorted(RETRIEVAL_MODES)) + NON_RETRIEVAL_MODES

# Edge families that make one piece of evidence depend on another. A selected
# function that calls an unselected one leaves the model guessing at the callee's
# behaviour, so these are the references the stopping rule tries to close over.
CLOSURE_EDGES = ("resolved_calls", "resolved_imports", "resolved_inherits")


@dataclass(frozen=True)
class StoppingRule:
    """When to stop adding evidence, independent of how much budget is left.

    Every arm except `oracle` and `graph_only` fills whatever budget it is given,
    which is why raising the budget raises cost without raising success: the
    ranked tail is padding. `graph_only` is the exception, and not by design --
    graph traversal has a bounded frontier, so it self-limits to ~191 items even
    when allowed 4,000. This rule makes that property explicit and available to
    every arm.

    Three independent conditions, any of which stops packing:

    - `closure`: every reference made by the selected evidence points at
      something also selected (or at something outside the index). Nothing
      further is needed to read what is already there.
    - `cliff`: the next candidate scores below `cliff_ratio` of the best one.
      Below that the ranking is not distinguishing evidence from vocabulary
      overlap.
    - `patience`: `patience` candidates in a row added no new file and closed no
      outstanding reference. The tail has gone flat.

    `min_items` guards against stopping on the first hit, which is trivially
    closed whenever it happens to call nothing.
    """

    cliff_ratio: float = 0.15
    patience: int = 4
    min_items: int = 3
    require_closure: bool = True
    enabled: bool = True


@dataclass
class StoppingDiagnostics:
    """Why packing stopped, so a run can be audited rather than trusted."""

    reason: str = "candidates_exhausted"
    items_selected: int = 0
    candidates_considered: int = 0
    open_references: int = 0
    tokens_left_unused: int = 0
    best_score: float = 0.0
    last_score: float = 0.0

    def as_dict(self) -> Dict[str, object]:
        return {
            "reason": self.reason,
            "items_selected": self.items_selected,
            "candidates_considered": self.candidates_considered,
            "open_references": self.open_references,
            # The headline for this rule: budget deliberately left on the table.
            "tokens_left_unused": self.tokens_left_unused,
            "best_score": round(self.best_score, 6),
            "last_score": round(self.last_score, 6),
        }


class ContextCompiler:
    def __init__(self, retriever: HybridRetriever,
                 stopping: Optional[StoppingRule] = None):
        self.retriever = retriever
        # Off by default: turning it on changes what every existing arm packs,
        # so it is a manipulated variable, not a silent improvement.
        self.stopping = stopping or StoppingRule(enabled=False)

    def compile(
        self,
        query: str,
        token_budget: int = 4000,
        task_type: Optional[str] = None,
        graph_depth: int = 2,
        max_items: int = 20,
        mode: Optional[str] = None,
        oracle_targets: Optional[Sequence[str]] = None,
    ) -> CompiledContext:
        if token_budget < 128:
            raise ValueError("token budget must be at least 128")
        task = task_type or classify_task(query)
        if mode in NON_RETRIEVAL_MODES:
            hits = self._unranked_hits(mode, oracle_targets)
        else:
            hits = self.retriever.retrieve(
                query,
                task_type=task,
                graph_depth=graph_depth,
                result_limit=max(max_items * 3, 30),
                mode=mode,
            )
        overhead = estimate_tokens(self._header(query, task, token_budget))
        remaining = max(0, token_budget - overhead)
        items: List[ContextItem] = []
        seen_regions = set()
        # Sufficiency bookkeeping. `open_references` holds keys the selected
        # evidence depends on but does not contain; `flat` counts consecutive
        # candidates that neither closed a reference nor introduced a new file.
        selected_keys: Set[str] = set()
        selected_paths: Set[str] = set()
        open_references: Set[str] = set()
        flat = 0
        rule = self.stopping
        active = rule.enabled and mode not in NON_RETRIEVAL_MODES
        best_score = hits[0].final_score if hits else 0.0
        diagnostics = StoppingDiagnostics(best_score=best_score)

        for rank, hit in enumerate(hits):
            diagnostics.candidates_considered = rank + 1
            if len(items) >= max_items:
                diagnostics.reason = "max_items"
                break
            if remaining < 24:
                diagnostics.reason = "token_budget"
                break
            if active and len(items) >= rule.min_items:
                stop = self._sufficiency_stop(rule, hit, best_score,
                                              open_references, flat)
                if stop is not None:
                    diagnostics.reason = stop
                    break
            region = (hit.node.path, hit.node.start_line, hit.node.end_line)
            if region in seen_regions:
                continue
            # A file-level item repeats source already represented by selected
            # symbols from that file. Prefer the narrower symbol evidence.
            if hit.node.kind == "file" and any(
                item.node.path == hit.node.path and item.node.kind != "file"
                for item in items
            ):
                continue
            item = self._fit_item(hit, remaining, rank, len(items) + 1, mode=mode)
            if item is None:
                continue
            items.append(item)
            seen_regions.add(region)
            remaining -= item.estimated_tokens
            diagnostics.last_score = hit.final_score
            if active:
                closed, new_path = self._absorb(hit.node, selected_keys,
                                                selected_paths, open_references)
                flat = 0 if (closed or new_path) else flat + 1

        diagnostics.items_selected = len(items)
        diagnostics.open_references = len(open_references)
        diagnostics.tokens_left_unused = remaining
        rendered = self._render(query, task, token_budget, items)
        return CompiledContext(
            query=query,
            task_type=task,
            token_budget=token_budget,
            estimated_tokens=estimate_tokens(rendered),
            items=items,
            rendered=rendered,
            stopping=diagnostics.as_dict(),
        )

    @staticmethod
    def _sufficiency_stop(rule: StoppingRule, hit: SearchHit, best_score: float,
                          open_references: Set[str], flat: int) -> Optional[str]:
        """Return the name of the condition that says stop, or None to continue."""
        if rule.require_closure and not open_references:
            return "closure"
        if best_score > 0 and hit.final_score < rule.cliff_ratio * best_score:
            return "score_cliff"
        if flat >= rule.patience:
            return "no_new_information"
        return None

    def _absorb(self, node: Node, selected_keys: Set[str], selected_paths: Set[str],
                open_references: Set[str]) -> Tuple[bool, bool]:
        """Record a selection and update the outstanding-reference set.

        Returns whether this node closed an outstanding reference and whether it
        introduced a file not already represented. Both are "did this add
        information" signals; a candidate that does neither is tail padding.
        """
        closed = node.key in open_references
        new_path = node.path not in selected_paths
        selected_keys.add(node.key)
        selected_paths.add(node.path)
        open_references.discard(node.key)
        for other, _kind, _confidence, direction in self.retriever.store.neighbors(
            node.key, list(CLOSURE_EDGES)
        ):
            # Only outbound dependencies: what this evidence needs in order to be
            # readable. Inbound callers are useful for impact analysis and are not
            # required to understand the code in hand.
            if direction == "out" and other not in selected_keys:
                open_references.add(other)
        return closed, new_path

    def _unranked_hits(
        self, mode: str, oracle_targets: Optional[Sequence[str]]
    ) -> List[SearchHit]:
        """Build hits for the full-file and oracle ablation arms.

        `full_file` is the naive upper-context baseline: every source file in
        path order, no ranking. `oracle` is the human-labeled ceiling: only the
        files and symbols the task declares as relevant.
        """
        store = self.retriever.store
        if mode == FULL_FILE_MODE:
            nodes = [node for node in store.all_nodes(("file",))]
            return [
                SearchHit(node=node, final_score=1.0,
                          reasons=["full-file baseline: entire repository file"])
                for node in nodes
            ]

        targets = list(oracle_targets or [])
        if not targets:
            raise ValueError("oracle mode requires oracle_targets")
        selected: List[SearchHit] = []
        seen = set()
        nodes = store.all_nodes()
        for target in targets:
            for node in nodes:
                matches_path = node.kind == "file" and node.path == target
                matches_symbol = node.kind != "file" and (
                    node.name == target or node.key == target
                    or "{}::{}".format(node.path, node.name) == target
                )
                if not (matches_path or matches_symbol) or node.key in seen:
                    continue
                seen.add(node.key)
                selected.append(SearchHit(
                    node=node, final_score=1.0,
                    reasons=["oracle: human-labeled relevant evidence ({})".format(target)],
                ))
        missing = [
            target for target in targets
            if not any(target in reason for hit in selected for reason in hit.reasons)
        ]
        if missing:
            raise ValueError("oracle targets not found in index: {}".format(", ".join(missing)))
        return selected

    def _fit_item(
        self, hit: SearchHit, remaining: int, rank: int, item_number: int,
        mode: Optional[str] = None,
    ) -> Optional[ContextItem]:
        preferred = self._preferred_details(hit, rank, mode)
        for detail in preferred:
            text = render_node(hit.node, detail)
            candidate = ContextItem(
                node=hit.node,
                detail=detail,
                text=text,
                estimated_tokens=0,
                score=hit.final_score,
                reasons=hit.reasons[:4],
            )
            cost = estimate_tokens(render_context_item(item_number, candidate))
            if cost <= remaining:
                return ContextItem(
                    node=candidate.node,
                    detail=candidate.detail,
                    text=candidate.text,
                    estimated_tokens=cost,
                    score=candidate.score,
                    reasons=candidate.reasons,
                )
        return None

    @staticmethod
    def _preferred_details(hit: SearchHit, rank: int, mode: Optional[str] = None) -> Sequence[str]:
        if mode in NON_RETRIEVAL_MODES:
            # These arms are defined by giving the model complete source.
            return ("full", "snippet", "signature")
        if hit.node.kind == "file":
            return ("signature",)
        if rank < 3 or hit.final_score >= 0.9:
            return DETAIL_ORDER
        if rank < 9:
            return ("snippet", "signature")
        return ("signature",)

    @staticmethod
    def _header(query: str, task: str, token_budget: int) -> str:
        return (
            "# Compiled repository context\n"
            "Task type: {}\n"
            "Token budget: {}\n"
            "Query: {}\n"
        ).format(task, token_budget, query)

    def _render(
        self,
        query: str,
        task: str,
        token_budget: int,
        items: Sequence[ContextItem],
    ) -> str:
        sections = [self._header(query, task, token_budget)]
        for index, item in enumerate(items, 1):
            sections.append(render_context_item(index, item))
        if not items:
            sections.append("No matching repository evidence was found.\n")
        return "\n".join(sections)


def render_node(node: Node, detail: str) -> str:
    if detail == "signature":
        return node.signature or "{} {}".format(node.kind, node.name)
    if detail == "snippet":
        lines = node.content.splitlines()
        if len(lines) <= 36:
            return node.content
        return "\n".join(lines[:24] + ["# ... context truncated ..."] + lines[-8:])
    return node.content


def estimate_tokens(text: str) -> int:
    if not text:
        return 0
    # Conservative model-independent approximation for source code and prose.
    return max(1, (len(text) + 2) // 3)


def render_context_item(index: int, item: ContextItem) -> str:
    reason = "; ".join(item.reasons) or "ranked repository evidence"
    return (
        "## Evidence {}: {}:{}-{}\n"
        "Kind: {} | Detail: {} | Score: {:.3f}\n"
        "Why included: {}\n\n"
        "```{}\n{}\n```\n"
    ).format(
        index,
        item.node.path,
        item.node.start_line,
        item.node.end_line,
        item.node.kind,
        item.detail,
        item.score,
        reason,
        _language_for_path(item.node.path),
        item.text.rstrip(),
    )


def _language_for_path(path: str) -> str:
    if path.endswith(".py"):
        return "python"
    if path.endswith((".ts", ".tsx")):
        return "typescript"
    if path.endswith((".js", ".jsx")):
        return "javascript"
    return "text"
