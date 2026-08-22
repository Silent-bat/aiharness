from typing import List, Optional, Sequence, Tuple

from .models import CompiledContext, ContextItem, Node, SearchHit
from .retrieval import RETRIEVAL_MODES, HybridRetriever, classify_task


DETAIL_ORDER = ("full", "snippet", "signature")

# Arms that bypass ranked retrieval entirely.
FULL_FILE_MODE = "full_file"
ORACLE_MODE = "oracle"
NON_RETRIEVAL_MODES = (FULL_FILE_MODE, ORACLE_MODE)
ALL_CONTEXT_MODES = tuple(sorted(RETRIEVAL_MODES)) + NON_RETRIEVAL_MODES


class ContextCompiler:
    def __init__(self, retriever: HybridRetriever):
        self.retriever = retriever

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

        for rank, hit in enumerate(hits):
            if len(items) >= max_items or remaining < 24:
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

        rendered = self._render(query, task, token_budget, items)
        return CompiledContext(
            query=query,
            task_type=task,
            token_budget=token_budget,
            estimated_tokens=estimate_tokens(rendered),
            items=items,
            rendered=rendered,
        )

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
