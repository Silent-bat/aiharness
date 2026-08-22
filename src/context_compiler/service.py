from pathlib import Path
from typing import Dict, List, Optional, Sequence

from .compiler import ContextCompiler
from .indexer import RepositoryIndexer
from .retrieval import HybridRetriever, classify_task
from .store import GraphStore


DEFAULT_STATE_DIRECTORY = ".context-compiler"


class ContextService:
    def __init__(self, repository: Path, database: Optional[Path] = None):
        self.repository = repository.resolve()
        self.database = database or (
            self.repository / DEFAULT_STATE_DIRECTORY / "graph.sqlite3"
        )
        self.store = GraphStore(self.database)
        self.retriever = HybridRetriever(self.store)
        self.compiler = ContextCompiler(self.retriever)

    def close(self) -> None:
        self.store.close()

    def __enter__(self) -> "ContextService":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()

    def index(self) -> Dict[str, object]:
        report = RepositoryIndexer(self.repository, self.store).index()
        self.retriever.invalidate_caches()
        return {
            "scanned": report.scanned,
            "indexed": report.indexed,
            "unchanged": report.unchanged,
            "removed": report.removed,
            "resolved_edges": report.resolved_edges,
            "errors": list(report.errors),
        }

    def search(
        self, query: str, task_type: Optional[str] = None, limit: int = 20
    ) -> List[Dict[str, object]]:
        task = task_type or classify_task(query)
        return [
            {
                "key": hit.node.key,
                "kind": hit.node.kind,
                "name": hit.node.name,
                "path": hit.node.path,
                "start_line": hit.node.start_line,
                "end_line": hit.node.end_line,
                "signature": hit.node.signature,
                "score": round(hit.final_score, 6),
                "reasons": hit.reasons,
            }
            for hit in self.retriever.retrieve(
                query, task_type=task, result_limit=limit
            )
        ]

    def compile_context(
        self,
        query: str,
        token_budget: int = 4000,
        task_type: Optional[str] = None,
        graph_depth: int = 2,
        max_items: int = 20,
        mode: Optional[str] = None,
        oracle_targets: Optional[Sequence[str]] = None,
    ) -> Dict[str, object]:
        result = self.compiler.compile(
            query,
            token_budget=token_budget,
            task_type=task_type,
            graph_depth=graph_depth,
            max_items=max_items,
            mode=mode,
            oracle_targets=oracle_targets,
        )
        return {
            "task_type": result.task_type,
            "retrieval_mode": mode or ("lexical" if graph_depth <= 0 else "lexical_graph"),
            "token_budget": result.token_budget,
            "estimated_tokens": result.estimated_tokens,
            "item_count": len(result.items),
            "items": [
                {
                    "key": item.node.key,
                    "path": item.node.path,
                    "start_line": item.node.start_line,
                    "end_line": item.node.end_line,
                    "detail": item.detail,
                    "estimated_tokens": item.estimated_tokens,
                    "score": round(item.score, 6),
                    "reasons": item.reasons,
                }
                for item in result.items
            ],
            "context": result.rendered,
        }

    def impact(self, symbol: str, depth: int = 2, limit: int = 40) -> Dict[str, object]:
        seeds = self.store.search(symbol, limit=5)
        if not seeds:
            return {"symbol": symbol, "matches": [], "impacted": []}
        seed = seeds[0][0]
        hits = self.retriever.retrieve(
            "refactor impact references callers {}".format(symbol),
            task_type="refactor",
            seed_limit=8,
            graph_depth=depth,
            result_limit=limit,
        )
        return {
            "symbol": symbol,
            "matches": [
                {
                    "key": node.key,
                    "path": node.path,
                    "start_line": node.start_line,
                    "signature": node.signature,
                }
                for node, _ in seeds
            ],
            "seed": seed.key,
            "impacted": [
                {
                    "key": hit.node.key,
                    "path": hit.node.path,
                    "start_line": hit.node.start_line,
                    "signature": hit.node.signature,
                    "score": round(hit.final_score, 6),
                    "reasons": hit.reasons,
                }
                for hit in hits
                if hit.node.key != seed.key
            ],
        }

    def stats(self) -> Dict[str, object]:
        return self.store.stats()

