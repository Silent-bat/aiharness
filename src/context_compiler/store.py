import json
import sqlite3
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from .models import Edge, Node


SCHEMA = """
PRAGMA journal_mode=WAL;
PRAGMA foreign_keys=ON;

CREATE TABLE IF NOT EXISTS files (
    path TEXT PRIMARY KEY,
    digest TEXT NOT NULL,
    language TEXT NOT NULL,
    indexed_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS nodes (
    key TEXT PRIMARY KEY,
    kind TEXT NOT NULL,
    name TEXT NOT NULL,
    path TEXT NOT NULL,
    start_line INTEGER NOT NULL,
    end_line INTEGER NOT NULL,
    signature TEXT NOT NULL,
    content TEXT NOT NULL,
    metadata TEXT NOT NULL DEFAULT '{}'
);

CREATE INDEX IF NOT EXISTS idx_nodes_path ON nodes(path);
CREATE INDEX IF NOT EXISTS idx_nodes_name ON nodes(name);
CREATE INDEX IF NOT EXISTS idx_nodes_kind ON nodes(kind);

CREATE TABLE IF NOT EXISTS edges (
    source TEXT NOT NULL,
    target TEXT NOT NULL,
    kind TEXT NOT NULL,
    confidence REAL NOT NULL DEFAULT 1.0,
    metadata TEXT NOT NULL DEFAULT '{}',
    PRIMARY KEY (source, target, kind)
);

CREATE INDEX IF NOT EXISTS idx_edges_source ON edges(source);
CREATE INDEX IF NOT EXISTS idx_edges_target ON edges(target);
CREATE INDEX IF NOT EXISTS idx_edges_kind ON edges(kind);

CREATE VIRTUAL TABLE IF NOT EXISTS node_search USING fts5(
    key UNINDEXED,
    name,
    signature,
    content,
    path,
    tokenize='unicode61 remove_diacritics 2'
);
"""


class GraphStore:
    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(str(path))
        self.connection.row_factory = sqlite3.Row
        self.connection.executescript(SCHEMA)

    def close(self) -> None:
        self.connection.close()

    def __enter__(self) -> "GraphStore":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()

    def file_digest(self, path: str) -> Optional[str]:
        row = self.connection.execute(
            "SELECT digest FROM files WHERE path = ?", (path,)
        ).fetchone()
        return row["digest"] if row else None

    def indexed_paths(self) -> List[str]:
        rows = self.connection.execute("SELECT path FROM files ORDER BY path").fetchall()
        return [row["path"] for row in rows]

    def replace_file(
        self,
        path: str,
        digest: str,
        language: str,
        nodes: Sequence[Node],
        edges: Sequence[Edge],
    ) -> None:
        # Legal Python can define one top-level name twice (platform branches,
        # `if TYPE_CHECKING`, a redefinition later in the file), which yields two
        # nodes with the same key. Keep the first and drop the rest, otherwise the
        # UNIQUE constraint aborts the entire index run for the whole repository.
        seen = set()
        deduplicated = []
        for node in nodes:
            if node.key in seen:
                continue
            seen.add(node.key)
            deduplicated.append(node)
        nodes = deduplicated
        with self.connection:
            old_keys = [
                row["key"]
                for row in self.connection.execute(
                    "SELECT key FROM nodes WHERE path = ?", (path,)
                ).fetchall()
            ]
            if old_keys:
                placeholders = ",".join("?" for _ in old_keys)
                self.connection.execute(
                    "DELETE FROM edges WHERE source IN ({0}) OR target IN ({0})".format(
                        placeholders
                    ),
                    tuple(old_keys) + tuple(old_keys),
                )
                self.connection.executemany(
                    "DELETE FROM node_search WHERE key = ?",
                    [(key,) for key in old_keys],
                )
            self.connection.execute("DELETE FROM nodes WHERE path = ?", (path,))
            self.connection.execute(
                """
                INSERT INTO files(path, digest, language, indexed_at)
                VALUES (?, ?, ?, CURRENT_TIMESTAMP)
                ON CONFLICT(path) DO UPDATE SET
                    digest=excluded.digest,
                    language=excluded.language,
                    indexed_at=CURRENT_TIMESTAMP
                """,
                (path, digest, language),
            )
            self.connection.executemany(
                """
                INSERT INTO nodes(
                    key, kind, name, path, start_line, end_line,
                    signature, content, metadata
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        node.key,
                        node.kind,
                        node.name,
                        node.path,
                        node.start_line,
                        node.end_line,
                        node.signature,
                        node.content,
                        json.dumps(node.metadata, sort_keys=True),
                    )
                    for node in nodes
                ],
            )
            self.connection.executemany(
                "INSERT INTO node_search(key, name, signature, content, path) VALUES (?, ?, ?, ?, ?)",
                [
                    (node.key, node.name, node.signature, node.content, node.path)
                    for node in nodes
                ],
            )
            self.connection.executemany(
                """
                INSERT OR REPLACE INTO edges(source, target, kind, confidence, metadata)
                VALUES (?, ?, ?, ?, ?)
                """,
                [
                    (
                        edge.source,
                        edge.target,
                        edge.kind,
                        edge.confidence,
                        json.dumps(edge.metadata, sort_keys=True),
                    )
                    for edge in edges
                ],
            )

    def remove_file(self, path: str) -> None:
        with self.connection:
            keys = [
                row["key"]
                for row in self.connection.execute(
                    "SELECT key FROM nodes WHERE path = ?", (path,)
                ).fetchall()
            ]
            if keys:
                placeholders = ",".join("?" for _ in keys)
                self.connection.execute(
                    "DELETE FROM edges WHERE source IN ({0}) OR target IN ({0})".format(
                        placeholders
                    ),
                    tuple(keys) + tuple(keys),
                )
                self.connection.executemany(
                    "DELETE FROM node_search WHERE key = ?", [(key,) for key in keys]
                )
            self.connection.execute("DELETE FROM nodes WHERE path = ?", (path,))
            self.connection.execute("DELETE FROM files WHERE path = ?", (path,))

    def rebuild_resolved_edges(self) -> int:
        """Resolve symbolic call/import targets after all files are indexed."""
        with self.connection:
            self.connection.execute("DELETE FROM edges WHERE kind LIKE 'resolved_%'")
            rows = self.connection.execute(
                "SELECT source, target, kind, confidence, metadata FROM edges WHERE kind LIKE 'symbolic_%'"
            ).fetchall()
            name_rows = self.connection.execute(
                "SELECT key, name, kind, path FROM nodes WHERE kind != 'file'"
            ).fetchall()
            by_name: Dict[str, List[sqlite3.Row]] = {}
            for row in name_rows:
                by_name.setdefault(row["name"], []).append(row)

            resolved: List[Tuple[str, str, str, float, str]] = []
            for row in rows:
                symbolic_target = row["target"]
                candidates = by_name.get(symbolic_target, [])
                if not candidates and "." in symbolic_target:
                    candidates = by_name.get(symbolic_target.rsplit(".", 1)[-1], [])
                for candidate in candidates[:8]:
                    score = 0.85 if len(candidates) == 1 else 0.55
                    resolved.append(
                        (
                            row["source"],
                            candidate["key"],
                            row["kind"].replace("symbolic_", "resolved_", 1),
                            score,
                            json.dumps({"symbol": symbolic_target}, sort_keys=True),
                        )
                    )
            self.connection.executemany(
                """
                INSERT OR REPLACE INTO edges(source, target, kind, confidence, metadata)
                VALUES (?, ?, ?, ?, ?)
                """,
                resolved,
            )
        return len(resolved)

    def get_node(self, key: str) -> Optional[Node]:
        row = self.connection.execute(
            "SELECT * FROM nodes WHERE key = ?", (key,)
        ).fetchone()
        return self._row_to_node(row) if row else None

    def get_nodes(self, keys: Iterable[str]) -> Dict[str, Node]:
        key_list = list(dict.fromkeys(keys))
        if not key_list:
            return {}
        placeholders = ",".join("?" for _ in key_list)
        rows = self.connection.execute(
            "SELECT * FROM nodes WHERE key IN ({})".format(placeholders), key_list
        ).fetchall()
        return {row["key"]: self._row_to_node(row) for row in rows}

    def all_nodes(self, kinds: Optional[Sequence[str]] = None) -> List[Node]:
        """Return every indexed node, optionally filtered by kind.

        Used by retrieval arms that must consider the whole repository
        (vector similarity, full-file context, oracle selection).
        """
        query = "SELECT * FROM nodes"
        params: List[object] = []
        if kinds:
            query += " WHERE kind IN ({})".format(",".join("?" for _ in kinds))
            params.extend(kinds)
        query += " ORDER BY path, start_line, key"
        rows = self.connection.execute(query, params).fetchall()
        return [self._row_to_node(row) for row in rows]

    def search(self, query: str, limit: int = 20) -> List[Tuple[Node, float]]:
        terms = [term for term in _tokenize_query(query) if len(term) > 1]
        if not terms:
            return []
        fts_query = " OR ".join('"{}"'.format(term.replace('"', '')) for term in terms)
        try:
            rows = self.connection.execute(
                """
                SELECT n.*, bm25(node_search, 0.0, 6.0, 3.0, 1.0, 2.0) AS rank
                FROM node_search JOIN nodes n ON n.key = node_search.key
                WHERE node_search MATCH ?
                ORDER BY rank ASC
                LIMIT ?
                """,
                (fts_query, limit),
            ).fetchall()
        except sqlite3.OperationalError:
            pattern = "%{}%".format(terms[0])
            rows = self.connection.execute(
                """
                SELECT *, 1.0 AS rank FROM nodes
                WHERE name LIKE ? OR signature LIKE ? OR content LIKE ? OR path LIKE ?
                LIMIT ?
                """,
                (pattern, pattern, pattern, pattern, limit),
            ).fetchall()
        results = []
        for row in rows:
            raw_rank = abs(float(row["rank"]))
            # bm25() returns negative values in which *more negative is better*,
            # so relevance grows with |bm25|. The earlier 1/(1+|bm25|) mapping
            # inverted that, handing the strongest lexical match the smallest
            # score. Row order was already best-first, so the inversion only
            # surfaced once a consumer re-ranked by score, which every retrieval
            # arm does.
            score = raw_rank / (1.0 + raw_rank)
            if row["name"].lower() in query.lower():
                score += 1.5
            results.append((self._row_to_node(row), score))
        return results

    def neighbors(
        self, key: str, edge_kinds: Optional[Sequence[str]] = None
    ) -> List[Tuple[str, str, float, str]]:
        params: List[object] = [key, key]
        condition = ""
        if edge_kinds:
            placeholders = ",".join("?" for _ in edge_kinds)
            condition = " AND kind IN ({})".format(placeholders)
            params.extend(edge_kinds)
        rows = self.connection.execute(
            """
            SELECT source, target, kind, confidence
            FROM edges
            WHERE (source = ? OR target = ?) {condition}
            """.format(condition=condition),
            params,
        ).fetchall()
        output = []
        for row in rows:
            other = row["target"] if row["source"] == key else row["source"]
            direction = "out" if row["source"] == key else "in"
            output.append((other, row["kind"], row["confidence"], direction))
        return output

    def stats(self) -> Dict[str, object]:
        counts = {}
        for table in ("files", "nodes", "edges"):
            counts[table] = self.connection.execute(
                "SELECT COUNT(*) AS count FROM {}".format(table)
            ).fetchone()["count"]
        counts["node_kinds"] = {
            row["kind"]: row["count"]
            for row in self.connection.execute(
                "SELECT kind, COUNT(*) AS count FROM nodes GROUP BY kind ORDER BY kind"
            ).fetchall()
        }
        counts["edge_kinds"] = {
            row["kind"]: row["count"]
            for row in self.connection.execute(
                "SELECT kind, COUNT(*) AS count FROM edges GROUP BY kind ORDER BY kind"
            ).fetchall()
        }
        return counts

    @staticmethod
    def _row_to_node(row: sqlite3.Row) -> Node:
        return Node(
            key=row["key"],
            kind=row["kind"],
            name=row["name"],
            path=row["path"],
            start_line=row["start_line"],
            end_line=row["end_line"],
            signature=row["signature"],
            content=row["content"],
            metadata=json.loads(row["metadata"] or "{}"),
        )


def _tokenize_query(query: str) -> List[str]:
    token = []
    tokens = []
    for character in query.lower():
        if character.isalnum() or character in "_./-":
            token.append(character)
        elif token:
            tokens.append("".join(token))
            token = []
    if token:
        tokens.append("".join(token))
    return tokens

