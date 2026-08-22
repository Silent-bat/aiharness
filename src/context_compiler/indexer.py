import ast
import hashlib
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

from .models import Edge, Node
from .store import GraphStore


DEFAULT_IGNORES = {
    ".git",
    ".hg",
    ".svn",
    ".venv",
    "venv",
    "node_modules",
    "dist",
    "build",
    "__pycache__",
    ".context-compiler",
}


@dataclass(frozen=True)
class IndexReport:
    scanned: int
    indexed: int
    unchanged: int
    removed: int
    errors: Tuple[str, ...]
    resolved_edges: int


class RepositoryIndexer:
    def __init__(self, root: Path, store: GraphStore):
        self.root = root.resolve()
        self.store = store

    def index(self) -> IndexReport:
        paths = list(self._source_files())
        current = {self._relative(path) for path in paths}
        indexed = 0
        unchanged = 0
        errors: List[str] = []

        for path in paths:
            relative = self._relative(path)
            try:
                raw = path.read_bytes()
                digest = hashlib.sha256(raw).hexdigest()
                if self.store.file_digest(relative) == digest:
                    unchanged += 1
                    continue
                text = raw.decode("utf-8")
                nodes, edges = PythonAnalyzer(relative, text).analyze()
                self.store.replace_file(relative, digest, "python", nodes, edges)
                indexed += 1
            except (UnicodeDecodeError, SyntaxError, OSError) as error:
                errors.append("{}: {}".format(relative, error))

        removed = 0
        for old_path in set(self.store.indexed_paths()) - current:
            self.store.remove_file(old_path)
            removed += 1

        resolved_edges = self.store.rebuild_resolved_edges()
        return IndexReport(
            scanned=len(paths),
            indexed=indexed,
            unchanged=unchanged,
            removed=removed,
            errors=tuple(errors),
            resolved_edges=resolved_edges,
        )

    def _source_files(self) -> Iterable[Path]:
        for directory, names, files in os.walk(str(self.root)):
            names[:] = sorted(name for name in names if name not in DEFAULT_IGNORES)
            for filename in sorted(files):
                if filename.endswith(".py"):
                    yield Path(directory) / filename

    def _relative(self, path: Path) -> str:
        return path.relative_to(self.root).as_posix()


class PythonAnalyzer(ast.NodeVisitor):
    def __init__(self, path: str, source: str):
        self.path = path
        self.source = source
        self.lines = source.splitlines()
        self.tree = ast.parse(source, filename=path)
        self.nodes: List[Node] = []
        self.edges: List[Edge] = []
        self.scope: List[str] = []
        self.scope_keys: List[str] = []
        self.file_key = "file:{}".format(path)

    def analyze(self) -> Tuple[List[Node], List[Edge]]:
        self.nodes.append(
            Node(
                key=self.file_key,
                kind="file",
                name=Path(self.path).name,
                path=self.path,
                start_line=1,
                end_line=max(1, len(self.lines)),
                signature=self.path,
                content=self.source,
            )
        )
        self.visit(self.tree)
        return self.nodes, self.edges

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        key = self._symbol_key("class", node.name)
        signature = "class {}{}".format(
            node.name,
            "({})".format(
                ", ".join(self._expression_name(base) for base in node.bases)
            )
            if node.bases
            else "",
        )
        self._add_symbol(node, key, "class", node.name, signature)
        for base in node.bases:
            name = self._expression_name(base)
            if name:
                self.edges.append(Edge(key, name, "symbolic_inherits", 0.9))
        self._visit_scope(node, node.name, key)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._visit_function(node, "function")

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._visit_function(node, "function")

    def visit_Import(self, node: ast.Import) -> None:
        source = self._current_scope_key()
        for alias in node.names:
            self.edges.append(
                Edge(source, alias.name, "symbolic_imports", 0.8, {"alias": alias.asname or ""})
            )

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        source = self._current_scope_key()
        module = node.module or ""
        for alias in node.names:
            target = "{}.{}".format(module, alias.name).strip(".")
            self.edges.append(
                Edge(source, target, "symbolic_imports", 0.8, {"alias": alias.asname or ""})
            )

    def visit_Call(self, node: ast.Call) -> None:
        name = self._expression_name(node.func)
        if name:
            self.edges.append(
                Edge(self._current_scope_key(), name, "symbolic_calls", 0.75)
            )
        self.generic_visit(node)

    def visit_Name(self, node: ast.Name) -> None:
        if isinstance(node.ctx, ast.Load) and self.scope_keys:
            self.edges.append(
                Edge(self._current_scope_key(), node.id, "symbolic_references", 0.35)
            )

    def _visit_function(self, node, kind: str) -> None:
        key = self._symbol_key(kind, node.name)
        prefix = "async def" if isinstance(node, ast.AsyncFunctionDef) else "def"
        signature = "{} {}({})".format(prefix, node.name, self._format_arguments(node.args))
        if node.returns:
            signature += " -> {}".format(self._safe_unparse(node.returns))
        self._add_symbol(node, key, kind, node.name, signature)
        self._visit_scope(node, node.name, key)

    def _visit_scope(self, node, name: str, key: str) -> None:
        self.scope.append(name)
        self.scope_keys.append(key)
        for child in node.body:
            self.visit(child)
        self.scope_keys.pop()
        self.scope.pop()

    def _add_symbol(self, node, key: str, kind: str, name: str, signature: str) -> None:
        start = getattr(node, "lineno", 1)
        end = getattr(node, "end_lineno", start)
        parent = self._current_scope_key()
        self.nodes.append(
            Node(
                key=key,
                kind=kind,
                name=name,
                path=self.path,
                start_line=start,
                end_line=end,
                signature=signature,
                content="\n".join(self.lines[start - 1 : end]),
                metadata={"qualified_name": ".".join(self.scope + [name])},
            )
        )
        self.edges.append(Edge(parent, key, "contains", 1.0))

    def _current_scope_key(self) -> str:
        return self.scope_keys[-1] if self.scope_keys else self.file_key

    def _symbol_key(self, kind: str, name: str) -> str:
        qualified = ".".join(self.scope + [name])
        return "{}:{}:{}".format(kind, self.path, qualified)

    def _format_arguments(self, arguments: ast.arguments) -> str:
        items = []
        positional = list(arguments.posonlyargs) + list(arguments.args)
        default_offset = len(positional) - len(arguments.defaults)
        for index, argument in enumerate(positional):
            text = argument.arg
            if argument.annotation:
                text += ": {}".format(self._safe_unparse(argument.annotation))
            if index >= default_offset:
                text += "={}".format(
                    self._safe_unparse(arguments.defaults[index - default_offset])
                )
            items.append(text)
        if arguments.vararg:
            items.append("*{}".format(arguments.vararg.arg))
        elif arguments.kwonlyargs:
            items.append("*")
        for argument, default in zip(arguments.kwonlyargs, arguments.kw_defaults):
            text = argument.arg
            if argument.annotation:
                text += ": {}".format(self._safe_unparse(argument.annotation))
            if default is not None:
                text += "={}".format(self._safe_unparse(default))
            items.append(text)
        if arguments.kwarg:
            items.append("**{}".format(arguments.kwarg.arg))
        return ", ".join(items)

    @staticmethod
    def _expression_name(node: ast.AST) -> str:
        if isinstance(node, ast.Name):
            return node.id
        if isinstance(node, ast.Attribute):
            prefix = PythonAnalyzer._expression_name(node.value)
            return "{}.{}".format(prefix, node.attr) if prefix else node.attr
        if isinstance(node, ast.Call):
            return PythonAnalyzer._expression_name(node.func)
        return ""

    @staticmethod
    def _safe_unparse(node: ast.AST) -> str:
        if hasattr(ast, "unparse"):
            try:
                return ast.unparse(node)
            except Exception:
                pass
        if isinstance(node, ast.Name):
            return node.id
        if isinstance(node, ast.Constant):
            return repr(node.value)
        return "..."

