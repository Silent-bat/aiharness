"""Two scripted coding agents on one tool surface, measured in turns and tokens.

The retrieval benchmark asks "given one shot at a context window, which evidence
selector wins". That is not how a terminal agent actually works: it takes turns,
and every turn re-sends the whole transcript, so total input grows as
`T * P + g * T^2 / 2` for `T` turns of average growth `g` on a prefix `P`. Under
that cost model an arm can win the one-shot recall test and still lose the session.

So this module measures the *tool economics* of two policies against the same
tools, the same repository and the same success rule:

  grep_loop   grep the prompt's identifiers, read the best-matching file whole,
              follow its imports and read those whole. This is the strong version
              of what a lexical CLI agent does — not a one-shot bm25 dump.
  map_expand  load the task-independent repository map once, then pull individual
              symbol bodies along call edges.

Both are deterministic scripted policies. **No model is involved**, so nothing
here is evidence about model competence; it is evidence about how many turns and
how many tokens each tool surface costs to put the required evidence in front of
a model. The success rule is the arm-blind one the mock client already uses: the
transcript must contain every string in the case's `mock_requires`.
"""

import argparse
import ast
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from context_compiler.compiler import estimate_tokens  # noqa: E402
from context_compiler.repo_map import build_repo_map, callees  # noqa: E402
from context_compiler.service import ContextService  # noqa: E402

MAX_TURNS = 20
GREP_MATCH_LIMIT = 40
STOP_WORDS = {"the", "a", "an", "is", "it", "and", "or", "to", "of", "in", "on",
              "for", "with", "that", "this", "accepts", "reject", "empty",
              "should", "must", "when", "then", "not", "no"}

# Anthropic-style cache economics: a cache read is billed at a tenth of a fresh
# input token. Only a byte-identical, append-only prefix can be read from cache.
CACHE_READ_RATE = 0.1


class Repository:
    """The tool surface. Both arms get exactly these calls."""

    def __init__(self, root: Path):
        self.root = root
        self.files = sorted(
            path for path in root.rglob("*.py")
            if ".context-compiler" not in path.parts and "__pycache__" not in path.parts)

    def _relative(self, path: Path) -> str:
        return str(path.relative_to(self.root))

    def grep(self, pattern: str) -> str:
        matcher = re.compile(pattern)
        lines: List[str] = []
        for path in self.files:
            try:
                text = path.read_text(encoding="utf-8")
            except OSError:
                continue
            for number, line in enumerate(text.splitlines(), start=1):
                if matcher.search(line):
                    lines.append("{}:{}: {}".format(self._relative(path), number,
                                                    line.strip()))
                    if len(lines) >= GREP_MATCH_LIMIT:
                        return "\n".join(lines) + "\n[truncated]"
        return "\n".join(lines) if lines else "(no matches)"

    def read_file(self, relative: str) -> str:
        path = self.root / relative
        if not path.is_file():
            return "(no such file: {})".format(relative)
        return path.read_text(encoding="utf-8")

    def imports_of(self, relative: str) -> List[str]:
        """Local modules imported by `relative`, as repository-relative paths."""
        text = self.read_file(relative)
        try:
            tree = ast.parse(text)
        except SyntaxError:
            return []
        modules: List[str] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                modules.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                modules.append(node.module)
        local: List[str] = []
        for module in modules:
            candidate = "{}.py".format(module.replace(".", "/"))
            if (self.root / candidate).is_file():
                local.append(candidate)
        return local


def _required_evidence(case: Dict[str, Any], service: ContextService) -> List[Tuple[str, str]]:
    """The bodies an agent must actually have seen, as (label, exact text) pairs.

    The mock client's `mock_requires` is a substring rule over definition headers,
    and the repository map emits headers by construction — so scoring the map arm
    against `mock_requires` hands it a 6/6 for free without it ever paying for a
    line of logic. The rule here is body-level: for every labelled relevant symbol
    and every labelled test symbol, the transcript must contain that symbol's
    source. Both arms read from the same files, so a body pulled by `read_symbol`
    is a byte-exact substring of the same file read by `read_file`, and neither
    arm is advantaged by how it obtained the text.
    """
    labels = case.get("labels", {})
    wanted = list(labels.get("relevant_symbols", [])) + list(labels.get("test_symbols", []))
    # Scope to the labelled files: a generated decoy that happens to reuse one of
    # these names would otherwise add its body to the requirement, which no arm
    # should have to find.
    scope = set(labels.get("relevant_files", [])) | set(labels.get("tests", []))
    evidence: List[Tuple[str, str]] = []
    for node in service.store.all_nodes(["function", "class"]):
        if node.name in wanted and node.path in scope and node.content.strip():
            evidence.append((node.name, node.content.strip()))
    return evidence


class Transcript:
    """Turn-by-turn token accounting under the re-send-everything cost model.

    A terminal agent does not pay for its context once. Turn `t` re-sends every
    byte of turns `1..t-1`, so the honest unit of cost is the sum of the prefix
    over turns, not the size of the final window. `cached` prices the same
    transcript when the unchanged prefix is served from cache, which is only
    available if growth is append-only and the prefix is byte-identical.

    Caching is a property of the *whole* prefix, not of the static header. Turn
    `t`'s input begins with turn `t-1`'s input verbatim, so the accumulated
    transcript is as cacheable as the system block is. An earlier version of this
    model discounted only the static header, which handed a 9.5x discount to an arm
    whose cost is a fixed map and almost none to an arm whose cost is a growing
    transcript -- an artefact of where an arm's tokens sat rather than of caching.
    `warm_prefix` covers the one thing append-only accounting cannot see: a
    byte-identical static block (a repository map) is already in cache on turn 1 of
    the second and later cases in a run, so it is a read rather than a write there.
    """

    def __init__(self, system: str, *, cacheable_prefix: str = "",
                 warm_prefix: bool = False):
        self.cacheable_prefix = cacheable_prefix
        self.system = system
        self.entries: List[Tuple[str, str]] = []
        self.turns = 0
        self.no_cache_input = 0
        self.cached_input = 0.0
        self.new_tokens = 0
        self._base = estimate_tokens(system) + estimate_tokens(cacheable_prefix)
        self._live = 0
        # What the provider has already seen, and would serve at the read rate.
        # Only the static block qualifies, and only because it is emitted before the
        # task-specific system text: a cache hit needs a byte-identical *prefix*, so
        # a per-task preamble ahead of the map would invalidate it every case.
        self._warm = estimate_tokens(cacheable_prefix) if warm_prefix else 0

    def act(self, call: str, observation: str) -> None:
        """Record one turn: the model is prompted, then a tool result is appended."""
        self.turns += 1
        # The model sees everything accumulated *before* this turn's result.
        prefix = self._base + self._live
        self.no_cache_input += prefix
        # Everything sent on an earlier turn is a byte-identical prefix of this
        # turn's input; only what this turn adds is billed fresh.
        warm = min(self._warm, prefix)
        self.cached_input += warm * CACHE_READ_RATE + (prefix - warm)
        self._warm = prefix
        added = estimate_tokens(call) + estimate_tokens(observation)
        self.new_tokens += added
        self._live += added
        self.entries.append((call, observation))

    def text(self) -> str:
        # Static block first: see `_warm`. Cache hits require a byte-identical
        # prefix, so anything task-specific has to come after it.
        blocks = [self.cacheable_prefix, self.system] if self.cacheable_prefix else [self.system]
        for call, observation in self.entries:
            blocks.append("$ {}\n{}".format(call, observation))
        return "\n\n".join(blocks)

    def final_tokens(self) -> int:
        return self._base + self._live

    def satisfies(self, required: Sequence[Tuple[str, str]]) -> bool:
        body = self.text()
        return all(text in body for _label, text in required)

    def missing(self, required: Sequence[Tuple[str, str]]) -> List[str]:
        body = self.text()
        return [label for label, text in required if text not in body]


def _identifiers(prompt: str) -> List[str]:
    """Prompt tokens an agent would plausibly grep for, longest first.

    Longest first matters: `submit_order` locates the fix site, `order` matches
    half the haystack. An agent that greps its best term first is the strong
    baseline; one that greps alphabetically is a straw man.
    """
    words = re.findall(r"[A-Za-z_][A-Za-z0-9_]*", prompt)
    kept = [word for word in words if word.lower() not in STOP_WORDS and len(word) > 2]
    unique = sorted(set(kept), key=lambda word: (-len(word), word))
    return unique


# Both arms are told which test is failing, because in a real session the test
# command says so before the agent does anything. Withholding it from one arm
# would make the comparison about who guesses the test path, not about evidence
# granularity.
SYSTEM_GREP = ("You are fixing a bug. Tools: grep(pattern), read_file(path).\n"
               "Failing test file(s): {tests}\n"
               "Task: {prompt}")
SYSTEM_MAP = ("You are fixing a bug. Tools: read_symbol(name), read_file(path).\n"
              "A map of the repository is provided; symbol bodies are omitted.\n"
              "Failing test file(s): {tests}\n"
              "Task: {prompt}")
SYSTEM_QUERY = ("You are fixing a bug. Tools: find_symbol(terms), "
                "read_symbol(name), read_file(path).\n"
                "Failing test file(s): {tests}\n"
                "Task: {prompt}")


def _test_paths(case: Dict[str, Any]) -> List[str]:
    return list(case.get("labels", {}).get("tests", []))
def run_grep_loop(repository: Repository, case: Dict[str, Any],
                  required: Sequence[Tuple[str, str]],
                  *, test_hint: bool = True) -> Transcript:
    """grep the best prompt term, read the best hit whole, follow its imports.

    This is the policy a competent engineer runs in a terminal, and it is the arm
    to beat. It is given every advantage the tool surface allows: it greps its
    highest-signal term first and it reads whole files, so it never misses
    evidence that is present in a file it opened.
    """
    hinted = _test_paths(case) if test_hint else []
    transcript = Transcript(SYSTEM_GREP.format(prompt=case["prompt"],
                                               tests=", ".join(hinted) or "(unknown)"))
    terms = _identifiers(case["prompt"])
    read: List[str] = []
    # Reading the failing test first is what the loop does in practice, and it is
    # the only way either arm learns the expected behaviour.
    queue: List[str] = list(hinted)

    def drain() -> bool:
        """Read every queued file whole, enqueuing each one's local imports."""
        while queue and transcript.turns < MAX_TURNS:
            path = queue.pop(0)
            read.append(path)
            transcript.act("read_file {}".format(path), repository.read_file(path))
            if transcript.satisfies(required):
                return True
            # Following imports is how the loop crosses a file boundary at all:
            # grep on the prompt cannot name a callee the prompt never mentions.
            for dependency in repository.imports_of(path):
                if dependency not in read and dependency not in queue:
                    queue.append(dependency)
        return False

    if drain():
        return transcript

    for term in terms:
        if transcript.turns >= MAX_TURNS:
            break
        matches = repository.grep(term)
        transcript.act("grep {}".format(term), matches)
        if transcript.satisfies(required):
            return transcript
        # Ordering by hit count is what a human does when scanning grep output:
        # the file that mentions the symbol most is the one to open.
        counts: Dict[str, int] = {}
        for line in matches.splitlines():
            if ":" not in line:
                continue
            counts[line.split(":", 1)[0]] = counts.get(line.split(":", 1)[0], 0) + 1
        for path in sorted(counts, key=lambda item: (-counts[item], item)):
            if path not in read and path not in queue:
                queue.append(path)
        if drain():
            return transcript
    return transcript


def run_map_expand(repository: Repository, case: Dict[str, Any],
                   service: ContextService,
                   required: Sequence[Tuple[str, str]],
                   *, test_hint: bool = True,
                   warm_prefix: bool = False) -> Transcript:
    """Map in the prefix once, then request only the symbol bodies on call edges."""
    repo_map = build_repo_map(service.store)
    hinted = _test_paths(case) if test_hint else []
    transcript = Transcript(SYSTEM_MAP.format(prompt=case["prompt"],
                                              tests=", ".join(hinted) or "(unknown)"),
                            cacheable_prefix=repo_map, warm_prefix=warm_prefix)
    terms = _identifiers(case["prompt"])

    # The map already names every symbol, so "which symbol" is answered without a
    # tool call. The only open question is which bodies to pay for.
    if hinted:
        tests = set(hinted)
    else:
        # Without the hint the map still supplies the test inventory for free. In
        # this haystack there is exactly one test directory, so that discovery is
        # nearly costless; in a repository with thousands of tests it would not be,
        # and this arm's no-hint numbers should be read as an upper bound.
        tests = {node.path for node in service.store.all_nodes(["function"])
                 if node.path.startswith("tests/") and node.name.startswith("test_")}
    wanted: List[str] = [node.name for node in service.store.all_nodes(["function"])
                         if node.path in tests]
    seeds = [term for term in terms if term in repo_map]
    for seed in seeds:
        if seed not in wanted:
            wanted.append(seed)
        for node in callees(service.store, seed, depth=2):
            if node.name not in wanted:
                wanted.append(node.name)

    for name in wanted:
        if transcript.turns >= MAX_TURNS:
            break
        bodies = [node for node in service.store.all_nodes(["function", "class"])
                  if node.name == name]
        if not bodies:
            continue
        rendered = "\n\n".join(
            "# {}:{}\n{}".format(node.path, node.start_line, node.content)
            for node in bodies)
        transcript.act("read_symbol {}".format(name), rendered)
        if transcript.satisfies(required):
            return transcript
    return transcript


def _bodies(service: ContextService, name: str) -> str:
    nodes = [node for node in service.store.all_nodes(["function", "class"])
             if node.name == name]
    return "\n\n".join("# {}:{}\n{}".format(node.path, node.start_line, node.content)
                       for node in nodes)


def run_map_query(repository: Repository, case: Dict[str, Any],
                  service: ContextService,
                  required: Sequence[Tuple[str, str]],
                  *, test_hint: bool = True) -> Transcript:
    """The map behind a tool call instead of in the prefix.

    `map_expand` pays for the whole skeleton on every turn to answer one question:
    where does this task start. This arm asks that question directly — one lookup
    returning the map lines for matching symbols — and then navigates call edges
    exactly as `map_expand` does. If the map's value really is entry-point
    discovery rather than evidence transport, this arm should keep `map_expand`'s
    solve rate at something close to `grep_loop`'s cost.
    """
    hinted = _test_paths(case) if test_hint else []
    transcript = Transcript(SYSTEM_QUERY.format(prompt=case["prompt"],
                                               tests=", ".join(hinted) or "(unknown)"))
    terms = _identifiers(case["prompt"])
    symbols = service.store.all_nodes(["function", "class"])

    # One lookup: which declared symbols mention the task's vocabulary, and where.
    lines: List[str] = []
    for node in symbols:
        # The hint names a file, so its symbols are in scope without a lookup;
        # without a hint the test inventory has to come from the index itself.
        in_hinted_test = node.path in set(hinted)
        discovered_test = (not hinted and node.path.startswith("tests/")
                           and node.name.startswith("test_"))
        if (any(term in node.name for term in terms) or in_hinted_test
                or discovered_test):
            lines.append("{}:{} {}".format(node.path, node.start_line,
                                          (node.signature or node.name).strip()))
    transcript.act("find_symbol {}".format(" ".join(terms)),
                   "\n".join(lines) or "(no matches)")
    if transcript.satisfies(required):
        return transcript

    wanted: List[str] = []
    for line in lines:
        name = line.split()[-1].split("(")[0].replace("def ", "").replace("class ", "")
        if name and name not in wanted:
            wanted.append(name)
    for seed in list(wanted):
        for node in callees(service.store, seed, depth=2):
            if node.name not in wanted:
                wanted.append(node.name)

    for name in wanted:
        if transcript.turns >= MAX_TURNS:
            break
        rendered = _bodies(service, name)
        if not rendered:
            continue
        transcript.act("read_symbol {}".format(name), rendered)
        if transcript.satisfies(required):
            return transcript
    return transcript


def _record(arm: str, case: Dict[str, Any], transcript: Transcript,
            required: Sequence[Tuple[str, str]]) -> Dict[str, Any]:
    return {
        "arm": arm,
        "case": case["id"],
        "solved": transcript.satisfies(required),
        "required_evidence": [label for label, _ in required],
        "missing_evidence": transcript.missing(required),
        "turns": transcript.turns,
        "final_transcript_tokens": transcript.final_tokens(),
        # The headline cost: what the provider bills across the whole session when
        # nothing caches, which is the status quo for a task-conditioned prompt.
        "no_cache_input_tokens": transcript.no_cache_input,
        "cached_input_tokens": round(transcript.cached_input),
        "new_tokens": transcript.new_tokens,
        "calls": [call for call, _ in transcript.entries],
    }


def run_case(case: Dict[str, Any], root: Path,
             *, test_hint: bool = True,
             warm_prefix: bool = False) -> List[Dict[str, Any]]:
    repository_path = (root / case["repository"]).resolve()
    repository = Repository(repository_path)
    service = ContextService(repository_path)
    try:
        service.index()
        required = _required_evidence(case, service)
        if not required:
            raise RuntimeError("{}: no labelled evidence to require".format(case["id"]))
        records = [_record("grep_loop", case,
                           run_grep_loop(repository, case, required,
                                         test_hint=test_hint), required)]
        records.append(_record("map_expand", case,
                               run_map_expand(repository, case, service, required,
                                              test_hint=test_hint,
                                              warm_prefix=warm_prefix),
                               required))
        records.append(_record("map_query", case,
                               run_map_query(repository, case, service, required,
                                             test_hint=test_hint),
                               required))
    finally:
        service.close()
    return records


def summarize(records: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    arms: Dict[str, List[Dict[str, Any]]] = {}
    for record in records:
        arms.setdefault(record["arm"], []).append(record)
    summary = {}
    for arm, rows in arms.items():
        solved = [row for row in rows if row["solved"]]
        # Cost is averaged over *solved* cases only: a case that never found the
        # evidence spent tokens without buying anything, and folding it into the
        # mean would reward an arm for giving up early.
        summary[arm] = {
            "cases": len(rows),
            "solved": len(solved),
            "mean_turns": round(sum(row["turns"] for row in solved) / len(solved), 2)
            if solved else None,
            "mean_no_cache_input_tokens": round(
                sum(row["no_cache_input_tokens"] for row in solved) / len(solved))
            if solved else None,
            "mean_cached_input_tokens": round(
                sum(row["cached_input_tokens"] for row in solved) / len(solved))
            if solved else None,
            "mean_final_transcript_tokens": round(
                sum(row["final_transcript_tokens"] for row in solved) / len(solved))
            if solved else None,
        }
    return summary


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", default="evaluation/haystack_cases.json")
    parser.add_argument("--root", default=".")
    parser.add_argument("--output")
    parser.add_argument("--no-test-hint", action="store_true",
                        help="withhold the failing test path from both arms")
    arguments = parser.parse_args(argv)

    root = Path(arguments.root).resolve()
    cases = json.loads(Path(arguments.cases).read_text(encoding="utf-8"))
    if isinstance(cases, dict):
        cases = cases["cases"]

    records: List[Dict[str, Any]] = []
    # A static repository map is only a cache *read* once some earlier case in the
    # run has already written it, and only for the same repository. Tracking which
    # ones have been seen keeps the first case in each repository paying the write.
    warmed: set = set()
    for case in cases:
        repository = case["repository"]
        records.extend(run_case(case, root, test_hint=not arguments.no_test_hint,
                                warm_prefix=repository in warmed))
        warmed.add(repository)

    summary = summarize(records)
    payload = {"cases": len(cases), "test_hint": not arguments.no_test_hint,
               "summary": summary, "runs": records,
               "note": "scripted policies, no model: tool-economics evidence only"}
    text = json.dumps(payload, indent=2)
    if arguments.output:
        Path(arguments.output).write_text(text + "\n", encoding="utf-8")
    print(json.dumps({"summary": summary}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
