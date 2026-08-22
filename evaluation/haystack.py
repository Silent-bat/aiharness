"""Generate a haystack repository whose source cannot fit the context budget.

The six seeded fixtures are 129-281 tokens each, so a 4,000-token budget never
binds and `full_file` wins by brute force. That regime cannot test retrieval.
This generator embeds a seeded bug in a repository an order of magnitude larger
than the budget, which makes `full_file` structurally impossible and forces every
arm to compete on *what it selects* rather than on how much it can include.

Fairness rules, so the haystack does not quietly rig the result for the graph:

* Distractor names are drawn from the same vocabulary as the target symbols, so
  lexical and vector arms face genuine near-miss competition rather than noise.
* Some distractors are deliberately stronger lexical matches for the task prompt
  than the real fix site is. Being misled by surface similarity is the realistic
  failure mode and the baseline is entitled to it.
* Distractors call each other across modules, so graph expansion also yields a
  large candidate set. The graph must discriminate by ranking; it is not handed a
  smaller search space than the other arms.
* Generation is deterministic given `--seed` and is blind to which arm will run.
"""

import argparse
import json
import random
import shutil
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

# Vocabulary shared with the seeded fixtures, so distractor symbols are genuine
# near misses for the task prompts rather than unrelated filler.
NOUNS = ("order", "cart", "amount", "payment", "invoice", "charge", "total",
         "payload", "request", "scope", "credential", "attempt", "retry",
         "slug", "title", "price", "ledger", "receipt", "session", "quota")
VERBS = ("validate", "check", "verify", "submit", "process", "compute", "apply",
         "normalize", "sanitize", "resolve", "authorize", "capture", "reject",
         "prepare", "settle", "reconcile", "audit", "flush", "assert", "permit")
QUALIFIERS = ("legacy", "v2", "internal", "strict", "cached", "async", "batch",
              "draft", "fallback", "shadow")

MODULE_TEMPLATE = '''"""Generated haystack module {index}. Not the fix site."""

{imports}

{constants}

{functions}
'''

FUNCTION_TEMPLATE = '''def {name}({params}) -> {returns}:
    """{doc}"""
{body}
'''


def _symbol(random_state: random.Random, *, qualified: bool = False) -> str:
    parts = [random_state.choice(VERBS), random_state.choice(NOUNS)]
    if qualified:
        parts.append(random_state.choice(QUALIFIERS))
    return "_".join(parts)


def _function(random_state: random.Random, name: str,
              callees: Sequence[str]) -> str:
    noun = random_state.choice(NOUNS)
    params = "{}: int = 0".format(noun) if random_state.random() < 0.6 else \
        "{}: dict".format(noun)
    returns = random_state.choice(("int", "bool", "str"))
    body_lines = [
        "    if not {}:".format(noun),
        "        return {}".format(
            {"int": "0", "bool": "False", "str": '""'}[returns]),
    ]
    for callee in callees:
        body_lines.append("    {} = {}({})".format(
            random_state.choice(NOUNS) + "_value", callee, noun))
        body_lines.append("    if not {}_value:".format(
            body_lines[-1].split("_value")[0].strip()))
        body_lines.append("        pass")
    body_lines.append("    return {}".format(
        {"int": "len(str({}))".format(noun),
         "bool": "bool({})".format(noun),
         "str": "str({})".format(noun)}[returns]))
    return FUNCTION_TEMPLATE.format(
        name=name, params=params, returns=returns,
        doc="Generated helper for {} handling.".format(noun),
        body="\n".join(body_lines))


def _decoy_module(random_state: random.Random, prompt_terms: Sequence[str],
                  index: int, names: Sequence[str]) -> str:
    """A module engineered to out-match the real fix site lexically.

    It must still export every name planned for this index: later modules import
    those names, and a decoy that dropped them would break the repository at
    import time.
    """
    terms = list(prompt_terms)
    random_state.shuffle(terms)
    doc = " ".join(terms)
    functions = []
    used = set()
    for name in names:
        functions.append(FUNCTION_TEMPLATE.format(
            name=name, params="value: int = 0", returns="bool",
            doc="Handles {}. This is a decoy, not the defect.".format(doc),
            body="    # {}\n    return bool(value)".format(doc)))
    for term in terms[:3]:
        safe = "".join(character if character.isalnum() or character == "_" else "_"
                       for character in term).strip("_").lower() or "handler"
        # "credential" and "credential." sanitize to the same slug; emitting both
        # would define one name twice in this module.
        if safe in used:
            continue
        used.add(safe)
        functions.append(FUNCTION_TEMPLATE.format(
            name="{}_{}".format(safe, index), params="value: int = 0",
            returns="bool",
            doc="Handles {}. This is a decoy, not the defect.".format(doc),
            body="    # {}\n    return bool(value)".format(doc)))
    return MODULE_TEMPLATE.format(
        index=index, imports="", constants='DECOY_NOTE = "{}"'.format(doc),
        functions="\n\n".join(functions))


def generate(base: Path, destination: Path, *, modules: int = 120, seed: int = 7,
             prompt: str = "") -> Dict[str, Any]:
    """Copy `base` into `destination` and surround it with generated modules."""
    if destination.exists():
        shutil.rmtree(destination)
    shutil.copytree(base, destination)
    shutil.rmtree(destination / ".context-compiler", ignore_errors=True)

    random_state = random.Random(seed)
    package = destination / "modules"
    package.mkdir(exist_ok=True)
    (package / "__init__.py").write_text("", encoding="utf-8")

    # Pass one: name every module's public symbols so later modules can import
    # earlier ones and produce real cross-module call edges.
    names: List[List[str]] = []
    for _ in range(modules):
        count = random_state.randint(2, 4)
        drawn = [_symbol(random_state, qualified=random_state.random() < 0.5)
                 for _ in range(count)]
        # Duplicate draws would emit two `def`s with one name; the second would
        # shadow the first and the module's exported surface would shrink.
        names.append(sorted(set(drawn)))

    prompt_terms = [term for term in prompt.replace(".", " ").split() if len(term) > 3]
    decoy_indexes = set(random_state.sample(range(modules), k=max(1, modules // 20)))

    for index in range(modules):
        if index in decoy_indexes and prompt_terms:
            (package / "mod_{:03d}.py".format(index)).write_text(
                _decoy_module(random_state, prompt_terms, index, names[index]),
                encoding="utf-8")
            continue
        imports: List[str] = []
        local = set(names[index])
        # A callee that shares a name with a local def would be shadowed by that
        # def, turning the generated call into unintended self-recursion.
        available = [(other, name) for other in range(index) for name in names[other]
                     if name not in local]
        functions = []
        for name in names[index]:
            callees: List[str] = []
            if available and random_state.random() < 0.7:
                for other, callee in random_state.sample(
                        available, k=min(len(available), random_state.randint(1, 2))):
                    imports.append("from .mod_{:03d} import {}".format(other, callee))
                    callees.append(callee)
            functions.append(_function(random_state, name, callees))
        (package / "mod_{:03d}.py".format(index)).write_text(
            MODULE_TEMPLATE.format(
                index=index, imports="\n".join(sorted(set(imports))),
                constants="MODULE_INDEX = {}".format(index),
                functions="\n\n".join(functions)),
            encoding="utf-8")

    sources = sorted(destination.rglob("*.py"))
    characters = sum(len(path.read_text(encoding="utf-8")) for path in sources)
    return {"repository": str(destination), "file_count": len(sources),
            "approximate_tokens": (characters + 2) // 3,
            "generated_modules": modules, "decoy_modules": sorted(decoy_indexes),
            "seed": seed}


def build_cases(task_cases: Sequence[Mapping[str, Any]], output_root: Path, *,
                modules: int, seed: int,
                only: Optional[Sequence[str]] = None) -> Dict[str, Any]:
    output_root.mkdir(parents=True, exist_ok=True)
    cases: List[Dict[str, Any]] = []
    reports: List[Dict[str, Any]] = []
    for task in task_cases:
        if only and task["id"] not in only:
            continue
        base = Path(str(task["repository"]))
        destination = output_root / "haystack_{}".format(base.name)
        report = generate(base, destination, modules=modules, seed=seed,
                          prompt=str(task.get("prompt", "")))
        report["task_id"] = task["id"]
        reports.append(report)
        case = dict(task)
        case["repository"] = str(destination)
        # A per-task budget would defeat the point: the budget must stay far below
        # repository size for every haystack case.
        case.pop("token_budget", None)
        cases.append(case)
    return {"cases": cases, "reports": reports}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("evaluation/haystack"))
    parser.add_argument("--cases-out", type=Path,
                        default=Path("evaluation/haystack_cases.json"))
    parser.add_argument("--modules", type=int, default=120)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--only", nargs="+", help="restrict to these task ids")
    args = parser.parse_args()
    tasks = json.loads(args.tasks.read_text(encoding="utf-8"))
    built = build_cases(tasks, args.out, modules=args.modules, seed=args.seed,
                        only=args.only)
    args.cases_out.parent.mkdir(parents=True, exist_ok=True)
    args.cases_out.write_text(json.dumps(built["cases"], indent=2) + "\n",
                              encoding="utf-8")
    print(json.dumps({"cases_written": str(args.cases_out),
                      "repositories": built["reports"]}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
