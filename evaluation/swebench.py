"""Convert SWE-bench Lite instances into benchmark cases, with test names withheld.

This exists because every number in this project so far comes from generated
repositories. The generator plants near-miss distractors that share vocabulary
with the target, and the graph arms' advantage may be an advantage over *that*
specific kind of distractor rather than over real code. If so, every result
collapses on contact with a real repository, and it is better to find out before
writing anything up.

Three design constraints, each of which a naive loader gets wrong:

1. **Test names are withheld.** A SWE-bench instance ships `FAIL_TO_PASS` node
   ids, and the problem statement frequently names the failing test, the test
   file, or a traceback frame inside it. Handing any of that to a retrieval system
   turns the task into a lookup: the test file names the symbol under test, so
   lexical search wins by construction and the comparison measures nothing. The
   loader redacts test identifiers from the prompt and keeps the node ids in a
   separate field the model never sees.
2. **Stratified by gold-patch file count.** A one-file fix is the regime where
   lexical search is sufficient and the graph cannot help; a three-file fix is the
   regime the graph exists for. Pooling them lets the stratum mix decide the
   result. Strata are `single_file`, `two_file`, `multi_file`.
3. **Pre-registered.** The endpoint, the arms, the strata, and the exclusion rules
   are fixed in `evaluation/PREREGISTRATION.md` before the run. A loader that
   emits a sample chosen after looking at the results is not evidence.

Input is a local JSON or JSONL export of `princeton-nlp/SWE-bench_Lite`, so the
loader has no network dependency and a run is reproducible from a file hash:

    python3 -m evaluation.swebench prepare --dataset swebench_lite.json \\
        --workspaces evaluation/swebench_repos \\
        --output evaluation/swebench_cases.json --per-stratum 10

`prepare` clones each instance's repository at its `base_commit` (network, once)
and writes the case file. `plan` does the stratification only, with no cloning,
so the sample can be inspected and registered first.
"""

import argparse
import hashlib
import json
import random
import re
import subprocess
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

# Fixed so a sample is reproducible from the dataset hash alone.
SAMPLING_SEED = 20260822

STRATA = ("single_file", "two_file", "multi_file")

# Anything that could name a test to the model. Applied to the problem statement,
# which is the only instance field the prompt is built from.
_TEST_PATTERNS = (
    # pytest node ids, with or without a class and parameters
    re.compile(r"\S*tests?/\S*::\S+"),
    re.compile(r"\btest_[A-Za-z0-9_]+\b"),
    re.compile(r"\b[A-Za-z0-9_]+_test\b"),
    re.compile(r"\bTest[A-Z][A-Za-z0-9_]*\b"),
    # bare paths into a test tree, including the unittest dotted form
    re.compile(r"\S*\btests?\b/\S+"),
    re.compile(r"\b[A-Za-z0-9_.]*\btests?\b[A-Za-z0-9_.]*\.test_[A-Za-z0-9_]+\b"),
)

REDACTION = "<test name withheld>"


def load_dataset(path: Path) -> List[Dict[str, Any]]:
    """Read a JSON list or a JSONL file of SWE-bench instances."""
    text = path.read_text(encoding="utf-8")
    stripped = text.lstrip()
    if stripped.startswith("["):
        rows = json.loads(text)
    else:
        rows = [json.loads(line) for line in text.splitlines() if line.strip()]
    if not isinstance(rows, list):
        raise ValueError("dataset must be a JSON list or JSONL")
    required = ("instance_id", "repo", "base_commit", "problem_statement", "patch")
    for row in rows:
        missing = [field for field in required if not row.get(field)]
        if missing:
            raise ValueError("instance {!r} is missing {}".format(
                row.get("instance_id", "?"), ", ".join(missing)))
    return rows


def dataset_fingerprint(path: Path) -> str:
    """Hash the dataset file so a pre-registration can name the exact sample."""
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return "sha256:{}".format(digest[:32])


def patch_files(patch: str) -> List[str]:
    """Files touched by a unified diff, excluding test files.

    SWE-bench gold patches are production-only by construction (test changes live
    in `test_patch`), but instances are not uniformly clean, and a test file
    counted into the stratum would misclassify the task's difficulty.
    """
    files: List[str] = []
    for match in re.finditer(r"^diff --git a/(\S+) b/(\S+)", patch, re.MULTILINE):
        path = match.group(2)
        if _is_test_path(path) or path in files:
            continue
        files.append(path)
    return files


def _is_test_path(path: str) -> bool:
    parts = path.split("/")
    name = parts[-1]
    return (any(part in ("test", "tests", "testing") for part in parts)
            or name.startswith("test_") or name.endswith("_test.py")
            or name == "conftest.py")


def stratum_of(file_count: int) -> str:
    if file_count <= 1:
        return "single_file"
    if file_count == 2:
        return "two_file"
    return "multi_file"


def withhold_test_names(text: str) -> str:
    """Redact test identifiers from a problem statement.

    Deliberately over-broad: a false redaction costs the model a little context,
    while a missed one hands the lexical arm the answer and silently invalidates
    the comparison.
    """
    redacted = text
    for pattern in _TEST_PATTERNS:
        redacted = pattern.sub(REDACTION, redacted)
    return redacted


def leaks_test_names(text: str, node_ids: Sequence[str]) -> List[str]:
    """Test identifiers still present in `text`, as a post-redaction assertion."""
    found: List[str] = []
    for node_id in node_ids:
        for token in re.split(r"[:\[\]/. ]+", node_id):
            if len(token) < 5 or not _looks_like_test_token(token):
                continue
            if token in text and token not in found:
                found.append(token)
    return found


def _looks_like_test_token(token: str) -> bool:
    return (token.startswith("test_") or token.endswith("_test")
            or token.startswith("Test") or token in ("test", "tests"))


def stratify(instances: Iterable[Mapping[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    """Group instances by gold-patch file count."""
    buckets: Dict[str, List[Dict[str, Any]]] = {name: [] for name in STRATA}
    for instance in instances:
        files = patch_files(str(instance["patch"]))
        record = dict(instance)
        record["_gold_files"] = files
        buckets[stratum_of(len(files))].append(record)
    return buckets


def sample(buckets: Mapping[str, Sequence[Mapping[str, Any]]], per_stratum: int,
           seed: int = SAMPLING_SEED) -> List[Dict[str, Any]]:
    """Draw up to `per_stratum` instances from each stratum, deterministically.

    Sorted by instance id before shuffling so the draw does not depend on the
    order the dataset file happened to be written in.
    """
    chosen: List[Dict[str, Any]] = []
    for name in STRATA:
        pool = sorted(buckets.get(name, []), key=lambda row: str(row["instance_id"]))
        rng = random.Random("{}:{}".format(seed, name))
        rng.shuffle(pool)
        chosen.extend(dict(row) for row in pool[:per_stratum])
    return chosen


def test_node_ids(instance: Mapping[str, Any]) -> List[str]:
    """The FAIL_TO_PASS ids, which are withheld from the prompt but run for grading."""
    raw = instance.get("FAIL_TO_PASS") or instance.get("fail_to_pass") or []
    if isinstance(raw, str):
        raw = json.loads(raw)
    return [str(item) for item in raw]


def build_case(instance: Mapping[str, Any], workspace: Path) -> Dict[str, Any]:
    """Turn one instance into a case in this harness's schema.

    `oracle_context` is the gold patch's file list: the human-labeled information
    floor, the same role it plays on the generated fixtures. No `mock_requires` or
    `mock_files` is emitted, because a mock run against a real repository would
    only restate the oracle labels back to itself.
    """
    node_ids = test_node_ids(instance)
    prompt = withhold_test_names(str(instance["problem_statement"])).strip()
    leaks = leaks_test_names(prompt, node_ids)
    gold_files = list(instance.get("_gold_files") or patch_files(str(instance["patch"])))
    return {
        "id": str(instance["instance_id"]),
        "repository": str(workspace),
        "prompt": prompt,
        "task_type": "debug",
        # Graded by running only the withheld tests. The command is never rendered
        # into the prompt, so naming them here does not leak them to the model.
        "test_command": _pytest_command(node_ids),
        "oracle_context": gold_files,
        "labels": {
            "relevant_files": gold_files,
            "gold_patch_file_count": len(gold_files),
            "stratum": stratum_of(len(gold_files)),
        },
        "swebench": {
            "repo": str(instance["repo"]),
            "base_commit": str(instance["base_commit"]),
            "environment_setup_commit": instance.get("environment_setup_commit"),
            "fail_to_pass": node_ids,
            "pass_to_pass": _as_list(instance.get("PASS_TO_PASS")),
            "test_patch_applied": bool(instance.get("test_patch")),
            # Non-empty means the redaction failed and the case must not be run.
            "prompt_test_name_leaks": leaks,
        },
    }


def _as_list(raw: Any) -> List[str]:
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except json.JSONDecodeError:
            return [raw]
    return [str(item) for item in (raw or [])]


def _pytest_command(node_ids: Sequence[str]) -> str:
    if not node_ids:
        raise ValueError("instance has no FAIL_TO_PASS tests, so it cannot be graded")
    quoted = " ".join("'{}'".format(node) for node in node_ids)
    return "python3 -m pytest -q {}".format(quoted)


def clone(instance: Mapping[str, Any], workspaces: Path) -> Path:
    """Clone the instance repository at its base commit, once, into `workspaces`.

    Detached at `base_commit` so the task starts from the pre-fix tree. The clone
    keeps its `.git` directory: `test_patch` application and per-arm resets both
    need it, and the indexer ignores it.
    """
    target = workspaces / str(instance["instance_id"])
    if (target / ".git").exists():
        return target
    workspaces.mkdir(parents=True, exist_ok=True)
    url = "https://github.com/{}.git".format(instance["repo"])
    subprocess.run(["git", "clone", "--quiet", url, str(target)], check=True)
    subprocess.run(["git", "-C", str(target), "checkout", "--quiet",
                    str(instance["base_commit"])], check=True)
    if instance.get("test_patch"):
        # The withheld tests only exist after the test patch is applied. Applied
        # to disk but never shown to the model, which is exactly the withholding
        # the design requires.
        patch_path = target / ".swebench_test_patch.diff"
        patch_path.write_text(str(instance["test_patch"]), encoding="utf-8")
        subprocess.run(["git", "-C", str(target), "apply", str(patch_path)], check=True)
        patch_path.unlink()
    return target


def plan_report(dataset: Path, per_stratum: int, seed: int = SAMPLING_SEED) -> Dict[str, Any]:
    """The pre-registrable description of a sample: what it is, before it is run."""
    instances = load_dataset(dataset)
    buckets = stratify(instances)
    drawn = sample(buckets, per_stratum, seed=seed)
    leaks = {}
    for instance in drawn:
        prompt = withhold_test_names(str(instance["problem_statement"]))
        found = leaks_test_names(prompt, test_node_ids(instance))
        if found:
            leaks[str(instance["instance_id"])] = found
    return {
        "dataset": str(dataset),
        "dataset_fingerprint": dataset_fingerprint(dataset),
        "instances_available": len(instances),
        "sampling_seed": seed,
        "per_stratum": per_stratum,
        "population_by_stratum": {name: len(rows) for name, rows in buckets.items()},
        "sampled_by_stratum": dict(Counter(
            stratum_of(len(row["_gold_files"])) for row in drawn)),
        "sampled_instance_ids": sorted(str(row["instance_id"]) for row in drawn),
        # Must be empty before any run. A non-empty map means the prompt still
        # names a test, which would hand the lexical arm the answer.
        "prompt_test_name_leaks": leaks,
    }


def prepare(dataset: Path, workspaces: Path, per_stratum: int,
            seed: int = SAMPLING_SEED, clone_repos: bool = True) -> List[Dict[str, Any]]:
    instances = load_dataset(dataset)
    drawn = sample(stratify(instances), per_stratum, seed=seed)
    cases = []
    for instance in drawn:
        workspace = (clone(instance, workspaces) if clone_repos
                     else workspaces / str(instance["instance_id"]))
        cases.append(build_case(instance, workspace))
    return cases


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("plan", "prepare"):
        item = sub.add_parser(name)
        item.add_argument("--dataset", type=Path, required=True,
                          help="local JSON or JSONL export of SWE-bench Lite")
        item.add_argument("--per-stratum", type=int, default=10)
        item.add_argument("--seed", type=int, default=SAMPLING_SEED)
        if name == "prepare":
            item.add_argument("--workspaces", type=Path, required=True,
                              help="directory to clone instance repositories into")
            item.add_argument("--no-clone", action="store_true",
                              help="write the case file without cloning; paths must already exist")
        item.add_argument("--output", type=Path)
    args = parser.parse_args(argv)

    if args.command == "plan":
        payload: Any = plan_report(args.dataset, args.per_stratum, seed=args.seed)
        if payload["prompt_test_name_leaks"]:
            print("REFUSING: {} sampled prompts still name a test after redaction".format(
                len(payload["prompt_test_name_leaks"])))
    else:
        payload = prepare(args.dataset, args.workspaces, args.per_stratum,
                          seed=args.seed, clone_repos=not args.no_clone)
        blocked = [case["id"] for case in payload
                   if case["swebench"]["prompt_test_name_leaks"]]
        if blocked:
            raise SystemExit("test names leaked into the prompt for: {}".format(
                ", ".join(blocked)))
    rendered = json.dumps(payload, indent=2, sort_keys=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
