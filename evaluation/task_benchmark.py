"""Run controlled classic-versus-GraphRAG coding-task evaluations.

Each arm gets an isolated copy of the same repository and the same user task.
Arms are named retrieval ablations (see ARMS): `classic` is lexical BM25 only,
`graphrag` is lexical + task-conditioned graph traversal, and the remaining arms
isolate each evidence source so the graph's marginal contribution is
identifiable. Models must return JSON containing complete file contents, which
makes edits reproducible and testable.
"""

import argparse
import json
import math
import re
import shutil
import subprocess
import tempfile
import time
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

from context_compiler.compiler import StoppingRule, estimate_tokens
from context_compiler.retrieval import RETRIEVAL_MODES
from context_compiler.providers import (ChatResponse, ModelClient, ProviderConfig,
                                        ProviderError, Usage, load_env_file)
from context_compiler.service import ContextService

from evaluation.variance import variance_report


# Ablation arms: display name -> (retrieval mode, human label).
ARMS: Dict[str, Dict[str, str]] = {
    "full_file": {"mode": "full_file", "label": "full-file context (no retrieval)"},
    "classic": {"mode": "lexical", "label": "lexical BM25 only (classic CLI)"},
    "vector": {"mode": "vector", "label": "vector (sparse tf-idf) only"},
    "graph_only": {"mode": "graph", "label": "graph traversal only"},
    "lexical_vector": {"mode": "lexical_vector", "label": "lexical + vector"},
    "graphrag": {"mode": "lexical_graph", "label": "lexical + graph (GraphRAG CLI)"},
    "lexical_vector_graph": {"mode": "lexical_vector_graph",
                             "label": "lexical + vector + graph"},
    # Rank-normalized and conditional replacements for the weighted hybrids
    # above. The weighted sum compares bm25, cosine, and path decay on
    # incomparable scales, and three separate results say that is what makes the
    # hybrid arms lose to the graph alone.
    "graphrag_rrf": {"mode": "lexical_graph_rrf",
                     "label": "lexical + graph, reciprocal-rank fusion"},
    "lexical_vector_graph_rrf": {"mode": "lexical_vector_graph_rrf",
                                 "label": "lexical + vector + graph, RRF"},
    "graph_escalate": {"mode": "graph_escalate",
                       "label": "graph first, lexical only on an empty frontier"},
    "graph_escalate_vector": {"mode": "graph_escalate_vector",
                              "label": "graph first, lexical + vector on an empty frontier"},
    "oracle": {"mode": "oracle", "label": "oracle human-selected context"},
}

DEFAULT_ARMS = ("classic", "graphrag")


@dataclass
class RunResult:
    task_id: str
    arm: str
    repetition: int
    success: bool
    test_exit_code: Optional[int]
    changed_files: int
    context_tokens: int
    input_tokens: Optional[int]
    output_tokens: Optional[int]
    cache_read_tokens: Optional[int]
    cache_write_tokens: Optional[int]
    latency_ms: float
    estimated_cost_usd: Optional[float]
    model: Optional[str]
    token_budget: Optional[int] = None
    retrieval_mode: Optional[str] = None
    context_items: Optional[int] = None
    # input_tokens counts only what the provider billed as fresh. A cached
    # read is still context the model saw, so cost claims must use the sum.
    total_prompt_tokens: Optional[int] = None
    # Which sufficiency condition ended evidence selection, and how much of the
    # budget was deliberately left unspent. An arm that always reports
    # "token_budget" with zero left over is filling its window, not deciding.
    stop_reason: Optional[str] = None
    tokens_left_unused: Optional[int] = None
    open_references: Optional[int] = None
    # A trial where the provider returned no content at all is a harness failure,
    # not a model failure. Scoring it as a loss transfers provider flakiness into
    # the arm's success rate, and whichever arm happens to draw the empty response
    # gets charged for it -- in the first live 8-arm run `classic` lost one task
    # this way, which is a third of its recorded failures.
    void: bool = False
    error: Optional[str] = None


# A file body carries newlines, double quotes and backslashes, which is exactly
# what JSON string escaping is worst at. Live models emit them raw, so a
# JSON-encoded body fails to parse on essentially every response and the benchmark
# ends up measuring escaping discipline instead of coding ability. The block format
# needs no escaping at all.
FILE_BLOCK = re.compile(
    r"^###\s*FILE:\s*(?P<path>\S+)\s*$\n(?P<body>.*?)^###\s*END\s*FILE\s*$",
    re.M | re.S)


def _response_blocks(text: str) -> Optional[Dict[str, str]]:
    """Parse the `### FILE: path` / `### END FILE` protocol, or return None."""
    matches = list(FILE_BLOCK.finditer(text))
    if not matches:
        return None
    return {match.group("path").strip(): match.group("body") for match in matches}


def _response_json(text: str) -> Mapping[str, Any]:
    candidate = text.strip()
    if candidate.startswith("```"):
        candidate = re.sub(r"^```(?:json)?\s*|\s*```$", "", candidate, flags=re.I | re.S).strip()
    try:
        # strict=False tolerates the literal newlines models put inside strings.
        # Unescaped quotes inside a body are still fatal, which is why the block
        # protocol above is preferred and this path is only a fallback.
        value = json.loads(candidate, strict=False)
    except json.JSONDecodeError as error:
        raise ValueError("model response is not valid JSON: {}".format(error)) from error
    if not isinstance(value, dict) or not isinstance(value.get("files"), dict):
        raise ValueError("model response must be an object with a files object")
    return value


def _response_files(text: str) -> Mapping[str, Any]:
    """Extract `{path: contents}` from either supported response protocol."""
    blocks = _response_blocks(text)
    if blocks is not None:
        return {"files": blocks, "explanation": "file-block protocol"}
    return _response_json(text)


def _apply_files(root: Path, files: Mapping[str, Any]) -> int:
    changed = 0
    for relative, content in files.items():
        if not isinstance(relative, str) or Path(relative).is_absolute() or ".." in Path(relative).parts:
            raise ValueError("unsafe model file path: {!r}".format(relative))
        if not isinstance(content, str):
            raise ValueError("file content must be a string: {!r}".format(relative))
        target = root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        before = target.read_text(encoding="utf-8") if target.exists() else None
        if before != content:
            target.write_text(content, encoding="utf-8")
            changed += 1
    return changed


def _cost(usage: Usage, input_rate: Optional[float], output_rate: Optional[float],
          cache_read_rate: Optional[float]) -> Optional[float]:
    if input_rate is None or output_rate is None:
        return None
    prompt = usage.input_tokens or 0
    completion = usage.output_tokens or 0
    cached = usage.cache_read_tokens or 0
    cached_rate = input_rate if cache_read_rate is None else cache_read_rate
    return (prompt * input_rate + completion * output_rate + cached * cached_rate) / 1_000_000


def _wilson_interval(successes: int, attempts: int, z: float = 1.959963984540054) -> list:
    if attempts == 0:
        return [0.0, 0.0]
    proportion = successes / attempts
    denominator = 1 + z * z / attempts
    center = (proportion + z * z / (2 * attempts)) / denominator
    margin = z * math.sqrt((proportion * (1 - proportion) + z * z / (4 * attempts)) / attempts) / denominator
    return [round(max(0.0, center - margin), 6), round(min(1.0, center + margin), 6)]


def _paired_exact_test(rows: Iterable[Mapping[str, Any]]) -> Dict[str, Any]:
    discordant = [row for row in rows if row["classic"] != row["graphrag"]]
    classic_wins = sum(1 for row in discordant if row["classic"] and not row["graphrag"])
    graph_wins = sum(1 for row in discordant if row["graphrag"] and not row["classic"])
    total = len(discordant)
    if total == 0:
        p_value = 1.0
    else:
        lower = sum(math.comb(total, k) for k in range(min(classic_wins, graph_wins) + 1)) / (2 ** total)
        p_value = min(1.0, 2 * lower)
    return {"discordant_pairs": total, "classic_only_wins": classic_wins,
            "graphrag_only_wins": graph_wins, "two_sided_exact_p_value": round(p_value, 6)}


class MockClient:
    """Deterministic local client used to test benchmark plumbing.

    It returns the expected patch only when the compiled context contains the
    task's required symbols. This is not a model-quality benchmark.
    """

    def __init__(self, task: Mapping[str, Any]):
        self.task = task

    def complete(self, messages, *, system=None, max_tokens=None, temperature=None):
        required = self.task.get("mock_requires", [])
        visible = str(system or "")
        files = self.task.get("mock_files", {}) if all(str(item) in visible for item in required) else {}
        payload = {"files": files, "explanation": "deterministic benchmark client"}
        text = json.dumps(payload, sort_keys=True)
        return ChatResponse(text=text, model="mock", usage=Usage(
            input_tokens=estimate_tokens(visible) + estimate_tokens(str(messages)),
            output_tokens=estimate_tokens(text),
        ))


class CodexCliClient:
    """Use the authenticated Codex CLI as a model while exposing usage data."""

    def __init__(self, cwd: Path, model: Optional[str] = None):
        self.cwd = cwd
        self.model = model

    def complete(self, messages, *, system=None, max_tokens=None, temperature=None):
        prompt = "SYSTEM CONTEXT:\n{}\n\nUSER REQUEST:\n{}".format(
            system or "", messages[-1].get("content", "") if messages else ""
        )
        command = ["codex", "exec", "--ephemeral", "-s", "read-only",
                   "--skip-git-repo-check", "--json"]
        if self.model:
            command.extend(["--model", self.model])
        command.append(prompt)
        completed = subprocess.run(command, cwd=str(self.cwd), text=True,
                                   capture_output=True, timeout=300)
        if completed.returncode != 0:
            raise RuntimeError((completed.stdout + "\n" + completed.stderr).strip()[-4000:])
        text = ""
        usage = Usage()
        model = self.model
        for line in completed.stdout.splitlines():
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if event.get("type") == "item.completed" and event.get("item", {}).get("type") == "agent_message":
                text = str(event["item"].get("text", ""))
            if event.get("type") == "turn.completed":
                raw_usage = event.get("usage") or {}
                total_input = raw_usage.get("input_tokens")
                cached_input = raw_usage.get("cached_input_tokens")
                uncached_input = (total_input - cached_input
                                  if isinstance(total_input, int) and isinstance(cached_input, int)
                                  else total_input)
                usage = Usage(uncached_input, raw_usage.get("output_tokens"),
                              cached_input, raw_usage.get("cache_write_input_tokens"))
        if not text:
            raise ValueError("Codex CLI returned no agent message")
        return ChatResponse(text=text, model=model or "codex-cli", usage=usage,
                            raw={"stdout": completed.stdout})


def run_task(task: Mapping[str, Any], arm: str, *, provider: Optional[ModelClient], mock: bool,
             codex_cli: bool,
             token_budget: int, max_tokens: Optional[int], input_rate: Optional[float],
             output_rate: Optional[float], cache_read_rate: Optional[float] = None,
             repetition: int = 1, force_token_budget: bool = False,
             max_items: Optional[int] = None, cache_bust: bool = True,
             stopping: Optional[StoppingRule] = None) -> RunResult:
    if arm not in ARMS:
        raise ValueError("unknown arm {!r}; expected one of {}".format(arm, ", ".join(ARMS)))
    mode = ARMS[arm]["mode"]
    # A per-task token_budget normally wins, which is how a task declares that it
    # needs a wider window. That silently defeats --token-budget, so the effective
    # budget is recorded on every row and can be forced uniform for budget sweeps.
    effective_budget = (
        int(token_budget) if force_token_budget
        else int(task.get("token_budget", token_budget))
    )
    source = Path(str(task["repository"])).resolve()
    with tempfile.TemporaryDirectory(prefix="graph-context-") as temporary:
        root = Path(temporary) / source.name
        shutil.copytree(source, root)
        started = time.perf_counter()
        context_service = ContextService(root, stopping=stopping)
        try:
            context_service.index()
            compiled = context_service.compile_context(
                str(task["prompt"]), token_budget=effective_budget,
                task_type=task.get("task_type"),
                # Read the depth off the mode rather than inferring it from the
                # name: `graph_escalate` and the RRF variants use traversal too,
                # and a name-suffix test silently gave them depth 0.
                graph_depth=RETRIEVAL_MODES[mode].graph_depth if mode in RETRIEVAL_MODES else 0,
                max_items=int(max_items if max_items is not None
                              else task.get("max_items", 20)),
                mode=mode,
                oracle_targets=task.get("oracle_context"),
            )
        finally:
            context_service.close()
        instruction = (
            "Fix the task by rewriting complete files. Reply with nothing but one "
            "block per changed file, in exactly this form:\n\n"
            "### FILE: relative/path.py\n"
            "<the complete new contents of that file>\n"
            "### END FILE\n\n"
            "Do not wrap the blocks in markdown fences and do not escape anything "
            "inside them.\n\nTask: "
            + str(task["prompt"])
        )
        if mock:
            client = MockClient(task)
        elif codex_cli:
            model_workspace = Path(temporary) / "model-workspace"
            model_workspace.mkdir()
            client = CodexCliClient(model_workspace)
        else:
            client = provider
        if client is None:
            raise ValueError("provider is required unless --mock is used")
        system = compiled["context"]
        if cache_bust:
            # Arms share a long prefix, so whichever arm runs second is served most
            # of its prompt from the provider's cache and reports a fraction of the
            # input tokens the first arm paid. That makes the iso-token comparison
            # an artefact of arm order: in the first live smoke run `graphrag` was
            # billed 82 uncached input tokens against `classic`'s 3,524 purely
            # because it ran second. A per-call nonce denies every arm that subsidy.
            system = "cache-bust: {}\n{}".format(uuid.uuid4(), system)
        transport_error = None
        try:
            response = client.complete([{"role": "user", "content": instruction}],
                                       system=system, max_tokens=max_tokens)
        except ProviderError as exc:
            # Retries are already exhausted by here, so the call is not coming back.
            # A provider that stops answering is the same class of event as one that
            # answers with nothing: it grades neither way, so it is a void trial and
            # not a loss for whichever arm was unlucky enough to be in flight. This
            # also keeps one dead socket from discarding every other call in the run,
            # which is how the first live budget sweep was lost.
            transport_error = str(exc)
            response = ChatResponse(text="", model=None)
        error = None
        changed = 0
        exit_code: Optional[int] = None
        try:
            if transport_error is not None:
                # Nothing to apply and nothing to grade; running the suite here would
                # only record the fixture's own pre-existing failure as this arm's.
                raise RuntimeError(transport_error)
            changed = _apply_files(root, _response_files(response.text)["files"])
            command = task.get("test_command", "python3 -m unittest discover -s tests -v")
            completed = subprocess.run(command, cwd=str(root), shell=True, text=True, capture_output=True, timeout=float(task.get("test_timeout_seconds", 120)))
            exit_code = completed.returncode
            if exit_code != 0:
                error = (completed.stdout + "\n" + completed.stderr).strip()[-4000:]
            elif re.search(r"Ran\s+0\s+tests?", completed.stdout + "\n" + completed.stderr):
                error = "test command ran zero tests"
        except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as exc:
            error = str(exc)
        usage = response.usage
        # No content at all means nothing was measured: an empty completion cannot
        # be graded for coding ability, and treating it as a wrong answer credits the
        # other arms for a provider hiccup.
        void = transport_error is not None or not str(response.text or "").strip()
        return RunResult(task_id=str(task["id"]), arm=arm, repetition=repetition, success=error is None and exit_code == 0,
                         test_exit_code=exit_code, changed_files=changed,
                         context_tokens=int(compiled["estimated_tokens"]),
                         input_tokens=usage.input_tokens, output_tokens=usage.output_tokens,
                         cache_read_tokens=usage.cache_read_tokens, cache_write_tokens=usage.cache_write_tokens,
                         latency_ms=round((time.perf_counter() - started) * 1000, 3),
                         estimated_cost_usd=_cost(usage, input_rate, output_rate, cache_read_rate),
                         model=response.model, token_budget=effective_budget,
                         retrieval_mode=mode,
                         context_items=int(compiled["item_count"]),
                         total_prompt_tokens=(usage.input_tokens or 0) + (usage.cache_read_tokens or 0),
                         stop_reason=(compiled.get("stopping") or {}).get("reason"),
                         tokens_left_unused=(compiled.get("stopping") or {}).get("tokens_left_unused"),
                         open_references=(compiled.get("stopping") or {}).get("open_references"),
                         void=void,
                         error=transport_error or error)


def benchmark(tasks: Iterable[Mapping[str, Any]], *, mock: bool, token_budget: int,
              max_tokens: Optional[int], input_rate: Optional[float], output_rate: Optional[float],
              cache_read_rate: Optional[float] = None, repetitions: int = 1,
              codex_cli: bool = False, arms: Sequence[str] = DEFAULT_ARMS,
              baseline_arm: str = "classic", treatment_arm: str = "graphrag",
              force_token_budget: bool = False, cache_bust: bool = True,
              max_items: Optional[int] = None,
              stopping: Optional[StoppingRule] = None,
              provider: Optional[Any] = None) -> Dict[str, Any]:
    tasks = list(tasks)
    arms = list(dict.fromkeys(arms))
    unknown = [arm for arm in arms if arm not in ARMS]
    if unknown:
        raise ValueError("unknown arms: {}".format(", ".join(unknown)))
    # Injectable so a test can supply a failing provider; a live run leaves it
    # unset and the credential is read from the environment as before.
    if provider is None and not (mock or codex_cli):
        provider = ModelClient(ProviderConfig.from_env())
    results = []
    for task in tasks:
        for arm in arms:
            if ARMS[arm]["mode"] == "oracle" and not task.get("oracle_context"):
                continue
            for repetition in range(1, repetitions + 1):
                results.append(run_task(task, arm, provider=provider, mock=mock, codex_cli=codex_cli, token_budget=token_budget,
                                        max_tokens=max_tokens, input_rate=input_rate, output_rate=output_rate,
                                        cache_read_rate=cache_read_rate,
                                        repetition=repetition,
                                        force_token_budget=force_token_budget,
                                        max_items=max_items, cache_bust=cache_bust,
                                        stopping=stopping))
    rows = [asdict(row) for row in results]
    budgets = sorted({row["token_budget"] for row in rows if row["token_budget"] is not None})
    summary: Dict[str, Any] = {
        "task_count": len(tasks), "arms": arms, "results": rows,
        "token_budgets_used": budgets,
        # Item count can bind before the token budget does. On a large repository
        # 20 items is reached well below 4,000 tokens, which would make the
        # comparison a top-20 ranking test rather than an iso-token test.
        "max_items": max_items,
        "max_items_uniform": max_items is not None,
        "stopping_rule": (None if stopping is None or not stopping.enabled
                          else {"cliff_ratio": stopping.cliff_ratio,
                                "patience": stopping.patience,
                                "min_items": stopping.min_items,
                                "require_closure": stopping.require_closure}),
        # Arms share a prompt prefix; without a per-call nonce the arm that runs
        # second is billed a fraction of the first arm's input tokens.
        "cache_bust": cache_bust,
        # An iso-token comparison is only meaningful when one budget was in force.
        "token_budget_uniform": len(budgets) <= 1,
    }
    for arm in arms:
        arm_rows = [row for row in rows if row["arm"] == arm]
        if not arm_rows:
            continue
        # A void trial returned no content, so it grades neither way. Both
        # denominators are reported: `success_rate` over every attempt, and
        # `success_rate_graded` over the trials that actually produced an answer.
        graded = [row for row in arm_rows if not row.get("void")]
        summary[arm] = {"label": ARMS[arm]["label"],
                        "retrieval_mode": ARMS[arm]["mode"],
                        "attempts": len(arm_rows),
                        "voids": len(arm_rows) - len(graded),
                        "graded_attempts": len(graded),
                        "successes": sum(row["success"] for row in arm_rows),
                        "success_rate": sum(row["success"] for row in arm_rows) / len(arm_rows),
                        "success_rate_graded": (sum(row["success"] for row in graded) / len(graded)
                                                if graded else None),
                        "success_rate_ci95": _wilson_interval(sum(row["success"] for row in arm_rows), len(arm_rows)),
                        "success_rate_graded_ci95": (_wilson_interval(
                            sum(row["success"] for row in graded), len(graded)) if graded else None),
                        "mean_context_tokens": sum(row["context_tokens"] for row in arm_rows) / len(arm_rows),
                        "mean_context_items": sum(row["context_items"] or 0 for row in arm_rows) / len(arm_rows),
                        "mean_latency_ms": sum(row["latency_ms"] for row in arm_rows) / len(arm_rows),
                        "total_input_tokens": sum(row["input_tokens"] or 0 for row in arm_rows),
                        "total_output_tokens": sum(row["output_tokens"] or 0 for row in arm_rows),
                        "total_cache_read_tokens": sum(row["cache_read_tokens"] or 0 for row in arm_rows),
                        "total_cost_usd": sum(row["estimated_cost_usd"] or 0 for row in arm_rows),
                        # Sufficiency accounting. `mean_tokens_left_unused` is the
                        # quantity the stopping rule is supposed to move: budget an
                        # arm chose not to spend while still passing.
                        "stop_reasons": _count_stop_reasons(arm_rows),
                        "mean_tokens_left_unused": (
                            sum(row["tokens_left_unused"] or 0 for row in arm_rows) / len(arm_rows)),
                        "mean_open_references": (
                            sum(row["open_references"] or 0 for row in arm_rows) / len(arm_rows))}
    by_task: Dict[Any, Dict[str, Any]] = {}
    for row in rows:
        by_task.setdefault((row["task_id"], row["repetition"]), {})[row["arm"]] = row
    if baseline_arm in arms and treatment_arm in arms:
        summary["comparison"] = {"baseline_arm": baseline_arm, "treatment_arm": treatment_arm}
        # A pair is only comparable when both sides produced an answer. Keeping a
        # pair whose baseline went void would score a provider hiccup as a win for
        # the treatment.
        pairs = [arms_row for (_task_id, _rep), arms_row in sorted(by_task.items())
                 if baseline_arm in arms_row and treatment_arm in arms_row
                 and not arms_row[baseline_arm].get("void")
                 and not arms_row[treatment_arm].get("void")]
        summary["comparison"]["comparable_pairs"] = len(pairs)
        summary["comparison"]["pairs_dropped_void"] = sum(
            1 for (_t, _r), arms_row in by_task.items()
            if baseline_arm in arms_row and treatment_arm in arms_row
            and (arms_row[baseline_arm].get("void") or arms_row[treatment_arm].get("void")))
        summary["paired_deltas"] = [
            {
                "task_id": arms_row[treatment_arm]["task_id"],
                "repetition": arms_row[treatment_arm]["repetition"],
                "success_delta_treatment_minus_baseline": int(arms_row[treatment_arm]["success"]) - int(arms_row[baseline_arm]["success"]),
                "context_token_delta_treatment_minus_baseline": arms_row[treatment_arm]["context_tokens"] - arms_row[baseline_arm]["context_tokens"],
                "input_token_delta_treatment_minus_baseline": (arms_row[treatment_arm]["input_tokens"] or 0) - (arms_row[baseline_arm]["input_tokens"] or 0),
                "latency_ms_delta_treatment_minus_baseline": round(arms_row[treatment_arm]["latency_ms"] - arms_row[baseline_arm]["latency_ms"], 3),
            }
            for arms_row in pairs
        ]
        summary["paired_exact_test"] = _paired_exact_test([
            {"classic": arms_row[baseline_arm]["success"],
             "graphrag": arms_row[treatment_arm]["success"]}
            for arms_row in pairs
        ])
    summary["marginal_graph_contribution"] = _marginal_contribution(summary, arms)
    # Within-arm dispersion, and the verdict on whether any success delta in this
    # summary clears it. With `repetitions=1` the verdict is null rather than
    # true: one run per arm cannot distinguish an effect from a coin flip, which
    # is how the first live comparison reported variance as a finding.
    summary["variance"] = variance_report(rows, arms=arms, baseline_arm=baseline_arm,
                                          treatment_arm=treatment_arm)
    return summary


def _count_stop_reasons(rows: Sequence[Mapping[str, Any]]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for row in rows:
        reason = row.get("stop_reason") or "unrecorded"
        counts[reason] = counts.get(reason, 0) + 1
    return counts


def _marginal_contribution(summary: Mapping[str, Any], arms: Sequence[str]) -> Dict[str, Any]:
    """Isolate what adding graph traversal buys over the same arm without it."""
    pairs = (("classic", "graphrag"), ("lexical_vector", "lexical_vector_graph"))
    output: Dict[str, Any] = {}
    for without, with_graph in pairs:
        if without in summary and with_graph in summary:
            output["{}_to_{}".format(without, with_graph)] = {
                "success_rate_delta": round(
                    summary[with_graph]["success_rate"] - summary[without]["success_rate"], 6),
                "mean_context_token_delta": round(
                    summary[with_graph]["mean_context_tokens"] - summary[without]["mean_context_tokens"], 3),
                "total_input_token_delta": summary[with_graph]["total_input_tokens"] - summary[without]["total_input_tokens"],
            }
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", type=Path, required=True)
    parser.add_argument("--mock", action="store_true", help="use deterministic local client; not a model-quality result")
    parser.add_argument("--codex-cli", action="store_true", help="use authenticated Codex CLI in a read-only empty workspace")
    parser.add_argument("--token-budget", type=int, default=4000)
    parser.add_argument("--force-token-budget", action="store_true",
                        help="ignore per-task token_budget overrides so every arm and "
                             "task shares one budget (required for iso-token claims)")
    parser.add_argument("--max-tokens", type=int)
    parser.add_argument("--max-items", type=int,
                        help="uniform context item cap for every arm and task; "
                             "raise it on large repositories so the token budget, "
                             "not the item count, is the binding constraint")
    parser.add_argument("--input-price-per-million", type=float)
    parser.add_argument("--output-price-per-million", type=float)
    parser.add_argument("--cache-read-price-per-million", type=float)
    parser.add_argument("--repetitions", type=int, default=1)
    parser.add_argument("--arms", nargs="+", default=list(DEFAULT_ARMS),
                        choices=sorted(ARMS), metavar="ARM",
                        help="ablation arms to run; default: {}".format(" ".join(DEFAULT_ARMS)))
    parser.add_argument("--all-arms", action="store_true",
                        help="run every ablation arm: {}".format(", ".join(sorted(ARMS))))
    parser.add_argument("--baseline-arm", default="classic", choices=sorted(ARMS))
    parser.add_argument("--treatment-arm", default="graphrag", choices=sorted(ARMS))
    parser.add_argument("--stopping", action="store_true",
                        help="stop packing evidence at sufficiency (reference closure, score "
                             "cliff, or a flat tail) instead of at the token budget; this is a "
                             "manipulated variable, so compare against a run without it")
    parser.add_argument("--stopping-cliff", type=float, default=StoppingRule.cliff_ratio,
                        help="stop when the next candidate scores below this fraction of the best")
    parser.add_argument("--stopping-patience", type=int, default=StoppingRule.patience,
                        help="stop after this many consecutive candidates that add no new file "
                             "and close no outstanding reference")
    parser.add_argument("--stopping-min-items", type=int, default=StoppingRule.min_items,
                        help="never stop on sufficiency before this many items are selected")
    parser.add_argument("--stopping-ignore-closure", action="store_true",
                        help="disable the reference-closure condition, leaving only the score "
                             "cliff and the flat-tail patience")
    parser.add_argument("--output", type=Path, help="also write the JSON report to this path")
    parser.add_argument("--no-cache-bust", action="store_true",
                        help="allow arms to share the provider prompt cache; makes "
                             "input-token comparisons depend on arm order")
    parser.add_argument("--env-file", default=".env",
                        help="KEY=value file to load credentials from for live runs")

    args = parser.parse_args()
    if args.mock and args.codex_cli:
        parser.error("--mock and --codex-cli are mutually exclusive")
    if not (args.mock or args.codex_cli):
        # A live run needs a credential, and an `export` in an interactive shell
        # never reaches this subprocess. Names only — the value is not logged.
        loaded = load_env_file(args.env_file)
        if loaded:
            print("loaded from {}: {}".format(args.env_file, ", ".join(sorted(loaded))))
        # Fail before any repository is copied or indexed. Without this the run
        # sends an unauthenticated request per arm per task and reports six
        # identical 401s as if they were model failures.
        config = ProviderConfig.from_env()
        if not config.api_key:
            parser.error(
                "no API key for provider {!r}: add its key to {} (see the commented "
                "line in that file), or use --mock".format(config.provider, args.env_file))
        print("live run: provider={} model={}".format(config.provider, config.model))
    arms: List[str] = sorted(ARMS) if args.all_arms else list(args.arms)
    tasks = json.loads(args.tasks.read_text(encoding="utf-8"))
    if not isinstance(tasks, list):
        raise SystemExit("tasks file must contain a JSON list")
    report = benchmark(tasks, mock=args.mock, token_budget=args.token_budget,
                       max_tokens=args.max_tokens, input_rate=args.input_price_per_million,
                       output_rate=args.output_price_per_million,
                       cache_read_rate=args.cache_read_price_per_million,
                       repetitions=max(1, args.repetitions), codex_cli=args.codex_cli,
                       arms=arms, baseline_arm=args.baseline_arm,
                       treatment_arm=args.treatment_arm,
                       force_token_budget=args.force_token_budget,
                       max_items=args.max_items,
                       cache_bust=not args.no_cache_bust,
                       stopping=(StoppingRule(cliff_ratio=args.stopping_cliff,
                                              patience=args.stopping_patience,
                                              min_items=args.stopping_min_items,
                                              require_closure=not args.stopping_ignore_closure)
                                 if args.stopping else None))
    rendered = json.dumps(report, indent=2, sort_keys=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
