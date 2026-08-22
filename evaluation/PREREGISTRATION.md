# Pre-registration: SWE-bench Lite replication

Registered before any SWE-bench instance has been run. Nothing below may be
changed after the first live call; deviations go in a "Deviations" section at the
bottom, dated, with the reason.

## Why this run exists

Every result in this project so far comes from generated repositories. The
generator plants near-miss distractors that share vocabulary with the target. The
graph arms' advantage may therefore be an advantage over *that* distractor
distribution rather than over real code. This run is the external-validity gate:
if the advantage does not survive real repositories, the earlier findings do not
generalize and the write-up must say so.

## Hypotheses

Directional, and stated so that a null is publishable.

- **H1 (primary).** `graph_escalate` reaches the target success rate at a lower
  token budget than `classic`.
- **H2 (co-primary).** `graph_escalate` induces fewer output tokens than `classic`
  at equal context size.
- **H3 (co-primary).** `graph_escalate` has lower wall-clock latency than
  `classic`.
- **H4 (interaction, directional).** The H1 advantage increases with gold-patch
  file count: absent or negative in `single_file`, largest in `multi_file`. This is
  the prediction that distinguishes "the graph helps" from "the generator was
  favourable" -- a uniform advantage across strata would be evidence for the
  latter.

## Endpoints

- **Primary:** budget-at-fixed-success — the smallest swept token budget at which
  an arm resolves 100% of its non-void trials in a stratum
  (`budget_sweep.py::_tokens_to_reach`). Arms that never reach it are reported as
  `unreached`, never as the largest swept budget.
- **Co-primary:** mean output tokens and mean wall-clock latency at the largest
  swept budget.
- **Secondary:** uncached input tokens; mean context tokens; stop reason
  distribution and mean unused budget when `--stopping` is on.
- **Precondition, not a contrast:** success rate. Reported only as "all arms
  reached N/N at budget B". A success difference at a single budget is not a
  result in this design, because within-arm swing has already been measured at 0.50
  against a between-arm spread of 0.42.
- **Bounds:** `oracle` (gold-patch files) is the information floor and `full_file`
  the ceiling. Every arm is reported as a ratio to both.

## Design

- **Sample.** 10 instances per stratum, 30 total, drawn by
  `evaluation/swebench.py::sample` with seed 20260822 from a local export of
  `princeton-nlp/SWE-bench_Lite`. The dataset SHA-256 prefix and the sampled
  instance ids are recorded by `python3 -m evaluation.swebench plan` before the
  run and are part of this registration.
- **Strata.** `single_file` (0-1 gold production files), `two_file` (2),
  `multi_file` (3+). Test files and `conftest.py` do not count toward the stratum.
- **Arms.** `classic` (lexical, baseline), `graph_escalate` (treatment),
  `graph_only`, `graphrag` (weighted fusion, retained as the negative control),
  `oracle` (floor), `full_file` (ceiling). Six arms.
- **Repetitions.** 5 per instance per arm. This is not optional: a single
  repetition produced the project's one non-replicating result.
- **Budgets swept.** 512, 1024, 2048, 4000, 8000, 16000 tokens, uniform across
  arms via `--force-token-budget`.
- **Stopping rule.** Off for the registered run. A second, separately reported run
  with `--stopping` at its default settings (cliff 0.15, patience 4, min items 3,
  closure required) is exploratory.
- **Model.** One model, fixed before the run, recorded in the result file. No model
  is swapped mid-run; a swap starts a new registration.
- **Mock.** No mock arm. Mock grading on real repositories would restate the oracle
  labels back to themselves.

## Test-name withholding

The prompt is the redacted `problem_statement` only. `evaluation/swebench.py`
strips pytest node ids, `test_*` / `*_test` / `Test*` identifiers, and paths into
test trees, and `plan` refuses to emit a sample whose prompts still contain any
`FAIL_TO_PASS` token. The `test_patch` is applied to disk, because the withheld
tests must exist to be run, but never enters the context. Grading runs only the
`FAIL_TO_PASS` node ids.

Rationale: an unredacted SWE-bench prompt frequently names the failing test, and a
test file names the symbol under test. Leaving that in makes the task a lookup and
hands the lexical arm the answer, which would confirm H1's null for a reason that
has nothing to do with retrieval.

## Analysis

- Paired at the instance level: each arm sees the same instances at the same
  budgets and repetition indices.
- Deltas are treatment-minus-baseline per instance, with 95% percentile bootstrap
  intervals over 4,000 resamples, seed 20260822
  (`evaluation/variance.py::paired_bootstrap`).
- An interval that straddles zero is reported as no effect. No p-value is used as a
  threshold for whether a finding is reported.
- Within-arm success swing is computed and printed first. If maximum within-arm
  swing is at least the between-arm spread, all success comparisons are labelled
  uninterpretable and only the continuous endpoints are discussed.
- Strata are analyzed separately and reported separately. The pooled number is
  reported too, but as a summary, not as the test of H1, since the stratum mix is
  set by the sampler and not by the population.
- No multiplicity correction across the three strata; the strata are reported as
  three separate directional results, not searched for a winner.

## Exclusions, fixed in advance

- **Void trials.** An empty provider completion is a harness failure, excluded from
  both numerator and denominator, and the void count is reported per arm. A void
  is never scored as a failed task.
- **Instances that cannot be prepared.** Clone failure, patch-application failure,
  or a test command that errors on the unmodified tree (so the withheld test does
  not actually fail before the fix) — excluded, listed by id, and replaced by the
  next instance in the same stratum's seeded draw order.
- **Instances with no `FAIL_TO_PASS`** are rejected at load, not scored.
- No instance is excluded after its results are seen.

## What would falsify the claim

- H1 null or reversed in `multi_file`: the graph advantage does not survive real
  repositories, and the generated-haystack results are an artifact of the
  distractor generator.
- H4 reversed — advantage largest in `single_file` — the advantage is not about
  cross-file structure at all.
- `graphrag` matching or beating `graph_escalate`: the fusion finding was specific
  to generated vocabulary overlap.
- H2 and H3 null: the only endpoints that have reproduced so far do not reproduce
  where it counts, and the project has no positive result.

Any of these outcomes is reported with the same prominence as a confirmation.

## Deviations

None yet.
