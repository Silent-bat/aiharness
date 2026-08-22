# Graph Context Compiler

Graph Context Compiler is a research prototype for constructing task-specific
repository context for terminal coding agents. It combines exact text search,
static code relationships, task-conditioned graph traversal, reranking, and
token-budgeted context selection.

It is designed to run by itself from a terminal or as an MCP tool provider for
DeepSeek Harness, Codex, or another agent runtime.

## Current scope

The prototype currently provides:

- incremental Python repository indexing with SHA-256 change detection;
- file, class, function, method, containment, call, import, inheritance, and
  reference relationships;
- SQLite FTS5 lexical retrieval;
- graph expansion tuned for debugging, features, refactoring, security,
  performance, and testing tasks;
- selection of full definitions, bounded snippets, or signatures under a
  fixed token budget;
- provenance explaining why each context item was selected;
- symbol impact analysis;
- a dependency-free terminal CLI and MCP server;
- an evaluation command for retrieval recall, precision, and reciprocal rank.

This is an MVP, not a claim that repository context is solved. Dynamic traces,
embeddings, additional languages, learned reranking, and agent feedback loops
are intentionally left as subsequent research stages.

## Architecture

```text
Repository
   | Python AST + incremental digest scan
   v
SQLite code graph + FTS5 index
   | exact search seeds
   | task-conditioned graph traversal
   v
Hybrid ranked evidence
   | detail-level selection
   | token-budget packing
   v
Compiled context with provenance
   | CLI or MCP
   v
Terminal coding agent
```

## Quick start

Python 3.9 or newer is required. The core has no third-party runtime
dependencies.

```bash
python3 -m venv .venv
. .venv/bin/activate
python3 -m pip install -e .

context-compiler --repo examples/sample_repo index
context-compiler --repo examples/sample_repo search \
  "fix submit_order failing for an empty cart"
context-compiler --repo examples/sample_repo context \
  "fix submit_order failing for an empty cart" \
  --token-budget 1200
context-compiler --repo examples/sample_repo impact validate_amount

# Optional: send a context-augmented request through a supported provider.
AI_PROVIDER=deepseek DEEPSEEK_API_KEY="$DEEPSEEK_API_KEY" \
  context-compiler --repo examples/sample_repo ask \
  "why does submit_order reject an empty cart?" \
  --token-budget 1200
```

## Model Providers

The optional `ask` command uses a dependency-free provider gateway. Set
`AI_PROVIDER` and the matching key environment variable:

| Provider | `AI_PROVIDER` | Key variable |
|---|---|---|
| OpenAI | `openai` | `OPENAI_API_KEY` |
| DeepSeek | `deepseek` | `DEEPSEEK_API_KEY` |
| Anthropic | `anthropic` | `ANTHROPIC_API_KEY` |
| Gemini | `gemini` | `GEMINI_API_KEY` |
| OpenRouter | `openrouter` | `OPENROUTER_API_KEY` |
| Groq | `groq` | `GROQ_API_KEY` |
| Ollama | `ollama` | optional |
| Custom OpenAI-compatible server | `openai-compatible` | `AI_API_KEY` |

Use `AI_BASE_URL` and `AI_MODEL` to override the endpoint or model. The gateway
normalizes text, finish reasons, input/output tokens, and cache-read tokens;
provider-specific authentication remains inside the adapter.

Without installation, set `PYTHONPATH=src`:

```bash
PYTHONPATH=src python3 -m context_compiler.cli \
  --repo examples/sample_repo index
```

## DeepSeek Harness integration

Run the MCP server with the target repository in the environment:

```bash
CONTEXT_COMPILER_REPO=/absolute/path/to/repository \
  context-compiler-mcp
```

Register that command as a stdio MCP server in the Harness profile. The server
exposes these tools:

- `index_repository`
- `search_code_graph`
- `compile_repository_context`
- `analyze_symbol_impact`
- `code_graph_stats`

A typical agent workflow is:

1. Call `index_repository` at session start and after material edits.
2. Call `compile_repository_context` before planning or editing.
3. Use `search_code_graph` when a hypothesis needs more evidence.
4. Use `analyze_symbol_impact` before cross-file refactoring.

For a production Harness plugin, these MCP tools can become native Cordis
services and inject compiled evidence during `agent/pre-step`.

## Evaluation

Run the included retrieval evaluation:

```bash
PYTHONPATH=src python3 -m context_compiler.evaluation \
  --repo examples/sample_repo \
  --cases evaluation/sample_cases.json \
  --limit 10
```

Run the lexical-only versus graph-hybrid comparison, including retrieval
latency and estimated compiled-context tokens:

```bash
PYTHONPATH=src python3 evaluation/benchmark.py \
  --repo examples/sample_repo \
  --cases evaluation/sample_cases.json \
  --limit 10 \
  --token-budget 1200
```

Add `--all-modes` to score every retrieval arm on the same labeled cases.
`hybrid` and `lexical_graph` are the same arm under two names, which keeps the
historical numbers comparable across the refactor:

| Arm | Recall@10 | Precision@10 | MRR | Mean context tokens |
| --- | --- | --- | --- | --- |
| `lexical` | 0.833 | 0.450 | 1.000 | 274 |
| `vector` | 0.833 | 0.367 | 1.000 | 412 |
| `lexical_vector` | 0.833 | 0.367 | 1.000 | 412 |
| `graph` | 1.000 | 0.278 | 0.750 | 744 |
| `lexical_graph` (`hybrid`) | 1.000 | 0.278 | 1.000 | 744 |
| `lexical_vector_graph` | 1.000 | 0.278 | 1.000 | 772 |

Only the graph arms reach full recall, and they pay for it in precision and
tokens. Retrieval-level recall is an intermediate signal: the arm that matters is
whichever one produces a correct patch, which is what the task benchmark below
measures.

Run the end-to-end coding-task benchmark. It copies each fixture into an
isolated temporary directory, asks the same provider for a complete-file JSON
patch, applies it, runs the fixture tests, and reports success, latency,
input/output tokens, cache-read tokens, and optional cost estimates:

```bash
PYTHONPATH=src python3 evaluation/task_benchmark.py \
  --tasks evaluation/task_cases.json --mock --token-budget 1200 \
  --input-price-per-million 1 --output-price-per-million 2
```

`--mock` validates the harness deterministically and is not evidence of model
quality: the mock client returns the correct patch exactly when the compiled
context contains the task's declared symbols, so it measures retrieval, not the
model. For a real comparison, omit `--mock` and configure the provider with
`AI_PROVIDER`, its matching API key, and optionally `AI_MODEL`/`AI_BASE_URL`.
The default arms are `classic` (lexical only) and `graphrag` (lexical + graph at
depth two); see "Retrieval ablations" for the rest. Keep
the model, task fixtures, token budget, and test command fixed when reporting
results. Cache-read values are reported when a provider exposes them; the
runner does not pretend that a provider without cache accounting had a cache
hit.

An authenticated Codex CLI can also be used without an API key. The model is
run read-only in an empty workspace and receives only the compiled context:

```bash
PYTHONPATH=src python3 evaluation/task_benchmark.py \
  --tasks evaluation/task_cases.json --codex-cli \
  --token-budget 1200 --repetitions 3 \
  --input-price-per-million 1 \
  --output-price-per-million 4 \
  --cache-read-price-per-million 0.5
```

Use `--repetitions 3` or more in live runs to expose prompt-cache behavior.
When cached input has a discounted price, pass
`--cache-read-price-per-million`; otherwise cached tokens conservatively use
the normal input rate in the cost estimate.

The checked-in controlled fixtures currently contain six tasks, each one a
cross-file bug where the needed callee definition is only reachable through the
graph. Every task carries a `labels` block naming the relevant files, symbols,
callers, tests, and expected behavior, plus an `oracle_context` list used by the
oracle arm.

The live Codex runs in this workspace are deliberately reported as evidence,
not as a universal claim: one run produced 1/2 classic versus 2/2 GraphRAG,
while later repeated runs varied between both arms succeeding and GraphRAG
underperforming on one stochastic repair. More repositories, tasks, fixed model
settings, and statistical analysis are required before publication.

## Retrieval ablations

Every evidence source can be isolated so the graph's marginal contribution is
identifiable rather than assumed. Arms are selected with `--arms` or `--all-arms`:

| Arm | Retrieval |
| --- | --- |
| `full_file` | every repository file, full content, no ranking |
| `classic` | lexical BM25 only |
| `vector` | sparse TF-IDF cosine only |
| `graph_only` | graph traversal only; lexical seeds are discarded from the ranking |
| `lexical_vector` | lexical + vector |
| `graphrag` | lexical + task-conditioned graph traversal |
| `lexical_vector_graph` | lexical + vector + graph |
| `oracle` | the human-labeled files in `oracle_context` |

`vector` is a *sparse* TF-IDF space over subword-split identifiers, not a dense
embedding model. It is labeled `vector (sparse tf-idf)` everywhere so results are
never mistaken for a neural-embedding baseline. A dense provider can be
substituted by supplying an alternative embedding callable to `VectorIndex`.

```bash
PYTHONPATH=src python3 evaluation/task_benchmark.py \
  --tasks evaluation/task_cases.json --mock --all-arms \
  --force-token-budget --token-budget 4000 \
  --input-price-per-million 1 --output-price-per-million 2 \
  --output evaluation/results/mock_ablation_6task.json
```

Deterministic-client result over the six fixtures at one uniform 4,000-token
budget (`evaluation/results/mock_ablation_6task.json`):

| Arm | Success | Mean context tokens | Mean items | Total input tokens |
| --- | --- | --- | --- | --- |
| `full_file` | 6/6 | 423 | 4.0 | 3,081 |
| `classic` | 0/6 | 376 | 3.3 | 2,799 |
| `vector` | 2/6 | 447 | 3.7 | 3,224 |
| `graph_only` | 6/6 | 609 | 4.8 | 4,198 |
| `lexical_vector` | 2/6 | 447 | 3.7 | 3,224 |
| `graphrag` | 6/6 | 609 | 4.8 | 4,198 |
| `lexical_vector_graph` | 6/6 | 622 | 4.8 | 4,274 |
| `oracle` | 6/6 | 379 | 3.0 | 2,820 |

Read this as a mechanism check, not as efficacy evidence. Three findings matter,
including one that argues against the system:

1. Graph traversal, not lexical or vector matching, is what reaches the callee
   holding the defect. Adding the graph moves success from 0/6 to 6/6 over
   `classic`, and from 2/6 to 6/6 over `lexical_vector`.
2. `full_file` also scores 6/6 at *fewer* tokens than `graphrag` (423 versus
   609). On four-file fixtures, pasting the whole repository is both cheaper and
   sufficient, so these fixtures cannot support any token-efficiency claim for
   retrieval. That claim requires repositories too large to fit in the budget,
   which is what the haystack section below builds.
3. `oracle` scores 6/6 at 379 tokens, below every retrieval arm. The gap between
   `oracle` and `graphrag` (230 tokens per task) is the headroom left in evidence
   selection.

`vector` and `lexical_vector` coincide exactly here because the fixtures hold
only three or four files, so both arms select the same top items within budget.
That degeneracy is an artifact of fixture size, not a property of the arms.

Two caveats bound every number in that table. The deterministic client returns
the correct patch if and only if the compiled context contains the task's
declared symbols, so its success column measures retrieval, not model
competence; the p-value from such a run describes a construction, not a model.
And `graph_only` matching `graphrag` reflects the fixtures being built so the
target is graph-reachable, which is the mechanism under test.

## Budget sweep: the comparison that is decidable

A single-budget arm comparison cannot settle anything. The fixtures total
129-281 tokens against a 4,000-token default, a 14-31x surplus, so the budget
never binds and `full_file` is simultaneously the information ceiling and near
the cost floor. Retrieval can only lose. At the other extreme every arm fails.
What is decidable is success as a function of budget:

```bash
PYTHONPATH=src:. python3 evaluation/budget_sweep.py \
  --tasks evaluation/task_cases.json --mock \
  --output evaluation/results/mock_budget_sweep.json
```

Success and mean context tokens per arm
(`evaluation/results/mock_budget_sweep.json`):

| Budget | `full_file` | `classic` | `graphrag` | `oracle` |
| --- | --- | --- | --- | --- |
| 128 | 1/6 @110 | 0/6 @113 | 0/6 @123 | 0/6 @110 |
| 160 | 1/6 @130 | 0/6 @124 | 0/6 @146 | 0/6 @136 |
| 200 | 1/6 @181 | 0/6 @175 | 0/6 @174 | 0/6 @180 |
| 256 | 3/6 @234 | 0/6 @232 | 0/6 @246 | 5/6 @224 |
| 320 | 3/6 @296 | 0/6 @296 | 0/6 @304 | 5/6 @290 |
| 512 | 5/6 @409 | 0/6 @376 | 4/6 @492 | 6/6 @379 |
| 1024 | 6/6 @423 | 0/6 @376 | 6/6 @609 | 6/6 @379 |
| 4000 | 6/6 @423 | 0/6 @376 | 6/6 @609 | 6/6 @379 |

Smallest budget reaching a target pass rate:

| Arm | 50% | 100% |
| --- | --- | --- |
| `full_file` | 256 | 1024 |
| `classic` | never | never |
| `graphrag` | 512 | 1024 |
| `oracle` | 256 | 512 |

**At every budget that actually binds, GraphRAG is the worst of the three
non-lexical arms.** It needs 512 tokens to pass half the tasks where `full_file`
and `oracle` need 256, and at 512 it scores 4/6 against `full_file`'s 5/6. The
graph reaches the right symbol — that is what separates it from `classic`, which
never succeeds at any budget — but the compiler then spends the budget on
lower-value evidence: roughly 4.8 items at full detail, including test files and
near-duplicate regions. On these fixtures the bottleneck is evidence *selection*,
not graph reach, and the `oracle`-to-`graphrag` gap is the size of that defect.
This conclusion is specific to repositories that fit the budget; the haystack
section below shows it reversing once they do not.

This sweep ignores per-task `token_budget` overrides by design. Five of the six
fixtures pin `token_budget: 700`, which silently overrode `--token-budget` in
earlier runs; `--force-token-budget` on the task benchmark restores one uniform
budget, and every result row now records the `token_budget` actually in force
alongside a `token_budget_uniform` flag. Without that, an iso-token claim is
only iso-token by accident.

The sweep inherits the mock caveat, but this is the mock's legitimate use: with a
client that succeeds exactly when the required evidence is present, the curve
measures how efficiently each arm retains that evidence per token. It says
nothing about how a model behaves once the evidence is there.

The included evaluator is deliberately simple. Report task success, test-pass
rate, context precision, relevant-symbol recall, tokens, latency, and cost. Keep
the model, prompt, tool permissions, and token budget fixed across arms.

## Haystack: repositories that cannot fit the budget

Every result above is bounded by fixture size. `evaluation/haystack.py` removes
that bound: it copies a seeded fixture and surrounds it with 120 generated
modules, producing a repository of roughly 39,000 tokens against a 4,000-token
budget, so `full_file` becomes structurally impossible and every arm must compete
on *what it selects*.

The generator is written not to rig the result for the graph:

- distractor symbols are drawn from the same vocabulary as the target symbols, so
  lexical and vector arms face genuine near-miss competition, not noise;
- six modules per repository are deliberately *stronger* bm25 matches for the task
  prompt than the real fix site is — being misled by surface similarity is the
  realistic baseline failure mode and the baseline is entitled to it;
- distractors call each other across modules, so graph expansion also produces a
  large candidate set rather than a conveniently small one;
- generation is deterministic given `--seed` and blind to which arm will run;
- `build_cases` strips every per-task `token_budget`, so the fixture cannot widen
  its own window.

```bash
PYTHONPATH=src:. python3 evaluation/haystack.py \
  --tasks evaluation/task_cases.json --modules 120 --seed 7

PYTHONPATH=src:. python3 evaluation/task_benchmark.py \
  --tasks evaluation/haystack_cases.json --mock --all-arms \
  --force-token-budget --token-budget 4000 --max-items 200 \
  --output evaluation/results/mock_haystack_ablation_4000.json
```

`--max-items` matters here. The default cap of 20 items is reached at roughly
2,000 tokens on a repository this size, so without it the comparison silently
becomes a top-20 ranking test instead of an iso-token test. Every run records
`max_items` and `max_items_uniform`.

Deterministic-client result at a uniform 4,000-token budget, 200-item cap
(`evaluation/results/mock_haystack_ablation_4000.json`):

| Arm | Success | Mean context tokens | Mean items | Total input tokens |
| --- | --- | --- | --- | --- |
| `full_file` | 0/6 | 3,976 | 13.7 | 24,403 |
| `classic` | 1/6 | 3,974 | 53.7 | 24,386 |
| `vector` | 1/6 | 3,964 | 42.5 | 24,328 |
| `graph_only` | 6/6 | 3,975 | 45.3 | 24,395 |
| `lexical_vector` | 1/6 | 3,974 | 43.5 | 24,390 |
| `graphrag` | 5/6 | 3,980 | 45.0 | 24,426 |
| `lexical_vector_graph` | 1/6 | 3,980 | 38.0 | 24,426 |
| `oracle` | 6/6 | 379 | 3.0 | 2,820 |

Budget sweep on the same repositories
(`evaluation/results/mock_haystack_budget_sweep.json`):

| Budget | `full_file` | `classic` | `vector` | `graphrag` | `graph_only` | `oracle` |
| --- | --- | --- | --- | --- | --- | --- |
| 512 | 0/6 @490 | 0/6 @498 | 0/6 @493 | 1/6 @492 | 6/6 @489 | 6/6 @379 |
| 1024 | 0/6 @1006 | 0/6 @994 | 0/6 @992 | 1/6 @1011 | 6/6 @1000 | 6/6 @379 |
| 2048 | 0/6 @2028 | 0/6 @2016 | 1/6 @2009 | 1/6 @2031 | 6/6 @2034 | 6/6 @379 |
| 4000 | 0/6 @3976 | 1/6 @3974 | 1/6 @3964 | 5/6 @3980 | 6/6 @3975 | 6/6 @379 |
| 8000 | 0/6 @7970 | 1/6 @7150 | 2/6 @7980 | 6/6 @7919 | 6/6 @5738 | 6/6 @379 |

Four findings, one of which contradicts the small-fixture section above:

1. **`full_file` fails at every budget.** Once the repository is ten times the
   window, brute force is not an option, which is the regime the earlier
   `full_file` win could not speak to.
2. **Graph traversal is what works.** `graph_only` reaches 6/6 at 512 tokens
   while `classic`, `vector`, and `lexical_vector` stay at or below 1/6 up to
   8,000. Paired over the six tasks, `graphrag` beats `classic` 4-0 on discordant
   pairs at 4,000 tokens (two-sided exact p = 0.125 — the direction is clean, six
   tasks are not enough for significance) at 40 more total input tokens across all
   six, so this is an iso-token comparison.
3. **Mixing lexical evidence into the graph arm is what costs it.** `graph_only`
   needs 512 tokens for 6/6; `graphrag` needs 8,000, and `lexical_vector_graph`
   never gets past 1/6. Under near-miss distractors the lexical and vector hits
   fill the budget with decoys and crowd out the traversal result. The fusion
   weights, not graph reach, are the defect.
4. **`oracle` still wins on cost**, 6/6 at 379 tokens against `graph_only`'s 489,
   so budget-aware selection remains the largest single loss.

Two defects in the retriever were found by this benchmark and fixed, both of which
only surface when a repository is large enough to contain competing matches:

- `GraphStore.search` mapped bm25 to `1/(1+|bm25|)`. bm25 is negative and more
  negative is better, so the strongest lexical match received the *smallest*
  score. Row order was already best-first, which hid the inversion until a
  consumer re-ranked by score — which every retrieval arm does.
- Graph traversal seeded from raw bm25 row order rather than from the ranking the
  arm itself produces. The decoy modules monopolize the top of raw bm25 order, so
  the real fix site (raw rank 46 of 121) was never a seed and traversal never
  reached its callee. Seeds now come from lexical score, which is what admits the
  exact-symbol-name bonus.

Both fixes leave the small-fixture numbers unchanged (`lexical` recall 0.833,
graph arms 1.000 on `examples/sample_repo`; mock ablation within three tokens per
task), so they are corrections, not retuning.

The haystack inherits the mock caveat in full: the deterministic client succeeds
exactly when the required evidence is present, so these curves measure how
efficiently each arm retains that evidence per token, not how a model behaves once
it has it.

### First live 8-arm run, and why its headline is not yet a result

`evaluation/results/live_haystack_ablation_4000.json` is a real run against
`stealth/ox-alpha` over OpenRouter, six haystack tasks, one budget (4,000 tokens),
per-call cache-busting on. It reads `graphrag` 6/6 and `graph_only` 6/6 against
`classic` 3/6 and `full_file` 3/6, with `oracle` 6/6 at 379 context tokens.

That file is committed as-is, and it should not be quoted as evidence, for three
reasons:

1. **One of `classic`'s three failures is void, not wrong.** On
   `security-authorization-scope` the provider returned no content and no usage
   block, which the harness at the time scored as a failed task. That is a third of
   the baseline's recorded failures. It also broke the cost comparison in the
   baseline's favour: the void trial's input tokens went unrecorded, so the entire
   apparent +2,895-token gap against `graphrag` comes from one missing usage block.
   `RunResult.void` now records this, `success_rate_graded` reports over answered
   trials only, and void pairs are dropped from the paired test.
2. **Six tasks cannot reach significance.** Three discordant pairs all favouring the
   treatment give a two-sided exact p of 0.125. No arrangement of six tasks clears
   0.05, so this run is a smoke test of the harness, not a measurement.
3. **The fixtures are cross-file by construction**, which is the regime the graph is
   built for. Real single-file fixes, where lexical search suffices, are absent by
   selection.

## Turn economics: what a terminal agent actually pays

Every result above scores a *one-shot* context window. A terminal agent does not
work that way: it takes turns, and each turn re-sends the whole transcript, so
total input for `T` turns on a prefix `P` growing by `g` per turn is roughly
`T·P + g·T²/2` — quadratic in turns. Under that cost model an arm can win the
one-shot recall test and still lose the session.

`evaluation/agent_arm.py` measures three scripted policies over one tool surface,
one repository, and one success rule:

| arm | policy |
| --- | --- |
| `grep_loop` | grep the prompt's strongest identifier, read the best-matching file whole, follow its local imports and read those whole |
| `map_expand` | carry the task-independent repository map (`src/context_compiler/repo_map.py`) in the prefix, then request individual symbol bodies along call edges |
| `map_query` | the same graph, but behind a `find_symbol` lookup instead of in the prefix, then the same call-edge expansion |

Fairness rules, all enforced in `tests/test_agent_arm.py`:

- **The success rule is body-level.** `mock_requires` is a substring rule over
  definition headers, and the map emits headers by construction — scored that way
  the map arm gets 6/6 without paying for a line of logic (it did, on the first
  run). The rule used instead is: the transcript must contain the *source* of every
  labelled relevant symbol and every labelled test symbol. Both arms read the same
  files, so a body from `read_symbol` is a byte-exact substring of the same file
  from `read_file`, and neither is advantaged by how it obtained the text.
- **Cost is the sum of prefixes over turns**, not the size of the final window.
  `cached_input_tokens` re-prices the same transcript with every byte the provider
  has already seen served at a tenth of an input token. Caching keys on a
  byte-identical *prefix*, and turn `t`'s input opens with turn `t-1`'s input
  verbatim — so the accumulated transcript is as cacheable as a static header is.
  An earlier version of this model discounted only the static block, which gave the
  map arm a 9.5x discount and the two transcript-heavy arms almost none; that is a
  measurement of where an arm's tokens sit, not of what caching does. A static
  repository map is additionally credited as a cache *read* rather than a write from
  the second case in a run onward, since it is byte-identical across tasks by
  construction, and it is emitted ahead of the task-specific text so the prefix
  actually holds.
- **Both arms are told which test is failing**, because a real session starts from
  a failing test command. The `--no-test-hint` ablation withholds it from both.
- Means are over solved cases only, so an arm cannot look cheap by giving up.

### Result: the map's value is entry-point discovery, not evidence transport

120-module haystacks (~39k tokens each), six cases, `evaluation/results/agent_arms_haystack*.json`:

| failing test known | arm | solved | turns | input tokens, no cache | with prefix cache |
| --- | --- | --- | --- | --- | --- |
| yes | `grep_loop` | **6/6** | 3.0 | **428** | **233** |
| yes | `map_expand` | 6/6 | 3.5 | 44,287 | 15,884 |
| yes | `map_query` | 6/6 | 5.7 | 2,645 | 860 |
| no | `grep_loop` | 2/6 | 9.0 | 27,371 | 6,884 |
| no | `map_expand` | **6/6** | 3.5 | 44,274 | 15,879 |
| no | `map_query` | **6/6** | 4.7 | **1,840** | **687** |

Four findings, one of which contradicts the design this map was written to support:

1. **Given the failing test, retrieval is unnecessary on these tasks.** Reading the
   test and following its imports reaches the fix site in three turns and 428
   tokens. The haystack's 120 decoy modules never enter the transcript, because the
   agent never ranks anything. Repository size is a problem for a one-shot
   retriever that must rank; it is not, by itself, a problem for an agent that
   navigates structurally.
2. **Without the failing test, the grep loop collapses** — 2/6 at 9 turns and 27k
   tokens, burning four turns on whole decoy modules that out-match the prompt
   lexically. This is where structure pays, and it is the only place it pays.
3. **Carrying the map in the prefix is the wrong way to spend it.** `map_expand`
   pays ~12.6k tokens of skeleton on every turn to answer one question — where does
   this task start. `map_query` asks that question directly and solves the same 6/6
   for **24× fewer uncached tokens**, and caching does not rescue the difference:
   under prefix caching the gap is **18×** (860 against 15,884). Caching helps the
   cheap arm at least as much as the expensive one, because a small transcript is
   just as byte-identical turn to turn as a large map is.
4. **The regime flips on small repositories.** On the 6-file fixture the map is
   cheaper to carry than to query, because one lookup turn costs more than the
   whole skeleton. Both directions are pinned by tests.

The honest summary: the tier-one map earns its place as an *index behind a tool
call*, not as a prefix, and its measurable value is confined to the case where the
agent does not already know where to start. These are scripted policies with no
model in the loop, so this is evidence about tool economics — turns and tokens to
put the required evidence in front of a model — and says nothing about what a model
does once it has that evidence.

```bash
PYTHONPATH=src python3 evaluation/agent_arm.py --output evaluation/results/agent_arms_haystack.json
PYTHONPATH=src python3 evaluation/agent_arm.py --no-test-hint \
  --output evaluation/results/agent_arms_haystack_no_hint.json
```

## Tests

```bash
PYTHONPATH=src python3 -m unittest discover -s tests -v
```

## Research roadmap

The highest-value next steps are:

- Tree-sitter parsers for TypeScript, Java, Go, Rust, and C/C++;
- an optional dense embedding provider and reciprocal-rank fusion;
- dynamic call edges collected from test execution;
- Git change-coupling and issue-to-symbol edges;
- incremental file watching during an agent session;
- a learned context-value reranker fed by agent feedback (which retrieved files
  were opened, which were edited, which evidence went unused);
- budget-aware evidence selection: the `oracle`-to-`graphrag` gap in the budget
  sweep shows the compiler wastes budget on full-detail test files and duplicate
  regions, which is a larger loss than any remaining graph-reach gap;
- score-scale-aware fusion: on the haystack, adding lexical and vector evidence to
  the graph arm *lowers* success because decoy hits crowd the budget, so the
  hybrid arms need rank normalization (reciprocal-rank fusion) rather than a
  weighted sum of scores on incomparable scales;
- cache-friendly context assembly: stable prefix, context fingerprints, and
  incremental evidence deltas instead of a fresh prompt per turn;
- evidence-preserving conversation memory;
- SWE-bench Lite and RepoBench integration on repositories too large to fit in
  the context budget. `evaluation/haystack.py` reaches that regime synthetically
  and already reverses the `full_file` result, but its distractors are generated,
  so a real-repository replication is still required before publication;
- ablations for graph depth, edge families, budget, and detail levels.

## Research claim

A defensible hypothesis for this system is:

> Task-conditioned hybrid graph retrieval improves repository-level coding
> task success and relevant-symbol recall under a fixed context budget compared
> with lexical or vector retrieval alone.

The implementation is structured so that each retrieval component can be
disabled or replaced for controlled experiments.
