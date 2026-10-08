# How to Use Weavy

Weavy is a local, CLI-first memory system built on top of:

- FalkorDB for canonical records, semantic graph state, themes, and system counters
- LiteLLM for all model calls — completions (`LLM_MODEL`) and embeddings (`EMBEDDING_MODEL`)
- local FalkorDB `RunTrace` nodes for audit history, with optional Langfuse visibility

There is no web app or API layer. The main way to use the app is:

1. Start the backing services
2. Initialize the `System` node
3. Add text (from any source) into the memory graph
4. Query the graph, or continue any prior session

## What Works Right Now

- Add any text into the memory graph via `add` — source-agnostic ingestion
- Steer ingestion with optional caller context (e.g., "These are chat logs between a user and an AI assistant")
- Ingest text into the semantic graph with provenance-aware node writes
- Ask grounded questions against the graph
- Continue any prior session — ingestion or query — to ask follow-up questions with full context
- A single unified agent handles both ingestion and query, sharing the same tool set and message history model
- Automatic theme maintenance after every successful CLI or SDK `add`
- Persist agent runs locally and optionally track them in Langfuse
- Export and import complete local backups as single JSON files

## What To Expect

- Weavy is synchronous and explicit. If a dependency is missing, it fails loudly rather than silently degrading.
- The graph is a derived memory layer, not the source of truth. Canonical sessions remain the primary records.
- All sessions store their full message history. Ingestion sessions contain the input text + the agent's analysis. Query sessions contain the full conversation.
- Pre-processing is external. Document extraction, audio transcription, and chat formatting happen before text reaches Weavy.
- Any session can be continued in query mode — `continue s:N "question"` loads the prior messages as context and runs the query agent.
- Query runs may mutate the graph if the agent decides a new statement should update memory.
- Theme updates run automatically after every successful CLI or SDK `add`. The theme agent queries all sessions completed since its last run and reconciles themes. `update-themes` is still available for explicit invocation when needed.
- Run traces are persisted locally as `RunTrace` nodes in FalkorDB. Langfuse adds observability when configured, but it is not the only trace store.

## Prerequisites

You need:

- `uv` — manages Python and dependencies (`curl -LsSf https://astral.sh/uv/install.sh | sh`)
- Docker Desktop
- a Gemini API key (or any provider supported by LiteLLM — set `LLM_MODEL` and `EMBEDDING_MODEL` accordingly)
- a running Langfuse stack if you want tracing

At minimum, copy the example env file:

```bash
cp .example.env .env
```

Then set the values you need:

```env
LITELLM_API_KEY=...

LANGFUSE_HOST=http://localhost:3100
LANGFUSE_PUBLIC_KEY=...
LANGFUSE_SECRET_KEY=...
```

Notes:

- `LANGFUSE_*` is required for tracing. The agent runs without it but you lose observability.
- FalkorDB defaults to `localhost:6379`.
- The default graph name is `weavy`.

## Start The Services

### FalkorDB

The project expects FalkorDB to be available before you use the CLI:

```bash
docker compose --profile falkordb up -d
```

### Langfuse

Langfuse is used for run tracing. To avoid browser session collisions between the two local UIs, open them on different hostnames:

- FalkorDB UI: `http://127.0.0.1:3000`
- Langfuse UI: `http://localhost:3100`

Start the included local stack with:

```bash
docker compose --profile langfuse up -d
```

After Langfuse is up, create or copy your API keys into `.env`.

## Initialize The System Node

Run this once per graph before using the app:

```bash
uv run python -m weavy.cli init-system
```

This creates the singleton `System` node, including:

- next IDs for nodes, edges, and sessions
- `theme_priority_order`
- `hot_theme_token_budget`
- `last_theme_run_at`

Safe to run again — uses `MERGE`.

Inspect current state any time:

```bash
uv run python -m weavy.cli status
```

If the `System` node does not exist, most operations will fail with a message telling you to run `init-system` first.

## Add Text To The Memory Graph

The primary entry point is `add`. It accepts any text — transcripts, chat logs, notes, journal entries, documents — and runs the ingestion agent to integrate it into the semantic graph.

```bash
uv run python -m weavy.cli add /path/to/notes.txt
```

Or pipe from stdin:

```bash
echo "Today I decided to switch to Rust for the backend." | uv run python -m weavy.cli add -
```

What happens:

1. A canonical `Session` is created with the text as the first user message
2. The ingestion agent reads the text, searches the existing graph, and creates/updates semantic nodes and edges
3. The CLI prints the session status, touched node IDs, and the ingestion summary
4. The theme agent runs automatically to reconcile themes against newly completed sessions

### Caller Context

Use `--context` to steer the agent's interpretation without modifying the prompt:

```bash
uv run python -m weavy.cli add chat-export.txt --context "These are chat sessions between a user and an AI assistant"
```

The context is injected into the system prompt as `- **Caller context:** <value>`. When omitted, the slot resolves to empty string. The agent reads the text and determines what it is on its own.

### Timestamps

Use `--timestamp` to set the session timestamp (defaults to now):

```bash
uv run python -m weavy.cli add journal.txt --timestamp 2026-04-01T10:00:00+00:00
```

### List Stored Sessions

```bash
uv run python -m weavy.cli list-sessions
uv run python -m weavy.cli list-sessions --limit 5
```

Output includes session ID, timestamp, and summary (if ingested):


## Continue Any Session

After adding content, you can continue that session to ask questions with the full text + agent analysis as context:

```bash
uv run python -m weavy.cli continue s:1 "What did I say about my mortgage?"
```

This always runs in query mode — the query agent is given the session's existing message history plus your new question. It can search the graph and answer based on both the stored graph and the conversation context.

This works on any session — ingestion or query — making every conversation resumable.

## Query The Graph

Run a one-shot query (creates a new session):

```bash
uv run python -m weavy.cli query "What have I been thinking about recently?"
```

What query mode does:

1. Mints a new session
2. Builds the system prompt from `weavy-query.md` plus hot themes
3. Lets the agent search the graph, inspect neighborhoods, and retrieve source evidence
4. Finalizes with `complete`
5. Persists the full conversation as a canonical `Session`

What the CLI prints:

- `Status: completed` or `Status: failed`
- the final answer if successful

### Interactive Chat Mode

Omit the question argument to start a REPL:

```bash
uv run python -m weavy.cli query
```

```text
Weavy chat — type 'exit' or Ctrl-D to quit.
```

Behavior to expect:

- one session (`s:N`) is created for the full REPL session
- each turn calls the query agent with the full accumulated history, loaded from FalkorDB
- history is persisted after each turn so the session is crash-safe
- if a turn fails, the REPL prints the error and continues

## Themes

Theme maintenance runs automatically after every successful CLI or SDK `add` call.
You can also invoke it manually:

```bash
uv run python -m weavy.cli update-themes
```

Lower-level application calls such as `run_add(...)` do not run themes automatically; callers that use `application/` own theme timing explicitly.

The theme agent is self-discovering. It queries all sessions completed since its last run (tracked via `last_theme_run_at` on the System node) and builds a journal of recent activity.

What the theme agent receives:

- the current full theme map
- a session journal listing each completed session's summary and graph changes

What to expect:

- new themes may be created
- existing themes may be updated or retired
- `theme_priority_order` on the `System` node may change
- `last_theme_run_at` is updated after a successful run

## Tracing And Langfuse

Each harness run persists a complete `RunTrace` node in FalkorDB, including turns,
tool calls, token usage, completion payload, touched graph IDs, status, and error.

When Langfuse is configured, each harness run also creates a Langfuse trace with nested spans for:

- the run root
- each turn
- each LLM generation
- each tool call

Use FalkorDB for the durable local audit trail. Use the Langfuse UI when you want richer run visualization.

## Backup, Export, And Import

Export writes one complete JSON backup containing the System node, canonical
sessions, semantic nodes, relationships, themes, and local run traces:

```bash
uv run python -m weavy.cli export backup.json
```

Import restores that backup into the current graph. It refuses to touch a
non-empty graph unless `--replace` is passed:

```bash
uv run python -m weavy.cli import backup.json --replace
```

Backups are plain JSON by design. If you need encrypted storage, encrypt the
file externally with your normal tooling.

## Evaluate Memory Quality

`weavy eval` measures how well the memory pipeline answers questions over a
known dataset. It ingests each conversation into its own isolated graph
(`<graph-prefix>_<dataset>_<conversation>`, reset first), asks every question at
a fixed `query_time`, and scores the resulting `RunTrace`s. It refuses to run if
an eval graph name equals your configured `GRAPH_NAME`.

```bash
uv run weavy eval                                   # built-in smoke dataset, keyword judge
uv run weavy eval --judge llm --judge-model openai/gpt-4o-mini
uv run weavy eval --dataset my_dataset.json --output results.json --cleanup
uv run weavy eval --conversation journal --limit 3 --no-themes
uv run weavy eval --list-datasets
```

The built-in `smoke` dataset is synthetic and hand-written (2 conversations,
7 sessions, 14 questions). It exists to exercise the pipeline end to end; it is
**not** a benchmark and its scores are not comparable with LoCoMo/LongMemEval.

Each run writes a JSON report (config incl. models and dataset sha256,
per-question rows, aggregates) and prints a summary. Metrics:

| Metric | Definition |
|---|---|
| accuracy | judge verdict; abstention questions are correct only when the answer says there is no record |
| session recall | gold evidence sessions that surfaced in read-tool results (`source_id`, `mentioned_by`, `get_session`) / gold sessions |
| session precision | gold sessions among those opened with `get_session` / sessions opened (n/a if none) |
| fact recall | evidence keywords present in read-tool results / evidence keywords |
| citation precision | cited sessions that are gold evidence / cited sessions |
| citation support | cited sessions that actually surfaced during the run / cited sessions |
| latency, tool calls, tokens | per query, from wall clock and the RunTrace |
| estimated cost | LLM completions only, from litellm's price table; reported as unknown (not $0) for unpriced models. Embedding calls are not included. |

### Retrieval-only evaluation (no LLM)

`--mode retrieval` scores the search primitive itself. It builds a fixture
graph directly (no ingestion agent), runs each query through the same
`search_graph` the agent calls, and reports recall@k, precision@k, MRR,
related recall (neighbours via graph expansion), source recall (provenance
exposed on results), and empty-correct (unrelated queries return nothing).

```bash
# Offline: deterministic bag-of-words hash embeddings, no API key needed.
uv run weavy eval --mode retrieval --embedding-model weavy/hash-512 --cleanup
# Calibrate the vector distance cut-off for your embedding model:
uv run weavy eval --mode retrieval --max-distance 0.5 --cleanup
uv run weavy eval --mode retrieval --max-distance none --cleanup
```

The hash embedder is **not semantic** — similarity is shared word tokens — so
offline runs measure retrieval mechanics (time filtering, thresholds,
provenance, expansion), not semantic quality. Use your real
`EMBEDDING_MODEL` for that. `retrieval-smoke` is a small synthetic fixture,
not a benchmark.

Judges: `keyword` (default) is deterministic — every `answer_keywords` group must
appear (alternatives separated by `|`). It can mark paraphrased correct answers
wrong. `llm` grades semantically but is non-deterministic and costs tokens
(reported separately as judge usage).

Dataset format (`weavy-eval/v1`) — see the docstring in `weavy/eval/dataset.py`.
Session ids in a dataset are local; the runner maps them to the `s:N` ids Weavy
mints during ingestion. Timestamps must carry a timezone offset. Other formats
(LoCoMo, LongMemEval) plug in by registering a converter in
`weavy.eval.dataset.DATASET_LOADERS`.

## Retrieval Settings

`search_graph` behaviour is configured through environment variables (see
`.example.env`; defaults shown):

| Setting | Default | Effect |
|---|---|---|
| `SEARCH_MAX_DISTANCE` | `0.6` | Vector hits with cosine distance above this are dropped; `none` disables. Uncalibrated, model-dependent default — calibrate with `weavy eval --mode retrieval`. |
| `SEARCH_CANDIDATE_POOL` | `50` | Candidates fetched per pass before time filtering and the top-k cut. |
| `SEARCH_EXPAND_DEPTH` | `1` | Hops of graph-neighbour expansion from direct hits (`0` = off). |
| `SEARCH_EXPAND_LIMIT` | `5` | Max neighbour rows added per search (marked with `via`). |
| `SEARCH_MAX_SOURCES` | `5` | Source sessions listed per result (total in `source_count`). |
| `NEIGHBORHOOD_LIMIT` | `25` | Default cap for `get_node_neighborhood`. |

Time ranges (`search_graph.time_range`, `list_sessions.date_range`) accept a
null bound for open-ended windows, and are applied before the top-k cut. Naive
datetimes are interpreted as UTC everywhere.

Query answers carry `citation_check` in the completion payload; the CLI prints
a warning for cited sessions or nodes the run never retrieved.

### Prompts

Agent prompts live in `weavy/prompts/` as local markdown files:

- `weavy-ingestion.md` — ingestion agent prompt (source-agnostic, uses `{{caller_context}}` for steering)
- `weavy-query.md` — query/chat agent prompt
- `weavy-theme.md` — theme maintenance prompt

Template variables (e.g. `{{session_id}}`, `{{themes_context}}`, `{{caller_context}}`) are replaced at runtime.

## Inspect The Database

FalkorDB is available on `localhost:6379`. Useful Cypher queries:

```cypher
MATCH (n:SemanticNode)
RETURN n.id, n.aliases, n.summary
LIMIT 20
```

```cypher
MATCH (t:Theme)
RETURN t.name, t.state, t.status
```

```cypher
MATCH (s:Session)
RETURN s.id, s.timestamp, s.completed_at, s.summary
ORDER BY s.timestamp DESC
```

```cypher
MATCH (s:System) RETURN s
```

## Full CLI Reference

All commands run through:

```bash
uv run python -m weavy.cli <command>
```

Commands:

```text
init-system
status

list-sessions [--limit N]

add <file_or_-> [--context "..."] [--timestamp ISO]
continue <session_id> <question>
update-themes
export <path>
import <path> --replace
set-preface <preface>
query [question]
eval [--dataset NAME|PATH] [--format F] [--judge keyword|llm] [--judge-model M]
     [--output PATH] [--limit N] [--conversation ID ...] [--graph-prefix P]
     [--no-themes] [--cleanup] [--verbose] [--list-datasets]
```

## Common Failure Modes

### `System node not found`

```bash
uv run python -m weavy.cli init-system
```

### Tracing failures

Check that Langfuse is running and `LANGFUSE_HOST`, `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY` are set.

### LLM or query failures

Check that your API key (e.g. `LITELLM_API_KEY`) is set and the graph is reachable.

## Shortest Happy Path

```bash
cp .example.env .env
# fill in LITELLM_API_KEY and LANGFUSE_* as needed

docker compose --profile falkordb --profile langfuse up -d

uv run python -m weavy.cli init-system
uv run python -m weavy.cli add notes.txt --context "Personal journal entry"
uv run python -m weavy.cli export backup.json
uv run python -m weavy.cli query "What has been on my mind recently?"
uv run python -m weavy.cli continue s:1 "Tell me more about what I said about X"
```
