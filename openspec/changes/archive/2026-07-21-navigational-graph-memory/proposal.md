## Why

Weavy was intended to be graph memory, but the current retrieval path does not commit to it. `search_graph` embeds semantic nodes, semantic edges, **and** raw `Chunk` excerpts into one flat cosine-ranked list where a chunk can outrank a node, and `weavy-query.md` instructs the agent to "trust episodes over summaries." The result is two competing memory systems — a distilled graph and chunk-RAG over raw text — coexisting because neither was committed to. That non-commitment is the source of a cascade of inelegance: a redundant second embedding (`identity_embedding`), note-history stuffed into the content vector "so past facts stay retrievable," and a set of char/token budget constants that only exist to reconcile the two systems at their seam.

This change commits fully to **navigational graph memory ("Model B")**: the semantic graph is the sole search surface, and episodes are ground truth reached *through* the graph by navigation — never as a parallel search surface.

## What Changes

- **BREAKING** — `search_graph` returns only `kind="node"` and `kind="edge"`. The `episode` result kind is removed. Ground truth is reached exclusively by navigation: `search_graph` → `get_node.mentioned_by` / edge provenance → `get_session`.
- **BREAKING** — The `Chunk` node type and its vector index are removed, along with `chunk_text`, `index_episode`, `create_chunks`, the `index_episode` call in `create_session`, the chunk/episode branches in `search_graph` and `_filter_by_time_range`, `get_char_budget`, and the `_CHUNK_TARGET_CHARS` / `_CHARS_PER_TOKEN_ESTIMATE` constants.
- **BREAKING** — The dual embedding collapses into one. `embed_node` embeds aliases + summary only (note-history stuffing removed), which makes the content vector identical to the former identity vector. `identity_embedding` and its separate vector index are removed; `DUPLICATE_DISTANCE` dedup compares against the single `embedding` vector.
- **Prompts rewritten.** `weavy-query.md` drops "trust episodes over summaries" and gains an explicit search → traverse → `get_session` navigational protocol. `weavy-ingestion.md` reframes the ingestion mandate around **coverage + linking + source-pointing**: every answer-bearing entity must be a reachable, connected node that points to its source episode; isolated nodes and missing entities are explicitly fatal.
- **Accepted bet, made explicit:** retrieval quality = index quality. There is no RAG safety net by design — a fact whose entity was never indexed (or was left unlinked) is unanswerable even though its episode text still exists. The full quality burden moves onto ingestion.
- **Unchanged:** Sessions still store raw episode text as ground truth; `get_session`, the `MENTIONS` linking (`mentioned_by`), the two clocks (`timestamp` / `happened_at`), and the theme subsystem all remain.

## Capabilities

### New Capabilities
- `graph-retrieval`: The retrieval contract — the semantic graph (nodes + edges) is the sole search surface, and ground-truth episodes are reachable only by navigation from graph results, not by independent search.
- `entity-storage`: Single-vector node storage and the one-node-per-entity dedup invariant — one `embedding` (aliases + summary) and one vector index, compared against `DUPLICATE_DISTANCE`.
- `ingestion-mandate`: The ingestion quality contract — every answer-bearing entity must be a reachable, connected, source-pointing node; coverage and linking are non-optional because there is no retrieval fallback.

### Modified Capabilities
<!-- None — no pre-existing specs in openspec/specs/. -->

## Impact

- **Code removed/changed:** `weavy/services/memory.py` (chunking, dedup vector), `weavy/services/embedding.py` (note-stuffing, char budget), `weavy/store/graph.py` (`search_graph`, `_filter_by_time_range`, `find_similar_nodes`, `_episode_row`), `weavy/store/canonical.py` (`create_chunks`), `weavy/store/system.py` (`_ensure_vector_index` loses Chunk + identity indexes), `weavy/application/session_runs.py` (`create_session`), `weavy/application/contracts.py` (`SearchResult.kind`), `weavy/prompts/weavy-query.md`, `weavy/prompts/weavy-ingestion.md`.
- **Data/schema:** existing graphs carry `Chunk` nodes and `identity_embedding` properties that become dead; `reset()` / re-ingestion produces the new shape. No migration is provided (benchmark/dev graphs are disposable).
- **Behavior:** detail-recall that previously leaned on chunk-RAG now depends entirely on ingestion coverage; short-term recall benchmarks may regress while ingestion quality matures. This is accepted — the goal is general graph memory, not benchmark scores.
- **Tests:** chunk/episode search tests removed; new tests assert the navigational contract (search returns only nodes/edges; episodes reachable via `mentioned_by` → `get_session`) and single-vector dedup.
