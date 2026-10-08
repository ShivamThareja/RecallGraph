## 1. Remove the Chunk layer

- [x] 1.1 Delete `chunk_text`, `index_episode`, and `_CHUNK_TARGET_CHARS` from `weavy/services/memory.py`
- [x] 1.2 Delete `create_chunks` from `weavy/store/canonical.py` and remove the private cross-module `_vecf32_literal` import it used
- [x] 1.3 Remove the `index_episode` call from `create_session` in `weavy/application/session_runs.py`
- [x] 1.4 Remove the `Chunk` vector index from `_ensure_vector_index` in `weavy/store/system.py`

## 2. Make the graph the sole search surface

- [x] 2.1 Remove the `chunk_vec` and `chunk_kw` query branches and `_episode_row` from `search_graph` in `weavy/store/graph.py`; keep the node and edge vector + keyword passes
- [x] 2.2 Remove the episode-filtering branch from `_filter_by_time_range` so it filters nodes and edges only
- [x] 2.3 Remove `kind="episode"` from `SearchResult` in `weavy/application/contracts.py` and adjust any typing/validation to allow only `"node"` and `"edge"`
- [x] 2.4 Confirm `search_graph`'s dedup and `(score is None, ...)` ranking still behave correctly with only node/edge results

## 3. Collapse the dual embedding into one vector

- [x] 3.1 Simplify `embed_node` in `weavy/services/embedding.py` to embed aliases + summary only; drop the `notes` parameter and note-packing logic
- [x] 3.2 Delete `get_char_budget`, `_CHARS_PER_TOKEN_ESTIMATE`, and `get_max_input_tokens` if now unused
- [x] 3.3 Remove `identity_embedding` from `create_node`/`update_node` in `weavy/store/graph.py` (write only `embedding`)
- [x] 3.4 Remove the `identity_embedding` vector index from `_ensure_vector_index` in `weavy/store/system.py`
- [x] 3.5 Update `find_similar_nodes` to query the single `embedding` index; update `update_node` in `weavy/services/memory.py` to stop computing an identity vector
- [x] 3.6 Verify `DUPLICATE_DISTANCE` dedup in `create_node` still compares candidate aliases+summary against the single `embedding` vector

## 4. Rewrite the prompts

- [x] 4.1 In `weavy/prompts/weavy-query.md`, remove "trust episodes over summaries"; add the search → traverse (`get_node_neighborhood`) → `mentioned_by` → `get_session` navigational protocol; describe episodes as navigation targets, not search results
- [x] 4.2 In `weavy/prompts/weavy-ingestion.md`, reframe the mandate around coverage + linking + source-pointing; state that isolated nodes and missing entities are fatal (no retrieval fallback); emphasize alias coverage as the primary entry signal
- [x] 4.3 Remove any remaining references to episode/chunk search results from both prompts and from action/tool descriptions in `weavy/harness/actions.py` (e.g., the `search_graph` description mentioning `kind="episode"`)

## 5. Tests and docs

- [x] 5.1 Remove or rewrite chunk/episode-search tests; assert `search_graph` returns only node/edge kinds
- [x] 5.2 Add a test that ground truth is reachable via navigation: `get_node.mentioned_by` → `get_session` returns the source episode text
- [x] 5.3 Add a test asserting a single node vector index and that `create_node`/`update_node` write only `embedding` (no `identity_embedding`)
- [x] 5.4 Add/adjust a dedup test proving `DUPLICATE_DISTANCE` refusal works against the single embedding
- [x] 5.5 Update `CLAUDE.md` (Key Design Decisions, Module Map, Episodes/dedup bullets) to describe navigational Model B, single embedding, and episodes-as-navigation-targets
- [x] 5.6 Run `uv run pytest`, `uv run ruff check .`, and `uv run ruff format .`; fix fallout

## 6. Verify the accepted trade-off end to end

- [x] 6.1 On a fresh graph, ingest a small multi-entity corpus and confirm queries answer by navigating graph → episode (no chunk search involved)
- [x] 6.2 Confirm the removed structures are gone at runtime (no `Chunk` nodes created, no `identity_embedding` property written) via a fresh `reset()` + ingest
