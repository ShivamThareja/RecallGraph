## Context

Weavy's stated architecture (see `CLAUDE.md`) is "the semantic graph is a lossy index; episodes are ground truth." The implementation does not honor this. `search_graph` runs six queries — vector and keyword over nodes, edges, **and** `Chunk` excerpts — and merges them into one flat cosine-ranked list in which a raw chunk can outrank a distilled node. `weavy-query.md` then instructs the agent to "trust episodes over summaries." In effect the index (graph) and the ground truth (episodes) sit on the same shelf competing for rank, and the agent is told to bypass the index when they disagree.

The chunk layer was almost certainly added as a recall safety net for a lossy extractor: when ingestion compresses away a specific, chunk-RAG catches it. But that net has a cost — it lets ingestion be lazy invisibly, and it forces a pile of reconciliation machinery to exist: a second per-node vector (`identity_embedding`) plus its index, note-history stuffed into the content vector, and char/token budget constants (`_CHUNK_TARGET_CHARS`, `_CHARS_PER_TOKEN_ESTIMATE`, `get_char_budget`). Every one of those is a symptom of running two memory systems at once.

This change commits to one system.

## Goals / Non-Goals

**Goals:**
- Make the semantic graph (nodes + edges) the **sole** search surface.
- Reach ground-truth episodes **only** by navigation: `search_graph` → `get_node.mentioned_by` / edge provenance → `get_session`.
- Collapse the dual embedding into a single `embedding` (aliases + summary) with one vector index.
- Delete the `Chunk` layer and all reconciliation machinery it required.
- Reframe the ingestion mandate so coverage + linking + source-pointing is explicit and non-optional.
- End with strictly less code and fewer tunable constants than before.

**Non-Goals:**
- No data migration for existing graphs. Dev/benchmark graphs are disposable (`reset()` + re-ingest).
- No within-episode locator (sub-episode chunking). In Model B, `get_session` returns the whole episode and the agent reads it. A future locator is out of scope and, if ever added, must not become a competing search surface.
- No demoted-fallback tier ("Model C"). This is deliberately excluded — see Decisions.
- No change to the two-clock temporal model, the `MENTIONS` linking, or the theme subsystem.
- Not optimizing for detail-recall benchmark scores; short-term regression is accepted.

## Decisions

### Decision 1: Model B (navigational), not Model A (flat union) or Model C (demoted fallback)

Three architectures were considered for how the graph and raw episodes relate at retrieval time:

- **Model A — flat union (current).** Search returns nodes, edges, and chunks in one distance-ranked list; the agent treats them as peers and the prompt says trust the chunk. *Rejected:* index and ground truth compete, so the graph is structurally second-class. This is the status quo being removed.
- **Model B — navigational (chosen).** Search returns only the index (nodes + edges). Each result points to its source episodes. The agent finds nodes, follows edges, then pulls episodes for specifics. Raw text is reached *through* the graph. *Chosen:* it is the only model where the graph is unambiguously the memory. It also cascades into deleting the dual embedding, note-stuffing, and the budget constants — a decision that removes machinery is the correct cut.
- **Model C — demoted fallback.** Graph is primary; a raw episode/chunk search runs as an explicit second tier that can never outrank a graph hit. *Rejected here* despite being the pragmatic choice: it keeps the safety net, which keeps ingestion laziness invisible and keeps a parallel search surface alive. The project goal is purity — general graph memory taken as far as it goes — so the net is removed on purpose.

**The accepted bet, stated plainly:** retrieval quality = index quality. A fact whose entity was never indexed, or was indexed but left unlinked, is unanswerable — even though its episode text still exists in storage. There is no graceful degradation. This is accepted because it makes ingestion quality *observable and non-optional* rather than something a RAG net silently papers over.

### Decision 2: Lossy summaries are fine; unreachable or unlinked entities are fatal

Model B reconciles "the index is lossy" with "no fact left behind" through navigation. The node summary is allowed to drop specifics, because the specifics live in the episode and are recovered by `get_session`. What is *not* allowed is for an answer-bearing entity to be absent, isolated, or missing its `mentioned_by` link home. This reframes ingestion's job:

- **Before (implicit):** extract every fact into the summary.
- **After (this change):** guarantee every answer-bearing entity is a reachable, connected node that points to its source episode.

The retrieval path this enables:

```
question
  → search_graph            (land on some relevant node; summary may be lossy)
  → get_node_neighborhood   (follow edges — multi-hop)
  → get_node.mentioned_by   (which episodes touched this entity)
  → get_session             (read verbatim ground truth for the specifics)
```

`aliases` therefore become the primary entry signal (the first foothold before any traversal), and edge density becomes the primary traversability signal. Both are promoted from "advice" in the prompt to load-bearing quality metrics.

### Decision 3: Collapse the dual embedding into one vector

`identity_embedding` (aliases + summary) existed so dedup could compare a fresh candidate against a stable identity representation, while `embedding` accumulated note history "so past facts stay retrievable." In Model B, past facts are recovered by navigating to episodes, so note-stuffing has no remaining justification. Remove it, and `embedding` becomes aliases + summary only — byte-for-byte the same representation `identity_embedding` held. The two vectors and their two indexes therefore collapse into one `embedding` + one index. `DUPLICATE_DISTANCE` dedup now queries that single vector; the comparison stays apples-to-apples because neither the candidate nor the stored vector carries notes anymore. `get_char_budget` and its constants die with note-stuffing.

### Decision 4: Keep episodes as stored, unembedded ground truth

Sessions continue to store raw text and remain retrievable by id via `get_session`. What changes is only that episodes are no longer independently embedded or surfaced as search results — they are navigation *targets*, not a search *surface*. `Chunk` nodes, `create_chunks`, `index_episode`, and the Chunk vector index are removed; `create_session` no longer indexes episode text.

## Risks / Trade-offs

- **Recall cliff, not a slope.** A missed or unlinked entity is permanently unanswerable until re-ingestion. → Mitigation: strengthen the ingestion mandate (Decision 2) and treat coverage/linking as first-class; accept that closing the gap is ingestion's job, not retrieval's.
- **Short-term benchmark regression** on detail recall that previously rode chunk-RAG. → Mitigation: explicitly out of scope to optimize; this is a known, accepted consequence of choosing purity over scores.
- **More tool-calls and tokens per query** (navigate-to-episode is chattier than flat search). → Mitigation: accepted cost of the model; neighborhood traversal already exists, so no new mechanism is needed.
- **Whole-episode reads** can be large (no sub-episode locator). → Mitigation: accepted for correctness; a future locator may be considered but must never become a competing search surface (Non-Goals).
- **Existing graphs carry dead `Chunk` nodes and `identity_embedding` props.** → Mitigation: no migration; `reset()` + re-ingest produces the clean shape (dev/benchmark graphs are disposable).

## Migration Plan

No runtime migration. Steps:
1. Land code + prompt changes behind the single change set (deletion-heavy, no feature flag).
2. For any active dev/benchmark graph, run `Weavy.reset()` (or `cli init-system` on a fresh graph) and re-ingest — this drops old `Chunk` / `identity_embedding` artifacts and rebuilds with a single embedding.
3. Rollback is `git revert` of the change; old graphs are unaffected by the revert because the removed structures were additive.

## Open Questions

- **Alias-coverage enforcement strength.** Decision 2 makes aliases the primary entry point. This change strengthens the *prompt* mandate; whether to add a structural check (e.g., ingestion warns when a created node has a single terse alias) is deferred — flag for a follow-up if entry-point recall proves to be the dominant failure mode.
- **Keyword search retention.** Keyword matching over node aliases/summaries and edge label/fact stays (it is part of the graph surface, not the chunk surface). Confirm during implementation that removing the chunk branches leaves the node/edge keyword paths intact.
