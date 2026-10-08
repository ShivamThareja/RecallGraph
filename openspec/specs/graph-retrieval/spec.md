# graph-retrieval

## Purpose

TBD — synced from `navigational-graph-memory`. Covers `search_graph`'s result contract, episode reachability via navigation, and time-range filtering.

## Requirements

### Requirement: Graph is the sole search surface

`search_graph` SHALL return only semantic entities (`kind="node"`) and relationship facts (`kind="edge"`). It SHALL NOT return raw episode excerpts as an independent search result kind. The `episode` result kind is removed from the retrieval contract.

#### Scenario: Search returns only nodes and edges

- **WHEN** `search_graph` is called with any query against a populated graph
- **THEN** every result has `kind` of either `"node"` or `"edge"`
- **AND** no result has `kind="episode"` or otherwise represents a raw text excerpt

#### Scenario: No chunk index is consulted

- **WHEN** `search_graph` runs its vector and keyword passes
- **THEN** it queries only `SemanticNode` and `RELATES` (node and edge) indexes
- **AND** it does not query any `Chunk` vector or keyword index

### Requirement: Episodes are reachable only by navigation

Ground-truth episodes SHALL be reachable exclusively by navigating from graph results — via `get_node`'s `mentioned_by` session ids or an edge's provenance `source_id` — to `get_session`. Episode text SHALL NOT be embedded or independently searchable.

#### Scenario: Reaching source text from a node

- **WHEN** an agent has a node id and needs the verbatim source
- **THEN** it reads the node's `mentioned_by` list to obtain `s:N` session ids
- **AND** calls `get_session` on one of those ids to read the episode's raw text

#### Scenario: Episode text is not indexed at ingestion

- **WHEN** a new session is created with episode text
- **THEN** no embedded chunk representation of that text is created
- **AND** the episode is retrievable only by its session id, not by vector or keyword search

### Requirement: Time-range filtering operates over graph results only

`search_graph`'s optional `time_range` filter SHALL apply to node and edge results by their log `happened_at` (falling back to record time). It SHALL NOT include an episode-filtering branch.

#### Scenario: Time-ranged search filters nodes and edges

- **WHEN** `search_graph` is called with a `time_range`
- **THEN** only nodes and edges with at least one log entry inside the range are returned
- **AND** the filter performs no episode/session-timestamp filtering pass
