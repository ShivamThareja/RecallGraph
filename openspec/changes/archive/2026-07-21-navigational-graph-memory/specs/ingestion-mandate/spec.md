## ADDED Requirements

### Requirement: Every answer-bearing entity must be reachable

Because there is no retrieval fallback outside the graph, ingestion SHALL ensure that every answer-bearing entity present in the input becomes a semantic node that is reachable by search — with aliases covering the names, abbreviations, and synonyms by which the entity may later be referenced.

#### Scenario: Entity coverage is required

- **WHEN** the ingestion agent processes input naming a durable entity
- **THEN** it creates or updates a node for that entity rather than leaving the fact only in the episode text
- **AND** the node's aliases include the forms by which the entity is likely to be queried

### Requirement: Nodes must be linked, not isolated

Ingestion SHALL connect entities with edges so that facts are reachable by traversal. An isolated node (an entity with no relationship edges) is a retrieval dead end and SHALL be treated as an ingestion failure to be corrected, not an acceptable outcome.

#### Scenario: New entities are wired into the graph

- **WHEN** ingestion creates nodes for related entities described in the input
- **THEN** it creates edges expressing the stated relationships between them
- **AND** it does not leave a newly created answer-bearing entity without any edges when the input supports a connection

### Requirement: Nodes must point to their source episode

Every node write SHALL record provenance linking the node to its source session (`mentioned_by`), so that the verbatim episode can be reached by navigation when the summary is too lossy to answer a question.

#### Scenario: Provenance links node to episode

- **WHEN** ingestion creates or updates a node
- **THEN** the touching session is recorded such that the node's `mentioned_by` includes that `s:N` id
- **AND** an agent can navigate from the node to `get_session` for the verbatim source

### Requirement: Summaries may be lossy but coverage may not

A node summary MAY omit specifics that remain recoverable from the source episode. Ingestion SHALL NOT treat a lossy summary as a failure, but SHALL treat a missing entity, an unlinked entity, or a missing source link as a failure.

#### Scenario: Lossy summary with recoverable detail is acceptable

- **WHEN** a node summary omits a specific detail that appears in the source episode
- **THEN** this is acceptable provided the node is reachable, linked, and points to the episode holding the detail
