# entity-storage

## Purpose

TBD — synced from `navigational-graph-memory`. Covers node embedding storage and duplicate-detection guarantees for semantic entities.

## Requirements

### Requirement: A node carries exactly one embedding

A semantic node SHALL store a single `embedding` computed from its aliases and current summary only. The system SHALL NOT store a separate `identity_embedding`, and SHALL NOT fold log-note history into the embedding. Exactly one node vector index SHALL exist.

#### Scenario: Node embedding is aliases plus summary

- **WHEN** a node is created or its aliases/summary are updated
- **THEN** its `embedding` is computed from aliases + current summary only
- **AND** no log-note text is included in the embedded content

#### Scenario: Only one node vector index exists

- **WHEN** the system initializes vector indexes
- **THEN** it creates a single vector index on `SemanticNode.embedding`
- **AND** it creates no `identity_embedding` index

#### Scenario: Updating a summary re-embeds over identity fields only

- **WHEN** a node's summary is updated
- **THEN** the new `embedding` reflects the new aliases + summary
- **AND** the embedding does not grow with the number of prior log entries

### Requirement: Duplicate detection uses the single embedding

The one-node-per-entity invariant SHALL be enforced by comparing a prospective node's aliases + summary embedding against existing nodes' `embedding` vectors, refusing creation when a match is within `DUPLICATE_DISTANCE` or shares an alias, unless `force=true`. The comparison SHALL be apples-to-apples because neither the candidate nor the stored vector carries note history.

#### Scenario: Refusing a likely duplicate

- **WHEN** `create_node` is called for an entity whose embedding is within `DUPLICATE_DISTANCE` of an existing node, or shares an alias
- **THEN** creation is refused with `ok=false` and the candidate node(s) are named
- **AND** the caller is directed to update the existing node or retry with `force=true`

#### Scenario: Forced creation of a genuinely distinct entity

- **WHEN** `create_node` is called with `force=true`
- **THEN** the node is created even if an existing node is within `DUPLICATE_DISTANCE` or shares an alias
