# prompt-guidance

## Purpose

TBD — synced from `remove-prompt-magic-numbers`. Covers how agent prompts express judgment-based behavior (qualitative intent vs. unenforced numeric thresholds) and preserves genuine reasoning guidance.

## Requirements

### Requirement: Agent-judgment guidance is expressed qualitatively, not as unenforced counts

Agent prompts SHALL express behavior that is left to the agent's judgment as qualitative intent, not as a numeric threshold that no code enforces. A prompt MUST NOT instruct the agent to perform an action a specific number of times, or gate an action on a specific count of evidence, unless that count is checked in code. When the intent is "do this thoroughly / when it recurs", the prompt states that intent in words rather than attaching a number to it.

#### Scenario: Search-breadth guidance carries no count

- **WHEN** the ingestion prompt (`weavy/prompts/weavy-ingestion.md`) or the query prompt (`weavy/prompts/weavy-query.md`) instructs the agent to search with varied phrasings
- **THEN** the instruction names the qualitative goal (different phrasings surface different regions of the graph)
- **AND** it contains no numeric count of phrasings or searches (no "3+", no "multiple" used as a required quantity)

#### Scenario: Theme-creation guidance carries no session count

- **WHEN** the theme prompt (`weavy/prompts/weavy-theme.md`) describes when to create a theme
- **THEN** it expresses the judgment that a theme is a recurring thread rather than a one-off
- **AND** it does not require a specific number of distinct sessions (no "at least two distinct sessions")

#### Scenario: A number in a prompt implies code enforcement

- **WHEN** a reviewer finds a numeric threshold in any agent prompt
- **THEN** there is corresponding code that enforces that threshold
- **OR** the number is removed and replaced by the qualitative intent

### Requirement: Genuine reasoning tasks remain prose

Prompt guidance that requires the agent to reason over ambiguous input — as opposed to following an unenforced count — is legitimately expressed in prose and is out of scope for numeric-threshold removal. Removing false-precision counts SHALL NOT strip instructions or worked examples that teach genuine reasoning.

#### Scenario: Temporal resolution guidance is preserved

- **WHEN** the ingestion prompt guides the agent to resolve a relative time expression (e.g. "yesterday", "last summer") to an absolute `happened_at`
- **THEN** that guidance, including its worked example, remains present and unchanged
- **AND** it is not treated as an unenforced numeric threshold, because it is reasoning over language rather than a count
