## Why

Three agent prompts encode behavioral policy as **unenforced numeric thresholds** — "search with **3+ different phrasings**" (ingestion), "search with **multiple phrasings**" (query), and theme creation "require evidence from **at least two distinct sessions**". No code checks any of these counts, so they neither constrain the agent nor inform it — they are false precision that reads as a rule while behaving as decoration. An A/B experiment (gpt-5-mini, LoCoMo, one conversation, 25 questions) confirmed the search-breadth number is inert: the agent averaged 1.44 searches per question whether or not it was told "multiple," and forcing breadth structurally (tool-side query expansion, "Option B") produced no retrieval-quality gain (9/25 vs 10/25 — the single flip was decided by a `get_session` navigation step, not by search breadth) while costing 2.2× latency and adding a new failure mode (noisy auto-expansions). The failures that dominate are ingestion coverage and reasoning/abstention, not search phrasing.

## What Changes

- Rewrite the search-breadth instruction in `weavy/prompts/weavy-ingestion.md` and `weavy/prompts/weavy-query.md` to drop the "3+" / "multiple" counts and state the qualitative intent instead ("search with varied phrasings — different phrasings surface different regions of the graph").
- Rewrite the theme-creation guidance in `weavy/prompts/weavy-theme.md` to drop "at least two distinct sessions" in favor of the judgment it already implies ("a recurring thread, not a one-off").
- Establish a durable prompt-guidance rule so the numbers do not creep back: prompts express agent-judgment behavior qualitatively; a number appears in a prompt only when code enforces it.
- **Explicitly out of scope / NOT changed:** no tool-side query expansion (Option B) is built — the experiment showed it does not pay. `happened_at` temporal resolution stays as prose, including its worked example; it is genuine reasoning over ambiguous language, not an unenforced threshold, so it is correctly prose and untouched.

## Capabilities

### New Capabilities
- `prompt-guidance`: How agent prompts express behavioral guidance — qualitative judgment over unenforced numeric thresholds. Establishes that a number in a prompt must be enforced by code or removed, while genuine reasoning tasks (e.g. temporal resolution) legitimately remain prose.

### Modified Capabilities
<!-- None. Existing specs (entity-storage, graph-retrieval, ingestion-mandate) do not encode prompt-phrasing or theme-creation counts, so no existing requirement changes. -->

## Impact

- **Prompts changed:** `weavy/prompts/weavy-ingestion.md`, `weavy/prompts/weavy-query.md`, `weavy/prompts/weavy-theme.md`. Prose only — no schema, tool, or code changes.
- **No code changes:** `search_graph` is untouched (Option B rejected); no new constants, tools, or dependencies.
- **Behavior:** no measurable change to retrieval quality (per the A/B experiment); the change removes false precision and prevents regression rather than altering agent capability.
- **Tests:** existing tests unaffected (no code paths change). Optionally a lightweight prompt-lint assertion that the prompts contain no unenforced numeric-count instruction.
