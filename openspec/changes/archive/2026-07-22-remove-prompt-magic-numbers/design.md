## Context

Weavy's agent behavior is programmed largely in prose prompts. Three of those prose instructions carry numeric thresholds that no code enforces:

- `weavy/prompts/weavy-ingestion.md` — "search with **3+ different phrasings**"
- `weavy/prompts/weavy-query.md` — "search with **multiple phrasings**"
- `weavy/prompts/weavy-theme.md` — theme creation "require evidence from **at least two distinct sessions** or a single session with clear forward momentum"

These are false precision: they read as rules but are enforced by nothing and informed by nothing (the counts are arbitrary). The theme instruction is even self-hedged ("...or a single session with forward momentum"), which proves the count was never a real gate.

Before committing to a fix, the search-breadth number was tested empirically. An A/B experiment ran gpt-5-mini query agents against one fixed LoCoMo conversation (19 sessions, 25 stratified questions), read-only, holding the graph constant:

- **A (judgment):** prompt says "varied phrasings", agent issues its own searches, `search_graph` unchanged.
- **B (structural):** `search_graph` internally expands one query into ~4 phrasings and merges; the tool guarantees breadth.

Result: A 9/25, B 10/25. The single differing question was won by B taking a `get_session` navigation step, **not** by search breadth (both did one agent search). B fanned out to 4× the query surface yet consulted *fewer* distinct nodes (53 vs 72), cost 2.2× wall-clock, and introduced a new failure mode (verbose auto-expansions that dilute the hybrid ranking). Trace analysis of the 15 both-wrong questions found the failures were ingestion-coverage, reasoning/abstention, or judge-strictness — essentially none were "couldn't find the node for lack of phrasings". The search-breadth lever is not where quality is lost.

## Goals / Non-Goals

**Goals:**
- Remove the three unenforced numeric thresholds, replacing each with the qualitative judgment it stands in for.
- Record a durable rule (the `prompt-guidance` spec) so the numbers do not creep back.
- Preserve genuine reasoning guidance (temporal resolution) untouched.
- End with prose-only edits: no new code, constants, tools, or dependencies.

**Non-Goals:**
- Building tool-side query expansion ("Option B"). Rejected on evidence (see Decisions).
- Changing `search_graph`, its ranking, or any tool schema.
- Touching `happened_at` temporal-resolution guidance or its worked example.
- Optimizing benchmark scores; this change is expected to be quality-neutral by design.

## Decisions

### Decision 1: Option A (delete the number, keep the judgment), not Option B (structural expansion)

The experiment showed B buys no retrieval-quality gain while adding latency, an LLM dependency inside the search primitive, and a new failure mode. The one flip separating A and B was navigation-driven, not breadth-driven, so by the pre-registered rule ("B wins only if it recovers answers A misses *because of breadth*") B earned nothing. The agent also averaged 1.44 searches whether or not told "multiple," so the count is inert regardless. Chosen: delete the counts; do not build B.

### Decision 2: Concrete rewrites

The applying agent should make these edits (wording may be adjusted, but must carry no count):

- **`weavy-ingestion.md`** — replace "search with **3+ different phrasings** — exact name, synonyms, abbreviations, related terms" with "search with **varied phrasings** — exact name, synonyms, abbreviations, related terms". Keep the rest of the sentence (hybrid search, unified ranked list) intact.
- **`weavy-query.md`** — replace "search with **multiple phrasings** — synonyms, abbreviations, related terms, alternate framings" with "search with **varied phrasings** — synonyms, abbreviations, related terms, alternate framings". Leave "The graph is the sole search surface" and everything else in that paragraph intact.
- **`weavy-theme.md`** — replace "Require evidence from at least two distinct sessions or a single session with clear forward momentum." with a count-free judgment, e.g. "Create only for a recurring thread — one that has shown up across sessions or has clear forward momentum — not a one-off topic."

### Decision 3: `prompt-guidance` is a new capability, not a modification

The existing specs (entity-storage, graph-retrieval, ingestion-mandate) do not encode prompt phrasing or theme counts, so nothing is modified. The rule is captured as a new, small spec whose scenarios are checkable against the prompt files, turning "don't re-add magic numbers" into a durable, testable contract.

### Decision 4: Optional prompt-lint test

A lightweight test may assert the three prompts contain no unenforced numeric-count instruction (e.g. regex for "\\d+\\+ .*phrasing", "multiple phrasings", "at least (two|2) .*session"). This is optional; the spec scenarios already define the contract. If added, keep it narrow so it does not fire on legitimate numbers (dates, examples).

## Risks / Trade-offs

- **Small experiment sample (25 questions, one conversation).** The 9-vs-10 result is within noise → mitigation: the decision does not rest on the score but on the qualitative trace finding that failures are not search-breadth failures, which is robust to N.
- **Graph was built by gpt-4o-mini with some duplicate edges.** Mediocre graph quality → mitigation: the graph was identical for both variants, so it cannot bias A vs B; and B's expansion did not help despite the imperfect index.
- **Removing "multiple" could make an agent search less.** → mitigation: the agent already averaged ~1.4 searches with the instruction present, so its behavior is not driven by the word; qualitative "varied phrasings" preserves the intent.
- **A future need for enforced breadth.** If evidence later shows breadth matters, the `prompt-guidance` rule permits a number *if code enforces it* — i.e. revisit as a real structural mechanism, not prose.

## Migration Plan

Prose-only change. Steps: edit the three prompt files per Decision 2; (optionally) add the prompt-lint test; run the existing suite (no code paths change). Rollback is a `git revert` — prompts are loaded at runtime, so revert takes effect immediately with no data implications.

## Open Questions

- Whether to add the optional prompt-lint test now or leave the spec scenarios as the sole contract. Defer to the applying agent / reviewer.
