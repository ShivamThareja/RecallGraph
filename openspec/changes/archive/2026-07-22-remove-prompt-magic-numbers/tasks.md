## 1. Rewrite prompt thresholds

- [x] 1.1 In `weavy/prompts/weavy-ingestion.md`, replace "search with **3+ different phrasings**" with "search with **varied phrasings**", keeping the rest of the sentence (synonyms/abbreviations list, hybrid-search explanation) intact.
- [x] 1.2 In `weavy/prompts/weavy-query.md`, replace "search with **multiple phrasings**" with "search with **varied phrasings**", leaving "The graph is the sole search surface" and the rest of that paragraph unchanged.
- [x] 1.3 In `weavy/prompts/weavy-theme.md`, replace "Require evidence from at least two distinct sessions or a single session with clear forward momentum." with a count-free judgment (e.g. "Create only for a recurring thread — one that has shown up across sessions or has clear forward momentum — not a one-off topic.").

## 2. Verify no unenforced counts remain and reasoning guidance is preserved

- [x] 2.1 Grep the three prompts to confirm none of "3+", "multiple phrasings", or "at least two/2 ... session" remain, and that no other unenforced numeric-count instruction was introduced.
- [x] 2.2 Confirm `weavy/prompts/weavy-ingestion.md` still contains the full `happened_at` temporal-resolution guidance and its worked example, unchanged (out of scope — must not be stripped).
- [x] 2.3 (Optional) Add a narrow prompt-lint test asserting the three prompts contain no unenforced numeric-count instruction, scoped so it does not fire on legitimate numbers (dates, examples).

## 3. Validate

- [x] 3.1 Run `uv run pytest` and `uv run ruff check .` — confirm the suite passes (no code paths change; prompts are auto-mocked in tests via `fetch_prompt`, so update any prompt-text fixtures/assertions if present).
- [x] 3.2 Run `openspec validate remove-prompt-magic-numbers` and confirm the change validates against the `prompt-guidance` spec scenarios.
