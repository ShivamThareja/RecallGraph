import re
from pathlib import Path

import pytest

PROMPTS_DIR = Path(__file__).resolve().parent.parent / "weavy" / "prompts"

UNENFORCED_COUNT_PATTERNS = [
    re.compile(r"\d+\+ .*phrasing", re.IGNORECASE),
    re.compile(r"multiple phrasings", re.IGNORECASE),
    re.compile(r"at least (two|2)\D*session", re.IGNORECASE),
]


@pytest.mark.parametrize(
    "prompt_file",
    ["weavy-ingestion.md", "weavy-query.md", "weavy-theme.md"],
)
def test_prompt_has_no_unenforced_numeric_count(prompt_file):
    text = (PROMPTS_DIR / prompt_file).read_text()
    for pattern in UNENFORCED_COUNT_PATTERNS:
        assert not pattern.search(text), (
            f"{prompt_file} contains an unenforced numeric-count instruction "
            f"matching {pattern.pattern!r}"
        )
