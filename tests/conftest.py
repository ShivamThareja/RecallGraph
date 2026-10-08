"""
Shared pytest fixtures and autouse mocks.

RunTracer, fetch_prompt, and embedding functions are automatically mocked for
all tests to prevent real service calls. Tests that specifically test Langfuse
behaviour (e.g. test_harness.py's RunTracer tests) override these mocks as needed.
"""

from unittest.mock import MagicMock, patch

import pytest

TEST_EMBEDDING_DIM = 8


@pytest.fixture(autouse=True)
def clear_langfuse_cache():
    try:
        from weavy.langfuse_client import get_langfuse

        get_langfuse.cache_clear()
        yield
        get_langfuse.cache_clear()
    except ImportError:
        yield


@pytest.fixture(autouse=True)
def mock_run_tracer():
    """Prevent real Langfuse calls from RunTracer in all tests."""
    try:
        from weavy.harness import runner
    except ImportError:
        yield None
        return

    mock = MagicMock()
    mock.get_trace_id.return_value = "mock-trace-id"
    with patch.object(runner, "RunTracer", return_value=mock):
        yield mock


@pytest.fixture(autouse=True)
def mock_fetch_prompt():
    """Mock prompt loading to avoid filesystem dependency in tests."""
    try:
        from weavy.application import prompts, theme_runs
    except ImportError:
        yield None
        return

    with (
        patch.object(prompts, "fetch_prompt", return_value="(mocked system prompt)"),
        patch.object(
            theme_runs, "fetch_prompt", return_value="(mocked system prompt)"
        ) as mock,
    ):
        yield mock


def _fake_embed(text: str) -> list[float]:
    """Deterministic per-text vector: same text -> same vector, different
    texts -> (near-)orthogonal vectors. Keeps similarity semantics (and the
    create_node duplicate guard) honest in tests without real embedding calls."""
    import hashlib
    import math

    digest = hashlib.sha256(text.encode()).digest()
    vec = [digest[i] - 127.5 for i in range(TEST_EMBEDDING_DIM)]
    norm = math.sqrt(sum(x * x for x in vec)) or 1.0
    return [x / norm for x in vec]


@pytest.fixture(autouse=True)
def mock_embeddings():
    """Mock embedding calls to avoid real LiteLLM/Gemini calls in tests."""
    try:
        from weavy.services import embedding
    except ImportError:
        yield None
        return

    with (
        patch.object(embedding, "embed", side_effect=_fake_embed),
        patch.object(
            embedding,
            "embed_node",
            side_effect=lambda aliases, summary: _fake_embed(
                " | ".join(aliases) + " — " + summary
            ),
        ),
        patch.object(
            embedding,
            "embed_edge",
            side_effect=lambda label, fact: _fake_embed(f"{label} — {fact}"),
        ),
        patch.object(embedding, "get_dimension", return_value=TEST_EMBEDDING_DIM),
    ):
        yield
