import json
from datetime import datetime, timezone
from typing import Any, Literal
from unittest.mock import MagicMock

from falkordb import Graph

from weavy.models.canonical import (
    ChatMessage,
    Session,
)
from weavy.models.traces import RunTrace
from weavy.store import canonical as store_canonical
from weavy.store.client import get_graph
from weavy.store.system import increment_counter, init_system

TEST_GRAPH = "weavy_test"

SAMPLE_TEXT = (
    "I've been thinking about changing jobs a lot lately. "
    "The mortgage scares me but I'm feeling trapped. "
    "Had a great talk with my mentor yesterday about risk."
)


def reset_test_graph(*extra_labels: str) -> Graph:
    """Reset the shared FalkorDB test graph and reinitialize system state."""
    from tests.conftest import TEST_EMBEDDING_DIM

    graph = get_graph(TEST_GRAPH)
    graph.query("MATCH (s:System) DELETE s")
    graph.query("MATCH (n:SemanticNode) DETACH DELETE n")
    graph.query("MATCH (t:RunTrace) DETACH DELETE t")
    for label in extra_labels:
        graph.query(f"MATCH (n:{label}) DETACH DELETE n")
    init_system(graph, embedding_dim=TEST_EMBEDDING_DIM)
    return graph


def store_test_session(graph: Any, text: str) -> str:
    """Create a session from text for integration-style tests."""
    session_id = increment_counter(graph, "session")
    store_canonical.create_session(
        graph,
        Session(
            id=session_id,
            timestamp=datetime.now(tz=timezone.utc),
            messages=[ChatMessage(role="user", content=text)],
        ),
    )
    return session_id


def store_test_session_with_id(
    graph: Any, session_id: str, text: str = SAMPLE_TEXT
) -> str:
    """Create a Session node with an explicit id (for MENTIONS / provenance tests)."""
    store_canonical.create_session(
        graph,
        Session(
            id=session_id,
            timestamp=datetime.now(tz=timezone.utc),
            messages=[ChatMessage(role="user", content=text)],
        ),
    )
    return session_id


def mock_tool_response(
    tool_name: str,
    args: dict[str, Any],
    call_id: str = "tc-1",
    *,
    usage: dict[str, int] | None = None,
) -> MagicMock:
    """Build a mock litellm response for a single tool call."""
    tc = MagicMock()
    tc.id = call_id
    tc.function.name = tool_name
    tc.function.arguments = json.dumps(args)

    msg = MagicMock()
    msg.tool_calls = [tc]
    msg.content = None
    msg.reasoning_content = None

    choice = MagicMock()
    choice.message = msg

    resp = MagicMock()
    resp.choices = [choice]
    if usage is None:
        resp.usage = None
        return resp

    usage_mock = MagicMock()
    usage_mock.prompt_tokens = usage.get("prompt_tokens", 0)
    usage_mock.completion_tokens = usage.get("completion_tokens", 0)
    usage_mock.total_tokens = usage.get("total_tokens", 0)
    usage_mock.completion_tokens_details = usage.get("completion_tokens_details")
    resp.usage = usage_mock
    return resp


def mock_text_response(content: str = "I think the answer is...") -> MagicMock:
    """Build a mock response where the model returns text without a tool call."""
    msg = MagicMock()
    msg.tool_calls = None
    msg.content = content
    msg.reasoning_content = None

    choice = MagicMock()
    choice.message = msg

    resp = MagicMock()
    resp.choices = [choice]
    resp.usage = None
    return resp


def make_test_trace(
    mode: Literal["ingestion", "query", "theme"] = "ingestion",
) -> RunTrace:
    return RunTrace(
        mode=mode,
        started_at=datetime.now(tz=timezone.utc),
        input_summary="test",
    )
