"""
Session CRUD tests.
Requires a running FalkorDB instance (provided by devenv up).
Uses the "weavy_test" graph to avoid touching the main graph.
"""

from datetime import datetime, timezone

import pytest
from falkordb import Graph

from weavy.models.canonical import ChatMessage, Session
from weavy.store.canonical import (
    create_session,
    get_session,
    get_session_messages,
    is_session_completed,
    list_sessions,
    persist_session_outcomes,
)
from weavy.store.system import increment_counter
from tests.helpers import reset_test_graph


@pytest.fixture
def graph() -> Graph:
    g = reset_test_graph("Session")
    yield g
    g.query("MATCH (s:System) DELETE s")
    g.query("MATCH (n:Session) DELETE n")


def _make_session(graph: Graph, messages: list[ChatMessage] | None = None) -> Session:
    if messages is None:
        messages = [
            ChatMessage(role="user", content="Hello"),
            ChatMessage(role="assistant", content="Hi there"),
            ChatMessage(role="user", content="How are you?"),
        ]
    session_id = increment_counter(graph, "session")
    s = Session(
        id=session_id,
        timestamp=datetime(2024, 6, 2, 10, 0, 0, tzinfo=timezone.utc),
        messages=messages,
    )
    create_session(graph, s)
    return s


# ---------------------------------------------------------------------------
# Session CRUD
# ---------------------------------------------------------------------------


def test_create_and_get_session(graph: Graph) -> None:
    s = _make_session(graph)
    fetched = get_session(graph, s.id)

    assert fetched.id == s.id
    assert fetched.timestamp == s.timestamp
    assert len(fetched.messages) == len(s.messages)
    assert fetched.messages[0].role == "user"
    assert fetched.messages[0].content == "Hello"
    assert fetched.messages[1].role == "assistant"


def test_get_session_not_found(graph: Graph) -> None:
    with pytest.raises(ValueError, match="not found"):
        get_session(graph, "s:999")


def test_list_sessions_no_filter(graph: Graph) -> None:
    s1 = _make_session(graph)
    s2 = _make_session(graph)
    output = list_sessions(graph)

    ids = [s.id for s in output.sessions]
    assert s1.id in ids
    assert s2.id in ids


def test_list_sessions_date_range(graph: Graph) -> None:
    s = _make_session(graph)
    start = datetime(2024, 1, 1, tzinfo=timezone.utc)
    end = datetime(2024, 12, 31, tzinfo=timezone.utc)
    output = list_sessions(graph, date_range=[start, end])
    assert any(ss.id == s.id for ss in output.sessions)

    past_start = datetime(2023, 1, 1, tzinfo=timezone.utc)
    past_end = datetime(2023, 12, 31, tzinfo=timezone.utc)
    output_empty = list_sessions(graph, date_range=[past_start, past_end])
    assert all(ss.id != s.id for ss in output_empty.sessions)


def test_list_sessions_limit(graph: Graph) -> None:
    for _ in range(3):
        _make_session(graph)
    output = list_sessions(graph, limit=2)
    assert len(output.sessions) <= 2


def test_get_session_messages_with_slice(graph: Graph) -> None:
    s = _make_session(graph)
    result = get_session_messages(graph, s.id, 1, 2)

    assert len(result.messages) == 1
    assert result.messages[0].role == "assistant"
    assert result.messages[0].content == "Hi there"


def test_get_session_messages_returns_all(graph: Graph) -> None:
    s = _make_session(
        graph,
        [
            ChatMessage(role="user", content="12345"),
            ChatMessage(role="assistant", content="67890"),
            ChatMessage(role="user", content="abcde"),
        ],
    )

    result = get_session_messages(graph, s.id, None, None)

    assert [message.content for message in result.messages] == [
        "12345",
        "67890",
        "abcde",
    ]


# ---------------------------------------------------------------------------
# Session outcomes
# ---------------------------------------------------------------------------


def test_persist_and_check_outcomes(graph: Graph) -> None:
    s = _make_session(graph)
    assert not is_session_completed(graph, s.id)

    persist_session_outcomes(
        graph,
        s.id,
        "Test summary",
        {"nodes_created": ["node:1"]},
        "2024-06-01T12:00:00Z",
    )
    assert is_session_completed(graph, s.id)
