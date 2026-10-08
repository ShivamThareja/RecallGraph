"""
Query/chat mode tests.
Mocks litellm.completion to avoid real LLM calls.
Uses the "weavy_test" graph to avoid touching the main graph.
"""

from unittest.mock import patch

import pytest
from falkordb import Graph

from weavy.application.session_runs import run_query
from tests.helpers import (
    SAMPLE_TEXT,
    mock_tool_response,
    reset_test_graph,
    store_test_session,
)


@pytest.fixture
def graph() -> Graph:
    return reset_test_graph("Theme", "Session")


# ---------------------------------------------------------------------------
# run_query
# ---------------------------------------------------------------------------


def test_query_delivers_response(graph: Graph) -> None:
    """Agent calls complete; trace is completed with answer in payload."""
    store_test_session(graph, SAMPLE_TEXT)

    deliver_args = {
        "summary": "Searched graph for job thoughts.",
        "answer": "You have been thinking about changing jobs.",
        "cited_sources": [],
        "consulted_nodes": [],
    }
    done_resp = mock_tool_response("complete", deliver_args)

    with patch("litellm.completion", return_value=done_resp):
        trace = run_query("What have I been thinking about?", graph)

    assert trace.status == "completed"
    assert (
        trace.completion_payload["answer"]
        == "You have been thinking about changing jobs."
    )


def test_query_creates_session(graph: Graph) -> None:
    """A Session record is persisted in the store after a completed query run."""
    store_test_session(graph, SAMPLE_TEXT)

    deliver_args = {
        "summary": "Answered question about job thoughts.",
        "answer": "You have been thinking about changing jobs.",
        "cited_sources": [],
        "consulted_nodes": [],
    }
    done_resp = mock_tool_response("complete", deliver_args)

    with patch("litellm.completion", return_value=done_resp):
        trace = run_query("What have I been thinking about?", graph)

    assert trace.status == "completed"

    result = graph.query(
        "MATCH (s:Session) WHERE s.id STARTS WITH 's:' RETURN s.id ORDER BY s.id"
    )
    assert len(result.result_set) >= 2


def test_query_conversation_captured(graph: Graph) -> None:
    """trace.conversation is populated with non-system messages after the run."""
    store_test_session(graph, SAMPLE_TEXT)

    deliver_args = {
        "summary": "Searched for career anxiety.",
        "answer": "Yes, career anxiety is a recurring theme.",
        "cited_sources": [],
        "consulted_nodes": [],
    }
    done_resp = mock_tool_response("complete", deliver_args)

    with patch("litellm.completion", return_value=done_resp):
        trace = run_query("Am I anxious about my career?", graph)

    assert trace.conversation is not None
    roles = [m["role"] for m in trace.conversation]
    assert "system" not in roles
    assert "user" in roles
