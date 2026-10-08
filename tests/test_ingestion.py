"""
Ingestion and theme mode tests.
Mocks litellm.completion to avoid real LLM calls.
Uses the "weavy_test" graph to avoid touching the main graph.
"""

from datetime import datetime, timezone
from unittest.mock import patch

import pytest
from falkordb import Graph

from weavy.application.session_runs import finalize_session, run_session
from weavy.application.theme_runs import run_theme_update
from weavy.models.graph import ProvenanceInput
from weavy.models.traces import RunTrace
from weavy.store import canonical as store_canonical
from weavy.store import graph as store_graph
from weavy.store import themes as store_themes
from weavy.store.system import increment_counter
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
# ingestion runs
# ---------------------------------------------------------------------------


def test_ingest_into_empty_graph(graph: Graph) -> None:
    """Agent creates a node then calls complete; node appears in graph."""
    session_id = store_test_session(graph, SAMPLE_TEXT)

    prov = ProvenanceInput(source_id=session_id)
    create_args = {
        "aliases": ["job change"],
        "summary": "Contemplating leaving current job.",
        "note": "First mention — feels trapped, mortgage a barrier.",
        "provenance": prov.model_dump(),
    }
    completion_args = {"summary": "Ingested text about career anxiety."}

    usage = {"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150}
    create_resp = mock_tool_response("create_node", create_args, "tc-1", usage=usage)
    done_resp = mock_tool_response("complete", completion_args, "tc-2", usage=usage)

    with patch("litellm.completion", side_effect=[create_resp, done_resp]):
        trace = run_session(session_id, "ingestion", graph)

    assert trace.status == "completed"
    assert len(trace.touched_nodes) == 1
    assert trace.touched_nodes[0].action == "created"
    node_id = trace.touched_nodes[0].node_id

    result = graph.query(
        "MATCH (n:SemanticNode {id: $id}) RETURN n.id", {"id": node_id}
    )
    assert result.result_set


def test_ingest_updates_existing_node(graph: Graph) -> None:
    """Agent calls update_node; log count increments."""
    session_id = store_test_session(graph, SAMPLE_TEXT)

    node_id = increment_counter(graph, "node")
    prov = ProvenanceInput(source_id=session_id)
    store_graph.create_node(
        graph, ["career anxiety"], "Initial state.", "first entry", prov, node_id
    )

    prov2 = ProvenanceInput(source_id=session_id)
    update_args = {
        "node_id": node_id,
        "note": "Mortgage fear now explicit.",
        "new_summary": "Career anxiety deepened by mortgage fear.",
        "provenance": prov2.model_dump(),
    }
    completion_args = {"summary": "Updated career anxiety node."}

    usage = {"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150}
    update_resp = mock_tool_response("update_node", update_args, "tc-1", usage=usage)
    done_resp = mock_tool_response("complete", completion_args, "tc-2", usage=usage)

    with patch("litellm.completion", side_effect=[update_resp, done_resp]):
        trace = run_session(session_id, "ingestion", graph)

    assert trace.status == "completed"
    assert len(trace.touched_nodes) == 1
    assert trace.touched_nodes[0].action == "updated"

    node_out = store_graph.get_node(graph, node_id)
    assert node_out.node.total_log_count == 2


def test_ingest_invalid_provenance_fails(graph: Graph) -> None:
    """Agent calls create_node without provenance; runner captures tool error and fails."""
    session_id = store_test_session(graph, SAMPLE_TEXT)

    bad_args = {
        "aliases": ["concept"],
        "summary": "Some concept.",
        "note": "No provenance.",
    }
    bad_resp = mock_tool_response(
        "create_node",
        bad_args,
        usage={"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150},
    )

    with patch("litellm.completion", return_value=bad_resp):
        trace = run_session(session_id, "ingestion", graph)

    assert trace.status == "failed"
    assert trace.touched_nodes == []


def test_ingest_missing_session_raises(graph: Graph) -> None:
    """Ingestion raises ValueError for a non-existent session id."""
    with pytest.raises(ValueError, match="not found"):
        run_session("s:999", "ingestion", graph)


# ---------------------------------------------------------------------------
# run_theme_update
# ---------------------------------------------------------------------------


def test_theme_update_creates_theme(graph: Graph) -> None:
    """Theme agent calls create_theme then complete_theme_update; Theme node persists."""
    session_id = store_test_session(graph, SAMPLE_TEXT)
    node_id = increment_counter(graph, "node")
    prov = ProvenanceInput(source_id=session_id)
    store_graph.create_node(
        graph, ["job change"], "Career decision node.", "init", prov, node_id
    )

    store_canonical.persist_session_outcomes(
        graph,
        session_id,
        "Ingested career text.",
        {"nodes_created": [node_id]},
        "2024-06-01T12:00:00Z",
    )

    create_theme_args = {
        "name": "career-direction",
        "state": "Weighing whether to leave job.",
        "anchors": [node_id],
    }
    completion_args = {"updated_themes": ["career-direction"]}

    usage = {"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150}
    create_resp = mock_tool_response(
        "create_theme", create_theme_args, "tc-1", usage=usage
    )
    done_resp = mock_tool_response(
        "complete_theme_update", completion_args, "tc-2", usage=usage
    )

    with patch("litellm.completion", side_effect=[create_resp, done_resp]):
        trace = run_theme_update(graph)

    assert trace.status == "completed"

    result = store_themes.get_theme(graph, "career-direction")
    assert result.name == "career-direction"
    assert node_id in result.anchors
    assert [t.name for t in store_themes.list_all_themes(graph)] == ["career-direction"]


def test_theme_update_empty_map_runs(graph: Graph) -> None:
    """Theme agent can run with no existing themes or sessions."""
    completion_args = {"updated_themes": []}
    done_resp = mock_tool_response(
        "complete_theme_update",
        completion_args,
        usage={"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150},
    )

    with patch("litellm.completion", return_value=done_resp):
        trace = run_theme_update(graph)

    assert trace.status == "completed"


# ---------------------------------------------------------------------------
# Session completion guard
# ---------------------------------------------------------------------------


def test_ingest_writes_outcomes(graph: Graph) -> None:
    """After a successful ingestion, session outcomes are persisted."""
    session_id = store_test_session(graph, SAMPLE_TEXT)

    prov = ProvenanceInput(source_id=session_id)
    create_args = {
        "aliases": ["job change"],
        "summary": "Contemplating leaving current job.",
        "note": "Feels trapped.",
        "provenance": prov.model_dump(),
    }
    completion_args = {"summary": "Ingested career text."}
    usage = {"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150}

    with patch(
        "litellm.completion",
        side_effect=[
            mock_tool_response("create_node", create_args, "tc-1", usage=usage),
            mock_tool_response("complete", completion_args, "tc-2", usage=usage),
        ],
    ):
        trace = run_session(session_id, "ingestion", graph)

    assert trace.status == "completed"
    assert store_canonical.is_session_completed(graph, session_id)


def test_reingest_blocked_when_completed(graph: Graph) -> None:
    """Second ingestion run raises ValueError when already completed."""
    session_id = store_test_session(graph, SAMPLE_TEXT)
    store_canonical.persist_session_outcomes(
        graph, session_id, "done", {}, "2024-06-01T12:00:00Z"
    )
    with pytest.raises(ValueError, match="already completed"):
        run_session(session_id, "ingestion", graph)


def test_finalize_session_persists_conversation(graph: Graph) -> None:
    session_id = store_test_session(graph, SAMPLE_TEXT)
    trace = RunTrace(
        mode="query",
        started_at=datetime.now(tz=timezone.utc),
        input_summary="test",
        status="completed",
        conversation=[],
        completion_payload={"summary": "Cleared history."},
    )

    finalize_session(graph, session_id, trace)

    stored = store_canonical.get_session(graph, session_id)
    assert stored.messages == []
