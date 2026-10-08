"""Citation verification, provenance enforcement on node writes, tool-input time
validation, the offline hash embedder, and UTC helpers."""

import json
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch

import pytest
from pydantic import ValidationError

from tests.helpers import (
    mock_tool_response,
    reset_test_graph,
    store_test_session_with_id,
)
from weavy.harness import actions
from weavy.harness.actions import SESSION_ACTIONS, ActionContext
from weavy.harness.runner import run
from weavy.harness.tool_models import (
    CreateNodeInput,
    ListSessionsInput,
    SearchGraphInput,
    UpdateNodeInput,
)
from weavy.models.traces import RunTrace, ToolCall, Turn
from weavy.services.embedding import hash_embed
from weavy.services.provenance import gather_evidence, verify_citations
from weavy.store.traces import get_trace
from weavy.timeutil import in_range, normalize_range, parse_utc, to_utc

NOW = datetime(2024, 1, 1, tzinfo=timezone.utc)


def _call(name, args, result=None, error=None) -> ToolCall:
    return ToolCall(
        tool_name=name,
        args=args,
        result=None if result is None else json.dumps(result),
        error=error,
        called_at=NOW,
    )


def _trace(calls, payload, session_id="s:9") -> RunTrace:
    return RunTrace(
        mode="query",
        started_at=NOW,
        input_summary="q",
        session_id=session_id,
        turns=[Turn(turn_number=1, tool_calls=calls, timestamp=NOW)],
        completion_payload=payload,
    )


SEARCH_RESULT = {
    "results": [
        {"kind": "node", "id": "node:1", "source_sessions": ["s:1", "s:2"]},
        {
            "kind": "edge",
            "id": "edge:4",
            "endpoints": ["node:1", "node:7"],
            "source_sessions": ["s:3"],
        },
    ]
}


# --- evidence + citation verification --------------------------------------


def test_gather_evidence_uses_search_provenance() -> None:
    ev = gather_evidence(
        [
            _call("search_graph", {"query": "x"}, SEARCH_RESULT),
            _call("get_session", {"session_id": "s:5"}, {"id": "s:5", "text": "t"}),
            _call("list_sessions", {}, {"sessions": [{"id": "s:6"}]}),
            _call("get_session", {"session_id": "s:8"}, error="boom"),
            _call("get_session", {"session_id": "s:9"}, {"id": "s:9"}),
        ],
        exclude_session="s:9",
    )
    assert ev.surfaced_sessions == {"s:1", "s:2", "s:3", "s:5"}
    assert ev.read_sessions == {"s:5"}
    assert ev.seen_memories == {"node:1", "node:7", "edge:4"}


def test_verify_citations_splits_verified_and_unsupported() -> None:
    trace = _trace(
        [
            _call("search_graph", {"query": "x"}, SEARCH_RESULT),
            _call("get_session", {"session_id": "s:1"}, {"id": "s:1"}),
        ],
        {
            "answer": "a",
            "cited_sources": ["s:1", "s:3", "s:42", "s:9", "s:1"],
            "consulted_nodes": ["node:1", "node:99"],
        },
    )
    check = verify_citations(trace)
    assert check == {
        "verified_sources": ["s:1", "s:3"],
        "unsupported_sources": ["s:42", "s:9"],  # s:9 is the run's own session
        "read_sources": ["s:1"],
        "verified_nodes": ["node:1"],
        "unsupported_nodes": ["node:99"],
    }


def test_verify_citations_none_without_citations() -> None:
    assert verify_citations(_trace([], {"answer": "a"})) is None
    assert verify_citations(_trace([], None)) is None


def test_runner_records_citation_check_on_persisted_trace() -> None:
    graph = reset_test_graph("Session")
    store_test_session_with_id(graph, "s:1", "Ada adopted Oscar.")
    script = [
        mock_tool_response("get_session", {"session_id": "s:1"}, "tc-1"),
        mock_tool_response(
            "complete",
            {"summary": "s", "answer": "Oscar", "cited_sources": ["s:1", "s:77"]},
            "tc-2",
        ),
    ]
    with patch("litellm.completion", side_effect=script):
        trace = run(
            mode="query",
            system_prompt="p",
            initial_messages=[{"role": "user", "content": "q"}],
            allowed_actions=SESSION_ACTIONS,
            run_context={"input_summary": "q"},
            graph=graph,
            session_id="s:2",
        )
    check = trace.completion_payload["citation_check"]
    assert check["verified_sources"] == ["s:1"]
    assert check["unsupported_sources"] == ["s:77"]
    assert get_trace(graph, trace.run_id).completion_payload["citation_check"] == check


# --- provenance enforcement on node writes ---------------------------------


def _ctx(session_id: str | None) -> ActionContext:
    return ActionContext(
        graph=MagicMock(),
        trace=RunTrace(mode="ingestion", started_at=NOW, input_summary="t"),
        session_id=session_id,
    )


@pytest.mark.parametrize("tool", ["create_node", "update_node"])
def test_node_write_with_foreign_provenance_is_refused(tool) -> None:
    params = (
        CreateNodeInput(
            aliases=["x"], summary="x", note="n", provenance={"source_id": "s:1"}
        )
        if tool == "create_node"
        else UpdateNodeInput(
            node_id="node:1", note="n", provenance={"source_id": "s:1"}
        )
    )
    ctx = _ctx("s:2")
    with patch.object(actions.memory, tool) as write:
        result = actions.ACTIONS[tool].fn(params, ctx)
    assert result.ok is False
    assert "s:2" in result.message
    write.assert_not_called()


def test_node_write_with_matching_provenance_proceeds() -> None:
    params = CreateNodeInput(
        aliases=["x"], summary="x", note="n", provenance={"source_id": "s:2"}
    )
    with patch.object(actions.memory, "create_node") as write:
        actions.ACTIONS["create_node"].fn(params, _ctx("s:2"))
    write.assert_called_once()
    assert write.call_args.kwargs["session_id"] == "s:2"


def test_sessionless_runs_keep_agent_provenance() -> None:
    params = CreateNodeInput(
        aliases=["x"], summary="x", note="n", provenance={"source_id": "s:1"}
    )
    with patch.object(actions.memory, "create_node") as write:
        actions.ACTIONS["create_node"].fn(params, _ctx(None))
    write.assert_called_once()


# --- tool input time validation --------------------------------------------


def test_tool_time_ranges_normalize_and_validate() -> None:
    parsed = SearchGraphInput(query="q", time_range=["2024-01-01T05:30:00+05:30", None])
    assert parsed.time_range == [datetime(2024, 1, 1, tzinfo=timezone.utc), None]
    with pytest.raises(ValidationError):
        SearchGraphInput(query="q", time_range=["2024-02-01", "2024-01-01"])
    with pytest.raises(ValidationError):
        ListSessionsInput(date_range=["2024-02-01"])
    with pytest.raises(ValidationError):
        SearchGraphInput(query="q", limit=0)


# --- UTC helpers ------------------------------------------------------------


def test_utc_helpers() -> None:
    ist = timezone(timedelta(hours=5, minutes=30))
    assert to_utc(datetime(2024, 1, 1)) == datetime(2024, 1, 1, tzinfo=timezone.utc)
    assert to_utc(datetime(2024, 1, 1, 5, 30, tzinfo=ist)) == datetime(
        2024, 1, 1, tzinfo=timezone.utc
    )
    assert parse_utc("2024-01-01T00:00:00") == datetime(2024, 1, 1, tzinfo=timezone.utc)
    assert normalize_range(None) is None
    assert normalize_range([None, None]) is None
    start = datetime(2024, 1, 1, tzinfo=timezone.utc)
    assert in_range(datetime(2024, 6, 1), start, None)
    assert not in_range(datetime(2023, 6, 1), start, None)


# --- offline hash embedder --------------------------------------------------


def test_hash_embed_is_deterministic_normalized_and_lexical() -> None:
    a = hash_embed("Biscuit the greyhound", 512)
    assert a == hash_embed("biscuit THE greyhound", 512)
    assert sum(v * v for v in a) == pytest.approx(1.0)
    cos = lambda x, y: sum(i * j for i, j in zip(x, y))  # noqa: E731
    related = hash_embed("greyhound named Biscuit", 512)
    unrelated = hash_embed("quantum chromodynamics lecture", 512)
    assert cos(a, related) > cos(a, unrelated)
    assert hash_embed("", 8) == [1.0] + [0.0] * 7


def test_hash_embedding_model_selected_by_name(monkeypatch) -> None:
    from weavy.services import embedding

    monkeypatch.setattr(embedding.settings, "EMBEDDING_MODEL", "weavy/hash-64")
    # conftest mocks embed/get_dimension; call the undecorated implementations.
    assert embedding._local_hash_dim("weavy/hash-64") == 64
    assert embedding._local_hash_dim("gemini/x") is None
    with pytest.raises(ValueError):
        embedding._local_hash_dim("weavy/hash-abc")
