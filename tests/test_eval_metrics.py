"""Retrieval / citation metric extraction from RunTraces. No FalkorDB or LLM."""

import json
from datetime import datetime, timezone

import pytest

from weavy.eval.metrics import (
    read_sessions,
    retrieval_metrics,
    surfaced_sessions,
    tool_call_count,
)
from weavy.models.traces import RunTrace, ToolCall, Turn

NOW = datetime(2024, 1, 1, tzinfo=timezone.utc)


def _call(name: str, args: dict, result: object = None, error: str | None = None):
    return ToolCall(
        tool_name=name,
        args=args,
        result=None if result is None else json.dumps(result),
        error=error,
        called_at=NOW,
    )


def make_trace(calls: list[ToolCall], cited=(), session_id="s:9") -> RunTrace:
    return RunTrace(
        mode="query",
        started_at=NOW,
        input_summary="q",
        session_id=session_id,
        status="completed",
        turns=[Turn(turn_number=1, tool_calls=calls, timestamp=NOW)],
        completion_payload={"answer": "x", "cited_sources": list(cited)},
    )


GET_NODE_RESULT = {
    "results": [
        {
            "node": {
                "id": "node:1",
                "log": [{"source_id": "s:1", "note": "Brightline"}],
            },
            "edges": [{"id": "edge:1", "source_id": "s:2", "log": []}],
            "mentioned_by": ["s:1", "s:3", "s:9"],
        }
    ],
    "not_found": [],
}


def test_surfaced_sessions_walks_provenance_fields() -> None:
    trace = make_trace(
        [
            _call("search_graph", {"query": "job"}, {"results": []}),
            _call("get_node", {"node_ids": ["node:1"]}, GET_NODE_RESULT),
            _call("get_session", {"session_id": "s:4"}, {"id": "s:4", "text": "t"}),
        ]
    )
    # s:9 is the query's own session and is excluded.
    assert surfaced_sessions(trace) == {"s:1", "s:2", "s:3", "s:4"}
    assert read_sessions(trace) == {"s:4"}
    assert tool_call_count(trace) == 3


def test_failed_calls_and_list_sessions_do_not_count() -> None:
    trace = make_trace(
        [
            _call("get_session", {"session_id": "s:5"}, error="not found"),
            _call(
                "list_sessions", {}, {"sessions": [{"id": "s:6", "source_id": "s:6"}]}
            ),
            _call("create_node", {}, {"ok": True, "source_id": "s:7"}),
        ]
    )
    assert surfaced_sessions(trace) == set()
    assert read_sessions(trace) == set()


def test_retrieval_metrics_values() -> None:
    trace = make_trace(
        [
            _call("get_node", {"node_ids": ["node:1"]}, GET_NODE_RESULT),
            _call(
                "get_session", {"session_id": "s:1"}, {"id": "s:1", "text": "Biscuit"}
            ),
            _call("get_session", {"session_id": "s:3"}, {"id": "s:3", "text": "x"}),
        ],
        cited=["s:1", "s:8", "s:1"],
    )
    m = retrieval_metrics(
        trace, {"s:1", "s:2"}, ["Brightline", "Biscuit", "Matosinhos"]
    )
    assert m["session_recall"] == 1.0  # s:1 and s:2 both surfaced
    assert m["session_precision"] == 0.5  # read s:1 (gold) and s:3 (not gold)
    assert m["fact_recall"] == pytest.approx(2 / 3)
    assert m["cited_sessions"] == ["s:1", "s:8"]  # de-duplicated, order kept
    assert m["citation_precision"] == 0.5  # s:1 gold, s:8 not
    assert m["citation_support"] == 0.5  # s:8 never surfaced
    assert m["unsupported_citations"] == ["s:8"]


def test_metrics_are_none_when_not_measurable() -> None:
    m = retrieval_metrics(make_trace([]), set(), [])
    assert m["session_recall"] is None
    assert m["session_precision"] is None
    assert m["fact_recall"] is None
    assert m["citation_precision"] is None
    assert m["citation_support"] is None


def test_unmapped_gold_counts_against_recall() -> None:
    trace = make_trace([_call("get_node", {"node_ids": ["node:1"]}, GET_NODE_RESULT)])
    m = retrieval_metrics(trace, {"s:1", "unmapped:lost"}, [])
    assert m["session_recall"] == 0.5
