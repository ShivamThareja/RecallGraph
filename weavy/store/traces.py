"""
RunTrace persistence in FalkorDB.

Langfuse is useful for observability, but Weavy also keeps a local copy of every
completed or failed harness run so traces survive without an external service.
"""

from __future__ import annotations

import json

from falkordb import Graph

from weavy.models.traces import RunTrace, graph_changes


def persist_trace(graph: Graph, trace: RunTrace) -> None:
    """Upsert a complete RunTrace as a local FalkorDB node."""
    trace_json = trace.model_dump_json()
    changes = graph_changes(trace.touched_nodes, trace.touched_edges)

    graph.query(
        """
        MERGE (t:RunTrace {id: $id})
        SET t.name = $id,
            t.mode = $mode,
            t.status = $status,
            t.session_id = $session_id,
            t.started_at = $started_at,
            t.ended_at = $ended_at,
            t.input_summary = $input_summary,
            t.error = $error,
            t.completion_payload = $completion_payload,
            t.total_usage = $total_usage,
            t.graph_changes = $graph_changes,
            t.trace_json = $trace_json
        """,
        {
            "id": trace.run_id,
            "mode": trace.mode,
            "status": trace.status,
            "session_id": trace.session_id,
            "started_at": trace.started_at.isoformat(),
            "ended_at": trace.ended_at.isoformat() if trace.ended_at else None,
            "input_summary": trace.input_summary,
            "error": trace.error,
            "completion_payload": json.dumps(trace.completion_payload or {}),
            "total_usage": trace.total_usage.model_dump_json(),
            "graph_changes": json.dumps(changes),
            "trace_json": trace_json,
        },
    )


def get_trace(graph: Graph, run_id: str) -> RunTrace:
    """Load a persisted RunTrace by run id."""
    result = graph.query(
        "MATCH (t:RunTrace {id: $id}) RETURN t.trace_json",
        {"id": run_id},
    )
    if not result.result_set:
        raise ValueError(f"RunTrace '{run_id}' not found.")
    return RunTrace.model_validate_json(result.result_set[0][0])
