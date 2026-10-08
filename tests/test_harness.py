"""
Agent harness, completion actions, and runner loop tests.
Runner tests mock litellm.completion to avoid real LLM calls.
Integration tests that touch FalkorDB use the "weavy_test" graph.
"""

import json
from unittest.mock import MagicMock, patch

import pytest
from falkordb import Graph

from weavy.harness import actions as reg
from weavy.harness import runner as harness_runner
from weavy.harness.runner import run
from weavy.harness.tracing import RunTracer
from weavy.models.traces import TurnUsage
from weavy.store.traces import get_trace
from tests.helpers import (
    make_test_trace,
    mock_text_response,
    mock_tool_response,
    reset_test_graph,
)


@pytest.fixture
def graph() -> Graph:
    return reset_test_graph()


# ---------------------------------------------------------------------------
# tracing.py — RunTracer (mocked Langfuse)
# ---------------------------------------------------------------------------


def _make_tracer(mode: str = "ingestion") -> tuple[RunTracer, MagicMock]:
    """Create a RunTracer with a mocked Langfuse client. Returns (tracer, mock_langfuse)."""
    mock_lf = MagicMock()
    mock_root = MagicMock()
    mock_lf.start_observation.return_value = mock_root

    with patch("weavy.langfuse_client.get_langfuse", return_value=mock_lf):
        tracer = RunTracer("test-run-id", mode, "test input")

    tracer._lf = mock_lf
    tracer._root = mock_root
    return tracer, mock_lf


def test_run_tracer_creates_langfuse_trace() -> None:
    mock_lf = MagicMock()
    mock_root = MagicMock()
    mock_lf.start_observation.return_value = mock_root

    with patch("weavy.langfuse_client.get_langfuse", return_value=mock_lf):
        tracer = RunTracer(
            "1eaa8338-f95d-4592-838f-00d18a352b81",
            "ingestion",
            "Ingesting s:1",
            session_id="s:1",
        )

    mock_lf.start_observation.assert_called_once()
    call_kwargs = mock_lf.start_observation.call_args[1]
    assert call_kwargs["trace_context"] == {
        "trace_id": "1eaa8338f95d4592838f00d18a352b81"
    }
    assert call_kwargs["name"] == "ingestion-run"
    assert call_kwargs["input"] == "Ingesting s:1"
    assert tracer.get_trace_id() == "1eaa8338-f95d-4592-838f-00d18a352b81"


def test_run_tracer_start_and_end_turn() -> None:
    tracer, _ = _make_tracer()
    mock_turn_span = MagicMock()
    tracer._root.start_observation.return_value = mock_turn_span

    tracer.start_turn(1, 5)
    assert tracer._current_turn_span is mock_turn_span
    call_kwargs = tracer._root.start_observation.call_args[1]
    assert call_kwargs["name"] == "turn-1"
    assert call_kwargs["input"] == {"message_count": 5}

    tracer.end_turn("some text")
    mock_turn_span.update.assert_called_with(output="some text")
    mock_turn_span.end.assert_called_once()
    assert tracer._current_turn_span is None


def test_run_tracer_record_tool_call() -> None:
    tracer, _ = _make_tracer()
    mock_turn_span = MagicMock()
    mock_tool_span = MagicMock()
    tracer._root.start_observation.return_value = mock_turn_span
    mock_turn_span.start_observation.return_value = mock_tool_span

    tracer.start_turn(1, 3)
    tracer.record_tool_call(
        1, "search_graph", {"query": "career"}, '{"results":[]}', 42.0
    )

    mock_turn_span.start_observation.assert_called_once()
    call_kwargs = mock_turn_span.start_observation.call_args[1]
    assert call_kwargs["name"] == "tool:search_graph"
    assert call_kwargs["input"] == {"query": "career"}
    assert call_kwargs["as_type"] == "tool"
    assert call_kwargs["metadata"] == {"duration_ms": 42.0}
    mock_tool_span.update.assert_called_once_with(output='{"results":[]}')
    mock_tool_span.end.assert_called_once()


def test_run_tracer_record_tool_error() -> None:
    tracer, _ = _make_tracer()
    mock_turn_span = MagicMock()
    mock_tool_span = MagicMock()
    tracer._root.start_observation.return_value = mock_turn_span
    mock_turn_span.start_observation.return_value = mock_tool_span

    tracer.start_turn(1, 2)
    tracer.record_tool_error(1, "create_node", {}, "Invalid provenance")

    call_kwargs = mock_turn_span.start_observation.call_args[1]
    assert call_kwargs["level"] == "ERROR"
    assert call_kwargs["status_message"] == "Invalid provenance"
    assert "metadata" not in call_kwargs
    mock_tool_span.end.assert_called_once()


def test_run_tracer_finalize_flushes() -> None:
    tracer, mock_lf = _make_tracer()
    trace = make_test_trace()
    trace.total_usage = TurnUsage(
        prompt_tokens=100, completion_tokens=50, total_tokens=150
    )
    trace.status = "completed"
    trace.completion_payload = {"summary": "done"}

    tracer.finalize(trace)

    tracer._root.set_trace_io.assert_called_once_with(output={"summary": "done"})
    tracer._root.update.assert_called_once()
    update_kwargs = tracer._root.update.call_args[1]
    assert update_kwargs["metadata"]["status"] == "completed"
    tracer._root.end.assert_called_once()
    mock_lf.flush.assert_called_once()


# ---------------------------------------------------------------------------
# actions.py
# ---------------------------------------------------------------------------


def test_actions_contains_all_expected_entries() -> None:
    expected = {
        "search_graph",
        "get_node",
        "get_node_neighborhood",
        "get_session",
        "list_sessions",
        "get_theme",
        "create_node",
        "update_node",
        "delete_node",
        "create_edge",
        "update_edge",
        "delete_edge",
        "create_theme",
        "update_theme",
        "retire_theme",
        "set_preface",
        "complete",
        "complete_theme_update",
    }
    assert expected == set(reg.ACTIONS.keys())


# ---------------------------------------------------------------------------
# runner.py — unit tests (no FalkorDB)
# ---------------------------------------------------------------------------


def _mock_run_tracer() -> MagicMock:
    """Mock RunTracer to avoid real Langfuse calls in runner unit tests."""
    mock = MagicMock(spec=RunTracer)
    mock.get_trace_id.return_value = "mock-trace-id"
    return mock


def test_run_fails_without_tool_call() -> None:
    with (
        patch("litellm.completion", return_value=mock_text_response()),
        patch("weavy.harness.runner.RunTracer", return_value=_mock_run_tracer()),
    ):
        trace = run(
            mode="ingestion",
            system_prompt="You are an agent.",
            initial_messages=[{"role": "user", "content": "process s:1"}],
            allowed_actions=reg.SESSION_ACTIONS,
            run_context={"input_summary": "s:1"},
            graph=MagicMock(),
        )
    assert trace.status == "failed"
    assert "Model stopped without calling a completion tool." in (trace.error or "")


def test_run_fails_on_unknown_tool() -> None:
    resp = mock_tool_response("nonexistent_tool", {})
    with (
        patch("litellm.completion", return_value=resp),
        patch("weavy.harness.runner.RunTracer", return_value=_mock_run_tracer()),
    ):
        trace = run(
            mode="ingestion",
            system_prompt="You are an agent.",
            initial_messages=[],
            allowed_actions=reg.SESSION_ACTIONS,
            run_context={"input_summary": "test"},
            graph=MagicMock(),
        )
    assert trace.status == "failed"
    assert "Unknown tool" in (trace.error or "")
    assert trace.turns[0].tool_calls[0].error is not None


def test_run_fails_on_bad_args() -> None:
    resp = mock_tool_response("search_graph", {"bad_field": 999})
    with (
        patch("litellm.completion", return_value=resp),
        patch("weavy.harness.runner.RunTracer", return_value=_mock_run_tracer()),
    ):
        trace = run(
            mode="ingestion",
            system_prompt="You are an agent.",
            initial_messages=[],
            allowed_actions=reg.SESSION_ACTIONS,
            run_context={"input_summary": "test"},
            graph=MagicMock(),
        )
    assert trace.status == "failed"
    assert "Invalid arguments for 'search_graph'" in (trace.error or "")


def test_run_records_tool_calls() -> None:
    search_resp = mock_tool_response(
        "search_graph", {"query": "career", "limit": 5}, "tc-1"
    )
    completion_resp = mock_tool_response("complete", {"summary": "done"}, "tc-2")

    search_output = MagicMock()
    search_output.model_dump_json.return_value = '{"results":[]}'

    with (
        patch("litellm.completion", side_effect=[search_resp, completion_resp]),
        patch("weavy.store.graph.search_graph", return_value=search_output),
        patch("weavy.harness.runner.RunTracer", return_value=_mock_run_tracer()),
    ):
        trace = run(
            mode="ingestion",
            system_prompt="You are an agent.",
            initial_messages=[],
            allowed_actions=reg.SESSION_ACTIONS,
            run_context={"input_summary": "test"},
            graph=MagicMock(),
        )

    assert trace.status == "completed"
    assert len(trace.turns) == 2
    assert trace.turns[0].tool_calls[0].tool_name == "search_graph"
    assert trace.turns[1].tool_calls[0].tool_name == "complete"
    assert "tool_call_id" not in json.dumps(
        [turn.model_dump(mode="json") for turn in trace.turns]
    )


def test_run_redacts_tool_call_ids_from_trace_and_tracer() -> None:
    search_resp = mock_tool_response(
        "search_graph", {"query": "career", "limit": 5}, "provider-tool-call-id-1"
    )
    completion_resp = mock_tool_response(
        "complete", {"summary": "done"}, "provider-tool-call-id-2"
    )

    search_output = MagicMock()
    search_output.model_dump_json.return_value = '{"results":[]}'
    tracer = _mock_run_tracer()

    with (
        patch("litellm.completion", side_effect=[search_resp, completion_resp]),
        patch("weavy.store.graph.search_graph", return_value=search_output),
        patch("weavy.harness.runner.RunTracer", return_value=tracer),
    ):
        trace = run(
            mode="ingestion",
            system_prompt="You are an agent.",
            initial_messages=[],
            allowed_actions=reg.SESSION_ACTIONS,
            run_context={"input_summary": "test"},
            graph=MagicMock(),
        )

    assert trace.status == "completed"
    first_llm_call = tracer.record_llm_response.call_args_list[0].kwargs
    assert first_llm_call["tool_calls"] == [
        {"name": "search_graph", "args": '{"query": "career", "limit": 5}'}
    ]
    # trace.conversation keeps tool_call_ids for session replay;
    # turns are sanitized before being sent to Langfuse.
    turns_data = json.dumps([turn.model_dump(mode="json") for turn in trace.turns])
    assert "tool_call_id" not in turns_data


def test_run_completes_on_completion_tool() -> None:
    resp = mock_tool_response("complete", {"summary": "all done"})
    with (
        patch("litellm.completion", return_value=resp),
        patch("weavy.harness.runner.RunTracer", return_value=_mock_run_tracer()),
    ):
        trace = run(
            mode="ingestion",
            system_prompt="system",
            initial_messages=[],
            allowed_actions=reg.SESSION_ACTIONS,
            run_context={"input_summary": "test"},
            graph=MagicMock(),
        )
    assert trace.status == "completed"
    assert trace.completion_payload["summary"] == "all done"
    assert trace.ended_at is not None


def test_run_fails_on_model_exception() -> None:
    with (
        patch("litellm.completion", side_effect=RuntimeError("network error")),
        patch("weavy.harness.runner.RunTracer", return_value=_mock_run_tracer()),
    ):
        trace = run(
            mode="query",
            system_prompt="system",
            initial_messages=[],
            allowed_actions=reg.SESSION_ACTIONS,
            run_context={"input_summary": "test"},
            graph=MagicMock(),
        )
    assert trace.status == "failed"
    assert "network error" in (trace.error or "")


def test_run_caches_context_limit_lookup() -> None:
    harness_runner._get_context_limit.cache_clear()
    resp = mock_tool_response("complete", {"summary": "all done"})

    with (
        patch(
            "litellm.get_model_info", return_value={"max_input_tokens": 123}
        ) as mock_info,
        patch("litellm.completion", return_value=resp),
        patch("weavy.harness.runner.RunTracer", return_value=_mock_run_tracer()),
    ):
        for _ in range(2):
            trace = run(
                mode="ingestion",
                system_prompt="system",
                initial_messages=[],
                allowed_actions=reg.SESSION_ACTIONS,
                run_context={"input_summary": "test"},
                graph=MagicMock(),
            )
            assert trace.status == "completed"

    assert mock_info.call_count == 1
    harness_runner._get_context_limit.cache_clear()


# ---------------------------------------------------------------------------
# runner.py — integration test (uses FalkorDB)
# ---------------------------------------------------------------------------


def test_run_with_graph_write_records_touched_nodes(graph: Graph) -> None:
    """Runner executes a create_node call and the trace captures the touched node."""
    from weavy.models.graph import ProvenanceInput

    prov = ProvenanceInput(source_id="s:1")
    create_args = {
        "aliases": ["test concept"],
        "summary": "A test concept node.",
        "note": "created during harness test",
        "provenance": prov.model_dump(),
    }
    completion_args = {"summary": "done"}

    create_resp = mock_tool_response("create_node", create_args, "tc-1")
    completion_resp = mock_tool_response("complete", completion_args, "tc-2")

    with (
        patch("litellm.completion", side_effect=[create_resp, completion_resp]),
        patch("weavy.harness.runner.RunTracer", return_value=_mock_run_tracer()),
    ):
        trace = run(
            mode="ingestion",
            system_prompt="You are an agent.",
            initial_messages=[{"role": "user", "content": "process"}],
            allowed_actions=reg.SESSION_ACTIONS,
            run_context={"input_summary": "s:1"},
            graph=graph,
        )

    assert trace.status == "completed"
    assert len(trace.touched_nodes) == 1
    assert trace.touched_nodes[0].action == "created"
    assert trace.touched_nodes[0].node_id.startswith("node:")


def test_run_persists_completed_trace_locally(graph: Graph) -> None:
    completion_resp = mock_tool_response(
        "complete",
        {"summary": "done"},
        usage={"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
    )

    with (
        patch("litellm.completion", return_value=completion_resp),
        patch("weavy.harness.runner.RunTracer", return_value=_mock_run_tracer()),
    ):
        trace = run(
            mode="ingestion",
            system_prompt="system",
            initial_messages=[],
            allowed_actions=reg.SESSION_ACTIONS,
            run_context={"input_summary": "trace test"},
            graph=graph,
            session_id="s:trace",
        )

    stored = get_trace(graph, trace.run_id)
    assert stored.status == "completed"
    assert stored.run_id == trace.run_id
    assert stored.session_id == "s:trace"
    assert stored.completion_payload
    assert stored.completion_payload["summary"] == "done"
    assert stored.total_usage.total_tokens == 15


def test_run_persists_failed_trace_locally(graph: Graph) -> None:
    with (
        patch("litellm.completion", side_effect=RuntimeError("network error")),
        patch("weavy.harness.runner.RunTracer", return_value=_mock_run_tracer()),
    ):
        trace = run(
            mode="query",
            system_prompt="system",
            initial_messages=[],
            allowed_actions=reg.SESSION_ACTIONS,
            run_context={"input_summary": "failed trace test"},
            graph=graph,
        )

    stored = get_trace(graph, trace.run_id)
    assert stored.status == "failed"
    assert "network error" in (stored.error or "")
