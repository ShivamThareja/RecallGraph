from datetime import datetime, timezone

import pytest

from weavy.models.graph import ProvenanceInput
from weavy.models.traces import RunTrace
from weavy.services import memory
from weavy.services.backup import export_backup, import_backup
from weavy.store import canonical as store_canonical
from weavy.store import graph as store_graph
from weavy.store import themes as store_themes
from weavy.store.client import delete_graph_if_exists, get_graph
from weavy.store.system import get_system, increment_counter, set_preface
from weavy.store.traces import get_trace, persist_trace
from tests.helpers import (
    SAMPLE_TEXT,
    TEST_GRAPH,
    make_test_trace,
    reset_test_graph,
    store_test_session,
)


@pytest.fixture
def graph():
    return reset_test_graph("Session", "Theme")


def _populate_graph(graph) -> dict[str, str]:
    set_preface(graph, "Backup test graph")
    session_id = store_test_session(graph, SAMPLE_TEXT)
    store_canonical.persist_session_outcomes(
        graph,
        session_id,
        "Ingested backup fixture.",
        {"nodes_created": ["node:1"]},
        "2026-01-01T00:00:00+00:00",
    )

    trace = make_test_trace()
    node_a = memory.create_node(
        graph,
        aliases=["backup concept"],
        summary="A concept used to test backups.",
        note="created for backup test",
        provenance=ProvenanceInput(source_id=session_id),
        trace=trace,
    ).id
    node_b = memory.create_node(
        graph,
        aliases=["restore target"],
        summary="A target used to test restore edges.",
        note="created for backup test",
        provenance=ProvenanceInput(source_id=session_id),
        trace=trace,
    ).id
    edge_id = memory.create_edge(
        graph,
        from_node_id=node_a,
        to_node_id=node_b,
        label="backs up to",
        fact="Backing up node_a produces node_b.",
        note="backup relationship",
        provenance=ProvenanceInput(source_id=session_id),
        trace=trace,
        source_id=session_id,
    ).id
    store_themes.create_theme(
        graph,
        "Backup Theme",
        "The graph can be exported and restored.",
        [node_a],
    )
    run_trace = RunTrace(
        mode="query",
        started_at=datetime.now(tz=timezone.utc),
        input_summary="backup trace",
        status="completed",
        session_id=session_id,
        completion_payload={"answer": "ok"},
    )
    persist_trace(graph, run_trace)
    return {
        "session_id": session_id,
        "node_a": node_a,
        "node_b": node_b,
        "edge_id": edge_id,
        "run_id": run_trace.run_id,
    }


def test_export_import_round_trips_complete_graph(graph, tmp_path) -> None:
    ids = _populate_graph(graph)
    backup_path = tmp_path / "backup.json"

    exported = export_backup(graph, backup_path, graph_name=TEST_GRAPH)
    assert exported.sessions == 1
    assert exported.semantic_nodes == 2
    assert exported.semantic_edges == 1
    assert exported.themes == 1
    assert exported.run_traces == 1

    imported = import_backup(graph, backup_path, replace=True, graph_name=TEST_GRAPH)
    assert imported == exported

    state = get_system(graph)
    assert state.preface == "Backup test graph"
    assert increment_counter(graph, "session") == "s:2"
    assert increment_counter(graph, "node") == "node:3"
    assert increment_counter(graph, "edge") == "edge:2"

    session = store_canonical.get_session(graph, ids["session_id"])
    assert session.messages[0].content == SAMPLE_TEXT

    node = store_graph.get_node(graph, ids["node_a"])
    assert node.node.aliases == ["backup concept"]
    assert node.edges[0].id == ids["edge_id"]

    theme = store_themes.get_theme(graph, "Backup Theme")
    assert theme.anchors == [ids["node_a"]]

    trace = get_trace(graph, ids["run_id"])
    assert trace.session_id == ids["session_id"]
    assert trace.completion_payload == {"answer": "ok"}


def test_import_refuses_non_empty_graph_without_replace(graph, tmp_path) -> None:
    _populate_graph(graph)
    backup_path = tmp_path / "backup.json"
    export_backup(graph, backup_path, graph_name=TEST_GRAPH)

    with pytest.raises(RuntimeError, match="not empty"):
        import_backup(graph, backup_path)

    assert store_themes.get_theme(graph, "Backup Theme").name == "Backup Theme"


def test_import_into_empty_graph_and_restored_search(graph, tmp_path) -> None:
    _populate_graph(graph)
    backup_path = tmp_path / "backup.json"
    export_backup(graph, backup_path, graph_name=TEST_GRAPH)

    target_graph_name = f"{TEST_GRAPH}_backup_import"
    target = get_graph(target_graph_name)
    delete_graph_if_exists(target)

    import_backup(target, backup_path, graph_name=target_graph_name)

    results = store_graph.search_graph(
        target,
        query="backup",
        query_embedding=[0.1] * 8,
        limit=5,
    )
    assert [result.id for result in results.results]
