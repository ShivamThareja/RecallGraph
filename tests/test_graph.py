"""
Phase 3 tests — Semantic graph CRUD with provenance validation and log management.
Requires a running FalkorDB instance (provided by devenv up).
Uses the "weavy_test" graph to avoid touching the main graph.
"""

import json
from datetime import datetime

import pytest
from falkordb import Graph
from pydantic import ValidationError

from weavy.harness.tool_models import (
    CreateEdgeInput,
    CreateNodeInput,
    UpdateEdgeInput,
    UpdateNodeInput,
)
from weavy.models.graph import LogEntry, ProvenanceInput
from weavy.services import memory
from weavy.store import graph as store_graph
from weavy.timeutil import to_utc
from tests.helpers import (
    make_test_trace,
    reset_test_graph,
    store_test_session_with_id,
)


@pytest.fixture
def graph() -> Graph:
    return reset_test_graph()


# ---------------------------------------------------------------------------
# create_node
# ---------------------------------------------------------------------------


def test_create_node_ingestion(graph: Graph) -> None:
    prov = ProvenanceInput(source_id="s:1")
    params = CreateNodeInput(
        aliases=["career anxiety", "work stress"],
        summary="Persistent worry about career direction.",
        note="First mention in s:1",
        provenance=prov,
    )
    result = memory.create_node(
        graph,
        aliases=params.aliases,
        summary=params.summary,
        note=params.note,
        provenance=params.provenance,
        trace=make_test_trace(),
    )
    assert result.ok
    assert result.id is not None
    assert result.id.startswith("node:")

    out = store_graph.get_node(graph, result.id)
    assert out.node.aliases == ["career anxiety", "work stress"]
    assert out.node.summary == "Persistent worry about career direction."
    assert out.node.total_log_count == 1
    assert len(out.node.log) == 1
    entry = out.node.log[0]
    assert isinstance(entry, LogEntry)
    assert entry.source_id == "s:1"
    assert entry.note == "First mention in s:1"


def test_create_node_query_provenance(graph: Graph) -> None:
    prov = ProvenanceInput(source_id="s:2")
    params = CreateNodeInput(
        aliases=["fear of failure"],
        summary="Fear of failing after a career change.",
        note="User mentioned during chat at message 3",
        provenance=prov,
    )
    result = memory.create_node(
        graph,
        aliases=params.aliases,
        summary=params.summary,
        note=params.note,
        provenance=params.provenance,
        trace=make_test_trace("query"),
    )
    assert result.ok
    assert result.id.startswith("node:")


def test_create_node_no_provenance_rejected() -> None:
    with pytest.raises(ValidationError):
        CreateNodeInput(
            aliases=["something"],
            summary="Some concept.",
            note="no provenance",
        )


def test_create_node_recorded_in_trace(graph: Graph) -> None:
    prov = ProvenanceInput(source_id="s:1")
    params = CreateNodeInput(
        aliases=["focus"],
        summary="Ability to focus deeply.",
        note="first mention",
        provenance=prov,
    )
    trace = make_test_trace()
    result = memory.create_node(
        graph,
        aliases=params.aliases,
        summary=params.summary,
        note=params.note,
        provenance=params.provenance,
        trace=trace,
    )
    assert len(trace.touched_nodes) == 1
    assert trace.touched_nodes[0].node_id == result.id
    assert trace.touched_nodes[0].action == "created"


# ---------------------------------------------------------------------------
# update_node
# ---------------------------------------------------------------------------


def test_update_node_archives_summary(graph: Graph) -> None:
    prov = ProvenanceInput(source_id="s:1")
    node_id = memory.create_node(
        graph,
        aliases=["anxiety"],
        summary="Original summary.",
        note="created",
        provenance=prov,
        trace=make_test_trace(),
    ).id

    update_prov = ProvenanceInput(source_id="s:2")
    memory.update_node(
        graph,
        node_id=node_id,
        note="Feeling resolved after therapy.",
        new_summary="Anxiety reduced significantly.",
        provenance=update_prov,
        trace=make_test_trace(),
    )

    out = store_graph.get_node(graph, node_id)
    assert out.node.summary == "Anxiety reduced significantly."
    assert out.node.total_log_count == 2
    # Last log entry (hot, most recent) should contain archived summary
    hot_entry = out.node.log[-1]
    assert isinstance(hot_entry, LogEntry)
    assert "[archived summary] Original summary." in hot_entry.note
    assert "Feeling resolved after therapy." in hot_entry.note


def test_update_node_aliases_only(graph: Graph) -> None:
    prov = ProvenanceInput(source_id="s:1")
    node_id = memory.create_node(
        graph,
        aliases=["dad"],
        summary="Relationship with father.",
        note="created",
        provenance=prov,
        trace=make_test_trace(),
    ).id

    update_prov = ProvenanceInput(source_id="s:2")
    memory.update_node(
        graph,
        node_id=node_id,
        note="User now refers to him as 'papa'.",
        new_aliases=["dad", "papa", "father"],
        provenance=update_prov,
        trace=make_test_trace(),
    )

    out = store_graph.get_node(graph, node_id)
    assert out.node.aliases == ["dad", "papa", "father"]
    assert out.node.summary == "Relationship with father."  # unchanged
    assert out.node.total_log_count == 2


def test_node_and_edge_ids_reject_trailing_punctuation() -> None:
    with pytest.raises(ValidationError, match=r"node:\d+"):
        UpdateNodeInput(
            node_id="node:4,",
            note="bad id",
            provenance=ProvenanceInput(source_id="s:1"),
        )

    with pytest.raises(ValidationError, match=r"node:\d+"):
        CreateEdgeInput(
            from_node_id="node:1",
            to_node_id="node:2.",
            label="related",
            note="x",
        )

    with pytest.raises(ValidationError, match=r"edge:\d+"):
        UpdateEdgeInput(edge_id="edge:3)", new_label="updated")


# ---------------------------------------------------------------------------
# delete_node
# ---------------------------------------------------------------------------


def test_delete_node_removes_node_and_edges(graph: Graph) -> None:
    prov = ProvenanceInput(source_id="s:1")
    a = memory.create_node(
        graph,
        aliases=["A"],
        summary="Node A",
        note="a",
        provenance=prov,
        trace=make_test_trace(),
    ).id
    b = memory.create_node(
        graph,
        aliases=["B"],
        summary="Node B",
        note="b",
        provenance=prov,
        trace=make_test_trace(),
    ).id
    memory.create_edge(
        graph,
        from_node_id=a,
        to_node_id=b,
        label="related",
        fact="A is related to B",
        note="A relates to B",
        provenance=prov,
        trace=make_test_trace(),
    )

    trace = make_test_trace()
    memory.delete_node(graph, node_id=a, trace=trace)
    assert trace.touched_nodes[-1].action == "deleted"

    # Node A gone
    with pytest.raises(ValueError):
        store_graph.get_node(graph, a)

    # Node B still exists, edge gone
    out_b = store_graph.get_node(graph, b)
    assert out_b.edges == []


def test_delete_node_not_found_raises(graph: Graph) -> None:
    with pytest.raises(ValueError):
        memory.delete_node(graph, node_id="node:9999", trace=make_test_trace())


# ---------------------------------------------------------------------------
# create_edge / update_edge / delete_edge
# ---------------------------------------------------------------------------


def test_create_edge(graph: Graph) -> None:
    prov = ProvenanceInput(source_id="s:1")
    a = memory.create_node(
        graph,
        aliases=["A"],
        summary="A",
        note="a",
        provenance=prov,
        trace=make_test_trace(),
    ).id
    b = memory.create_node(
        graph,
        aliases=["B"],
        summary="B",
        note="b",
        provenance=prov,
        trace=make_test_trace(),
    ).id

    trace = make_test_trace()
    result = memory.create_edge(
        graph,
        from_node_id=a,
        to_node_id=b,
        label="causes",
        fact="A causes B",
        note="A causes B",
        provenance=prov,
        trace=trace,
    )
    assert result.ok
    assert result.id.startswith("edge:")
    assert trace.touched_edges[0].action == "created"

    out = store_graph.get_node(graph, a)
    assert len(out.edges) == 1
    assert out.edges[0].label == "causes"
    assert out.edges[0].fact == "A causes B"
    assert out.edges[0].total_log_count == 1
    assert out.edges[0].log[0].note == "A causes B"


def test_update_edge_label(graph: Graph) -> None:
    prov = ProvenanceInput(source_id="s:1")
    a = memory.create_node(
        graph,
        aliases=["A"],
        summary="A",
        note="a",
        provenance=prov,
        trace=make_test_trace(),
    ).id
    b = memory.create_node(
        graph,
        aliases=["B"],
        summary="B",
        note="b",
        provenance=prov,
        trace=make_test_trace(),
    ).id
    edge_id = memory.create_edge(
        graph,
        from_node_id=a,
        to_node_id=b,
        label="old label",
        fact="A had old relationship to B",
        note="initial relationship",
        provenance=prov,
        trace=make_test_trace(),
    ).id

    memory.update_edge(
        graph,
        edge_id=edge_id,
        note="relationship changed",
        new_label="new label",
        new_fact="A now has new relationship to B",
        provenance=ProvenanceInput(source_id="s:2"),
        trace=make_test_trace(),
    )

    out = store_graph.get_node(graph, a)
    assert out.edges[0].label == "new label"
    assert out.edges[0].fact == "A now has new relationship to B"
    assert out.edges[0].total_log_count == 2
    assert out.edges[0].log[-1].note == "relationship changed"


def test_delete_edge(graph: Graph) -> None:
    prov = ProvenanceInput(source_id="s:1")
    a = memory.create_node(
        graph,
        aliases=["A"],
        summary="A",
        note="a",
        provenance=prov,
        trace=make_test_trace(),
    ).id
    b = memory.create_node(
        graph,
        aliases=["B"],
        summary="B",
        note="b",
        provenance=prov,
        trace=make_test_trace(),
    ).id
    edge_id = memory.create_edge(
        graph,
        from_node_id=a,
        to_node_id=b,
        label="linked",
        fact="A is linked to B",
        note="A linked to B",
        provenance=prov,
        trace=make_test_trace(),
    ).id

    trace = make_test_trace()
    memory.delete_edge(graph, edge_id=edge_id, trace=trace)
    assert trace.touched_edges[-1].action == "deleted"

    # Both nodes remain, edge gone
    out_a = store_graph.get_node(graph, a)
    assert out_a.edges == []


# ---------------------------------------------------------------------------
# search_graph
# ---------------------------------------------------------------------------


def test_search_graph_alias_match(graph: Graph) -> None:
    prov = ProvenanceInput(source_id="s:1")
    memory.create_node(
        graph,
        aliases=["career anxiety", "work stress"],
        summary="Anxiety about career.",
        note="x",
        provenance=prov,
        trace=make_test_trace(),
    )
    out = store_graph.search_graph(graph, query="career")
    assert len(out.results) >= 1
    assert any(r.kind == "node" and "career" in r.label.lower() for r in out.results)


def test_search_graph_summary_match(graph: Graph) -> None:
    prov = ProvenanceInput(source_id="s:1")
    memory.create_node(
        graph,
        aliases=["meditation"],
        summary="Daily mindfulness practice helping anxiety.",
        note="x",
        provenance=prov,
        trace=make_test_trace(),
    )
    out = store_graph.search_graph(graph, query="mindfulness")
    assert len(out.results) >= 1


def test_search_graph_no_match(graph: Graph) -> None:
    out = store_graph.search_graph(graph, query="xyzzy_impossible_query_12345")
    assert out.results == []


# ---------------------------------------------------------------------------
# get_node
# ---------------------------------------------------------------------------


def test_get_node_returns_edges(graph: Graph) -> None:
    prov = ProvenanceInput(source_id="s:1")
    a = memory.create_node(
        graph,
        aliases=["A"],
        summary="A",
        note="a",
        provenance=prov,
        trace=make_test_trace(),
    ).id
    b = memory.create_node(
        graph,
        aliases=["B"],
        summary="B",
        note="b",
        provenance=prov,
        trace=make_test_trace(),
    ).id
    memory.create_edge(
        graph,
        from_node_id=a,
        to_node_id=b,
        label="test edge",
        fact="A has a test edge to B",
        note="A test edge from A to B",
        provenance=prov,
        trace=make_test_trace(),
    )

    out = memory.get_node(graph, node_ids=[a])
    assert len(out.results[0].edges) == 1
    assert out.results[0].edges[0].label == "test edge"
    assert out.results[0].edges[0].fact == "A has a test edge to B"


def test_get_node_returns_multiple_results_and_not_found(graph: Graph) -> None:
    prov = ProvenanceInput(source_id="s:1")
    a = memory.create_node(
        graph,
        aliases=["A"],
        summary="A",
        note="a",
        provenance=prov,
        trace=make_test_trace(),
    ).id
    b = memory.create_node(
        graph,
        aliases=["B"],
        summary="B",
        note="b",
        provenance=prov,
        trace=make_test_trace(),
    ).id

    out = memory.get_node(graph, node_ids=[a, "node:9999", b])

    assert [result.node.id for result in out.results] == [a, b]
    assert out.not_found == ["node:9999"]


def test_get_node_json_has_iso_timestamps(graph: Graph) -> None:
    prov = ProvenanceInput(source_id="s:1")
    node_id = memory.create_node(
        graph,
        aliases=["test node"],
        summary="Test.",
        note="initial",
        provenance=prov,
        trace=make_test_trace(),
    ).id

    out = memory.get_node(graph, node_ids=[node_id])
    payload = json.loads(out.model_dump_json())
    ts = payload["results"][0]["node"]["log"][0]["timestamp"]
    datetime.fromisoformat(ts)


# ---------------------------------------------------------------------------
# get_node_neighborhood
# ---------------------------------------------------------------------------


def test_get_node_neighborhood(graph: Graph) -> None:
    prov = ProvenanceInput(source_id="s:1")
    a = memory.create_node(
        graph,
        aliases=["career"],
        summary="Career direction concerns.",
        note="a",
        provenance=prov,
        trace=make_test_trace(),
    ).id
    b = memory.create_node(
        graph,
        aliases=["anxiety"],
        summary="Persistent anxiety.",
        note="b",
        provenance=prov,
        trace=make_test_trace(),
    ).id
    c = memory.create_node(
        graph,
        aliases=["mentor"],
        summary="Senior mentor at work.",
        note="c",
        provenance=prov,
        trace=make_test_trace(),
    ).id
    memory.create_edge(
        graph,
        from_node_id=a,
        to_node_id=b,
        label="causes",
        fact="Career anxiety causes persistent anxiety",
        note="Career anxiety causes persistent anxiety",
        provenance=prov,
        trace=make_test_trace(),
    )
    memory.create_edge(
        graph,
        from_node_id=c,
        to_node_id=a,
        label="influences",
        fact="Mentor influences career direction",
        note="Mentor influences career direction",
        provenance=prov,
        trace=make_test_trace(),
    )

    out = store_graph.get_node_neighborhood(graph, a)
    assert out.node.id == a
    assert out.node.summary == "Career direction concerns."
    neighbor_ids = {n.node_id for n in out.neighbors}
    assert b in neighbor_ids
    assert c in neighbor_ids
    # Neighbors expose the edge fact, not a bare note.
    assert any(n.edge_fact for n in out.neighbors)


# ---------------------------------------------------------------------------
# Edge-fact search (unified ranked results) and MENTIONS provenance
# ---------------------------------------------------------------------------


def test_search_returns_edge_facts(graph: Graph) -> None:
    """Keyword search surfaces edge facts as kind='edge' rows alongside nodes."""
    prov = ProvenanceInput(source_id="s:1")
    a = memory.create_node(
        graph,
        aliases=["Ada"],
        summary="A person.",
        note="x",
        provenance=prov,
        trace=make_test_trace(),
    ).id
    b = memory.create_node(
        graph,
        aliases=["Acme"],
        summary="A company.",
        note="x",
        provenance=prov,
        trace=make_test_trace(),
    ).id
    memory.create_edge(
        graph,
        from_node_id=a,
        to_node_id=b,
        label="works at",
        fact="Ada works at Acme as a staff engineer",
        note="employment",
        provenance=prov,
        trace=make_test_trace(),
    )

    out = store_graph.search_graph(graph, query="staff engineer")
    edge_hits = [r for r in out.results if r.kind == "edge"]
    assert len(edge_hits) >= 1
    assert "staff engineer" in edge_hits[0].text
    assert edge_hits[0].endpoints == [a, b]


def test_create_node_links_mention(graph: Graph) -> None:
    """A node written during a session is reachable from that session via MENTIONS."""
    store_test_session_with_id(graph, "s:1")
    node_id = memory.create_node(
        graph,
        aliases=["focus"],
        summary="Deep focus.",
        note="x",
        provenance=ProvenanceInput(source_id="s:1"),
        trace=make_test_trace(),
        session_id="s:1",
    ).id

    out = memory.get_node(graph, node_ids=[node_id])
    assert out.results[0].mentioned_by == ["s:1"]


# ---------------------------------------------------------------------------
# Temporal — two clocks: timestamp (recorded) vs happened_at (valid time)
# ---------------------------------------------------------------------------


def test_happened_at_distinct_from_record_time(graph: Graph) -> None:
    """happened_at (when it was true) is stored independently of the episode time."""
    recorded = datetime(2023, 5, 8)  # session date — when discussed
    happened = datetime(2023, 5, 7)  # "yesterday" — when it actually happened
    node_id = memory.create_node(
        graph,
        aliases=["support group visit"],
        summary="Caroline attended the LGBTQ support group.",
        note="mentioned as happening yesterday",
        provenance=ProvenanceInput(source_id="s:1"),
        trace=make_test_trace(),
        event_time=recorded,
        happened_at=happened,
    ).id

    entry = store_graph.get_node(graph, node_id).node.log[0]
    # Naive inputs are stored as UTC (see weavy/timeutil.py).
    assert entry.timestamp == to_utc(recorded)
    assert entry.happened_at == to_utc(happened)


def test_happened_at_defaults_to_event_time(graph: Graph) -> None:
    """With no temporal cue, happened_at falls back to the episode time."""
    recorded = datetime(2023, 5, 8)
    node_id = memory.create_node(
        graph,
        aliases=["plain fact"],
        summary="A fact with no stated event date.",
        note="no temporal cue",
        provenance=ProvenanceInput(source_id="s:1"),
        trace=make_test_trace(),
        event_time=recorded,
    ).id

    entry = store_graph.get_node(graph, node_id).node.log[0]
    assert entry.happened_at == to_utc(recorded)
    assert entry.timestamp == to_utc(recorded)


def test_edge_happened_at_round_trips(graph: Graph) -> None:
    prov = ProvenanceInput(source_id="s:1")
    a = memory.create_node(
        graph,
        aliases=["A"],
        summary="A",
        note="a",
        provenance=prov,
        trace=make_test_trace(),
    ).id
    b = memory.create_node(
        graph,
        aliases=["B"],
        summary="B",
        note="b",
        provenance=prov,
        trace=make_test_trace(),
    ).id
    memory.create_edge(
        graph,
        from_node_id=a,
        to_node_id=b,
        label="attended",
        fact="A attended B",
        note="happened the day before it was mentioned",
        provenance=prov,
        trace=make_test_trace(),
        event_time=datetime(2023, 5, 8),
        happened_at=datetime(2023, 5, 7),
    )

    edge = store_graph.get_node(graph, a).edges[0]
    assert edge.log[0].timestamp == to_utc(datetime(2023, 5, 8))
    assert edge.log[0].happened_at == to_utc(datetime(2023, 5, 7))


def test_search_time_range_filters_on_happened_at(graph: Graph) -> None:
    """time_range matches valid time (happened_at), not when the fact was recorded."""
    prov = ProvenanceInput(source_id="s:1")
    # Discussed in August, but happened in May → inside a May window.
    memory.create_node(
        graph,
        aliases=["alpha widget"],
        summary="A widget event that happened in May.",
        note="x",
        provenance=prov,
        trace=make_test_trace(),
        event_time=datetime(2023, 8, 1),
        happened_at=datetime(2023, 5, 7),
    )
    # Discussed in May, but happened in January → outside a May window.
    memory.create_node(
        graph,
        aliases=["beta widget"],
        summary="A widget event that happened in January.",
        note="x",
        provenance=prov,
        trace=make_test_trace(),
        event_time=datetime(2023, 5, 15),
        happened_at=datetime(2023, 1, 1),
    )

    window = [datetime(2023, 5, 1), datetime(2023, 5, 31)]
    out = store_graph.search_graph(graph, query="widget", time_range=window)
    labels = {r.label for r in out.results}
    assert "alpha widget" in labels
    assert "beta widget" not in labels
