"""Navigational retrieval (graph -> episode) and the create_node duplicate guard."""

import pytest

from tests.conftest import _fake_embed
from tests.helpers import reset_test_graph
from weavy.application.session_runs import create_session
from weavy.harness.tracing import new_trace
from weavy.models.graph import ProvenanceInput
from weavy.models.traces import RunTrace
from weavy.services import memory
from weavy.store import graph as store_graph


@pytest.fixture
def graph():
    yield reset_test_graph("Session")


def _trace() -> RunTrace:
    return new_trace("ingestion", "")


def _provenance() -> ProvenanceInput:
    return ProvenanceInput(source_id="s:1")


# --- search_graph returns only nodes/edges --------------------------------


def test_search_graph_never_returns_episode_kind(graph):
    session_id = create_session("Ada: I adopted a guinea pig named Oscar.", graph)
    memory.create_node(
        graph,
        aliases=["Oscar"],
        summary="A guinea pig Ada adopted.",
        note="n",
        provenance=_provenance(),
        trace=_trace(),
        session_id=session_id,
    )

    out = memory.search_graph(graph, query="guinea pig named Oscar")
    assert out.results
    assert all(r.kind in ("node", "edge") for r in out.results)


def test_query_session_without_text_creates_no_search_surface(graph):
    session_id = create_session("", graph)
    # No node/edge exists for this empty session, so nothing to find — but
    # critically, no chunk/episode search surface was created for it either.
    out = memory.search_graph(graph, query=session_id)
    assert out.results == []


# --- ground truth is reachable only by navigation -------------------------


def test_episode_reachable_via_mentioned_by_then_get_session(graph):
    text = "Ada: I adopted a guinea pig named Oscar."
    session_id = create_session(text, graph)
    created = memory.create_node(
        graph,
        aliases=["Oscar"],
        summary="A guinea pig.",
        note="n",
        provenance=_provenance(),
        trace=_trace(),
        session_id=session_id,
    )
    assert created.ok

    node_result = memory.get_node(graph, node_ids=[created.id]).results[0]
    assert session_id in node_result.mentioned_by

    from weavy.store import canonical as store_canonical

    episode = store_canonical.get_session_output(graph, session_id)
    assert "Oscar" in episode.text


# --- create_node duplicate guard ------------------------------------------


def test_create_node_refuses_alias_duplicate(graph):
    first = memory.create_node(
        graph,
        aliases=["Melanie"],
        summary="Melanie is a potter.",
        note="n",
        provenance=_provenance(),
        trace=_trace(),
    )
    assert first.ok

    dup = memory.create_node(
        graph,
        aliases=["melanie"],
        summary="A supportive parent who paints.",
        note="n",
        provenance=_provenance(),
        trace=_trace(),
    )
    assert not dup.ok
    assert first.id in (dup.message or "")


def test_create_node_force_overrides_guard(graph):
    memory.create_node(
        graph,
        aliases=["Alex"],
        summary="Alex the engineer.",
        note="n",
        provenance=_provenance(),
        trace=_trace(),
    )
    forced = memory.create_node(
        graph,
        aliases=["Alex"],
        summary="A different Alex — a chef.",
        note="n",
        provenance=_provenance(),
        trace=_trace(),
        force=True,
    )
    assert forced.ok


def test_find_similar_nodes_uses_single_embedding_after_updates(graph):
    """A node's embedding is aliases+summary only, so it does not drift as log
    notes accumulate — dedup keeps matching the same node across updates that
    only touch `note`, using the single `embedding` vector (no separate
    identity vector)."""

    created = memory.create_node(
        graph,
        aliases=["Melanie"],
        summary="Melanie is a potter.",
        note="n",
        provenance=_provenance(),
        trace=_trace(),
    )
    assert created.ok

    # Note-only updates never touch aliases/summary, so the embedding must not move.
    for i in range(5):
        memory.update_node(
            graph,
            node_id=created.id,
            note=f"unrelated tangent number {i} " + "z" * 300,
            provenance=_provenance(),
            trace=_trace(),
        )

    probe_vec = _fake_embed("Melanie — Melanie is a potter.")
    similar = store_graph.find_similar_nodes(
        graph,
        aliases=["distinct-alias"],
        embedding=probe_vec,
        max_distance=0.01,
    )
    assert any(nid == created.id for nid, _, _ in similar)


def test_create_node_allows_distinct_entities(graph):
    a = memory.create_node(
        graph,
        aliases=["Oscar"],
        summary="A guinea pig.",
        note="n",
        provenance=_provenance(),
        trace=_trace(),
    )
    b = memory.create_node(
        graph,
        aliases=["Pottery class"],
        summary="A weekly pottery class.",
        note="n",
        provenance=_provenance(),
        trace=_trace(),
    )
    assert a.ok and b.ok


# --- single embedding, no identity_embedding -------------------------------


def test_create_and_update_node_write_only_embedding(graph):
    created = memory.create_node(
        graph,
        aliases=["Oscar"],
        summary="A guinea pig.",
        note="n",
        provenance=_provenance(),
        trace=_trace(),
    )
    props = (
        graph.query("MATCH (n:SemanticNode {id: $id}) RETURN n", {"id": created.id})
        .result_set[0][0]
        .properties
    )
    assert "embedding" in props
    assert "identity_embedding" not in props

    memory.update_node(
        graph,
        node_id=created.id,
        note="n2",
        provenance=_provenance(),
        trace=_trace(),
        new_summary="A guinea pig named Oscar.",
    )
    props = (
        graph.query("MATCH (n:SemanticNode {id: $id}) RETURN n", {"id": created.id})
        .result_set[0][0]
        .properties
    )
    assert "identity_embedding" not in props


def test_single_node_vector_index_exists(graph):
    indexes = graph.query("CALL db.indexes()").result_set
    node_vector_indexes = [
        row
        for row in indexes
        if row[0] == "SemanticNode" and "embedding" in (row[1] or [])
    ]
    assert node_vector_indexes
    identity_indexes = [
        row
        for row in indexes
        if row[0] == "SemanticNode" and "identity_embedding" in (row[1] or [])
    ]
    assert not identity_indexes
