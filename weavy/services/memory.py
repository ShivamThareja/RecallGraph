from __future__ import annotations

from datetime import datetime
from typing import Any

from falkordb import Graph

from weavy.config import settings
from weavy.application.contracts import (
    GetNodeOutput,
    OperationResult,
    SearchGraphOutput,
)
from weavy.models.graph import ProvenanceInput
from weavy.models.traces import RunTrace, TouchedEdge, TouchedNode
from weavy.services import embedding
from weavy.store import graph as store_graph
from weavy.store import system as store_system

# Cosine distance below which a prospective node is treated as an existing
# entity. Deliberately tight: in the 0.3–0.4 band same-entity and
# distinct-entity pairs overlap (summaries describe content, not identity), so
# the vector check only catches near-certain rephrasings; alias equality is the
# primary identity signal and is checked independently of distance.
#
# This is an irreducible judgment call, not a derivable fact — it is scoped to
# the embedding model actually configured (settings.EMBEDDING_MODEL), not
# assumed universal across models.
DUPLICATE_DISTANCE = 0.2


def get_node(graph: Graph, *, node_ids: list[str]) -> GetNodeOutput:
    nodes_by_id = store_graph.get_nodes(graph, node_ids)
    results = [nodes_by_id[node_id] for node_id in node_ids if node_id in nodes_by_id]
    not_found = [node_id for node_id in node_ids if node_id not in nodes_by_id]
    return GetNodeOutput(results=results, not_found=not_found)


# Sentinel: "use the configured setting" (None is a meaningful value for
# max_distance — it disables the cut-off).
_FROM_SETTINGS: Any = object()


def search_graph(
    graph: Graph,
    *,
    query: str,
    limit: int = 10,
    time_range: list[datetime | None] | None = None,
    max_distance: float | None = _FROM_SETTINGS,
    candidate_pool: int | None = _FROM_SETTINGS,
    expand_depth: int = _FROM_SETTINGS,
    expand_limit: int = _FROM_SETTINGS,
) -> SearchGraphOutput:
    """Agent-facing hybrid search with the configured retrieval policy.

    Any keyword argument left at its default is read from ``settings`` at call
    time (``SEARCH_MAX_DISTANCE``, ``SEARCH_CANDIDATE_POOL``,
    ``SEARCH_EXPAND_DEPTH``, ``SEARCH_EXPAND_LIMIT``); evaluations pass
    explicit values to compare configurations.
    """

    def pick(value: Any, setting: str) -> Any:
        return getattr(settings, setting) if value is _FROM_SETTINGS else value

    query_embedding = embedding.embed(store_graph.normalize_query(query))
    return store_graph.search_graph(
        graph,
        query=query,
        limit=limit,
        query_embedding=query_embedding,
        time_range=time_range,
        max_distance=pick(max_distance, "SEARCH_MAX_DISTANCE"),
        candidate_pool=pick(candidate_pool, "SEARCH_CANDIDATE_POOL"),
        expand_depth=pick(expand_depth, "SEARCH_EXPAND_DEPTH"),
        expand_limit=pick(expand_limit, "SEARCH_EXPAND_LIMIT"),
    )


def create_node(
    graph: Graph,
    *,
    aliases: list[str],
    summary: str,
    note: str,
    provenance: ProvenanceInput,
    trace: RunTrace,
    event_time: datetime | None = None,
    happened_at: datetime | None = None,
    session_id: str | None = None,
    force: bool = False,
) -> OperationResult:
    vec = embedding.embed_node(aliases, summary)

    # One node per entity is a storage invariant, not a prompt convention:
    # refuse writes that collide with an existing entity unless forced.
    if not force:
        similar = store_graph.find_similar_nodes(
            graph, aliases=aliases, embedding=vec, max_distance=DUPLICATE_DISTANCE
        )
        if similar:
            listing = ", ".join(
                f"{nid} ({alias!r}, distance {dist:.2f})"
                for nid, alias, dist in similar
            )
            return OperationResult(
                ok=False,
                message=(
                    f"Not created — existing node(s) likely denote this entity: "
                    f"{listing}. Call update_node on the match instead. If this "
                    f"truly is a distinct entity, retry with force=true."
                ),
            )

    node_id = store_system.increment_counter(graph, "node")
    result = store_graph.create_node(
        graph,
        aliases=aliases,
        summary=summary,
        note=note,
        provenance=provenance,
        node_id=node_id,
        embedding=vec,
        event_time=event_time,
        happened_at=happened_at,
    )
    if session_id is not None:
        store_graph.link_mention(graph, session_id, node_id)
    trace.touched_nodes.append(TouchedNode(node_id=node_id, action="created"))
    return result


def update_node(
    graph: Graph,
    *,
    node_id: str,
    note: str,
    provenance: ProvenanceInput,
    trace: RunTrace,
    new_summary: str | None = None,
    new_aliases: list[str] | None = None,
    event_time: datetime | None = None,
    happened_at: datetime | None = None,
    session_id: str | None = None,
) -> OperationResult:
    # The embedding is aliases + current summary only, so it only needs
    # recomputing when either actually changes — not on every log entry.
    fetched_summary: str | None = None
    vec: list[float] | None = None
    if new_summary is not None or new_aliases is not None:
        current = store_graph.get_node(graph, node_id)
        fetched_summary = current.node.summary
        aliases = new_aliases if new_aliases is not None else current.node.aliases
        summary = new_summary if new_summary is not None else fetched_summary
        vec = embedding.embed_node(aliases, summary)

    result = store_graph.update_node(
        graph,
        node_id=node_id,
        note=note,
        new_summary=new_summary,
        new_aliases=new_aliases,
        provenance=provenance,
        embedding=vec,
        current_summary=fetched_summary,
        event_time=event_time,
        happened_at=happened_at,
    )
    if session_id is not None:
        store_graph.link_mention(graph, session_id, node_id)
    trace.touched_nodes.append(TouchedNode(node_id=node_id, action="updated"))
    return result


def delete_node(graph: Graph, *, node_id: str, trace: RunTrace) -> OperationResult:
    result = store_graph.delete_node(graph, node_id)
    trace.touched_nodes.append(TouchedNode(node_id=node_id, action="deleted"))
    return result


def create_edge(
    graph: Graph,
    *,
    from_node_id: str,
    to_node_id: str,
    label: str,
    fact: str,
    note: str,
    provenance: ProvenanceInput,
    trace: RunTrace,
    event_time: datetime | None = None,
    happened_at: datetime | None = None,
    source_id: str | None = None,
) -> OperationResult:
    edge_id = store_system.increment_counter(graph, "edge")
    vec = embedding.embed_edge(label, fact)
    result = store_graph.create_edge(
        graph,
        from_node_id=from_node_id,
        to_node_id=to_node_id,
        label=label,
        fact=fact,
        note=note,
        edge_id=edge_id,
        provenance=provenance,
        embedding=vec,
        event_time=event_time,
        happened_at=happened_at,
        source_id=source_id,
    )
    if result.ok:
        trace.touched_edges.append(TouchedEdge(edge_id=edge_id, action="created"))
        _link_edge_mentions(graph, source_id, from_node_id, to_node_id)
    return result


def _link_edge_mentions(
    graph: Graph, session_id: str | None, from_node_id: str, to_node_id: str
) -> None:
    """A session that states a relationship touched both of its endpoints.

    Linking MENTIONS for both makes the relationship's source episode
    reachable from either entity via ``get_node.mentioned_by``.
    """
    if session_id is None:
        return
    for node_id in (from_node_id, to_node_id):
        store_graph.link_mention(graph, session_id, node_id)


def update_edge(
    graph: Graph,
    *,
    edge_id: str,
    note: str,
    provenance: ProvenanceInput,
    trace: RunTrace,
    new_label: str | None = None,
    new_fact: str | None = None,
    event_time: datetime | None = None,
    happened_at: datetime | None = None,
    session_id: str | None = None,
) -> OperationResult:
    vec: list[float] | None = None
    current = store_graph.get_edge(graph, edge_id) if session_id is not None else None
    if new_label is not None or new_fact is not None:
        current = current or store_graph.get_edge(graph, edge_id)
        label = new_label if new_label is not None else current.label
        fact = new_fact if new_fact is not None else current.fact
        vec = embedding.embed_edge(label, fact)

    result = store_graph.update_edge(
        graph,
        edge_id=edge_id,
        note=note,
        new_label=new_label,
        new_fact=new_fact,
        provenance=provenance,
        embedding=vec,
        event_time=event_time,
        happened_at=happened_at,
    )
    trace.touched_edges.append(TouchedEdge(edge_id=edge_id, action="updated"))
    if current is not None:
        _link_edge_mentions(graph, session_id, current.from_node_id, current.to_node_id)
    return result


def delete_edge(graph: Graph, *, edge_id: str, trace: RunTrace) -> OperationResult:
    result = store_graph.delete_edge(graph, edge_id)
    trace.touched_edges.append(TouchedEdge(edge_id=edge_id, action="deleted"))
    return result
