"""
Semantic graph CRUD — SemanticNode and SemanticEdge operations in FalkorDB.
Implemented in Phase 3.
"""

import json
import re
import unicodedata
from datetime import datetime
from typing import Any, NamedTuple

from falkordb import Graph

from weavy.application.contracts import (
    GetNodeNeighborhoodOutput,
    GetNodeResult,
    NeighborSummary,
    OperationResult,
    SearchDiagnostics,
    SearchGraphOutput,
    SearchResult,
)
from weavy.config import settings
from weavy.models.graph import LogEntry, ProvenanceInput, SemanticEdge, SemanticNode
from weavy.timeutil import in_range, normalize_range, utc_now

_SESSION_ID_RE = re.compile(r"^s:\d+$")
_WORD_RE = re.compile(r"[\w'’-]+")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _vecf32_literal(vec: list[float]) -> str:
    """Format a Python float list as a FalkorDB vecf32() Cypher literal.

    FalkorDB's vecf32() requires an inline array literal — it cannot accept
    parameterized values via $variable. All official docs use f-string
    interpolation. This helper centralizes validation and formatting.
    """
    import math

    for i, v in enumerate(vec):
        if not isinstance(v, (int, float)):
            raise ValueError(
                f"Embedding element {i} is {type(v).__name__}, expected float"
            )
        if math.isnan(v) or math.isinf(v):
            raise ValueError(f"Embedding element {i} is {v}, expected finite float")
    return f"vecf32({vec})"


def _serialize_log_entry(entry: LogEntry) -> str:
    return json.dumps(entry.model_dump(mode="python"), default=_json_default)


def _json_default(value: object) -> str:
    if isinstance(value, datetime):
        return value.isoformat()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _deserialize_log_entry(s: str) -> LogEntry:
    data = json.loads(s)
    return LogEntry(**data)


def _make_log_entry(
    provenance: ProvenanceInput | None,
    note: str,
    event_time: datetime | None = None,
    happened_at: datetime | None = None,
) -> LogEntry:
    if provenance is None:
        raise ValueError("Log entry requires provenance")
    timestamp = event_time or utc_now()
    return LogEntry(
        source_id=provenance.source_id,
        timestamp=timestamp,  # transaction time — when recorded/discussed
        happened_at=happened_at or timestamp,  # valid time — defaults to episode time
        note=note,
    )


def _node_from_props(
    props: dict[str, Any], log_tail: int | None = None
) -> SemanticNode:
    """Construct a SemanticNode from raw FalkorDB properties."""
    raw_log = props.get("log") or []
    raw_tail = raw_log[-log_tail:] if log_tail is not None else raw_log
    log = [_deserialize_log_entry(s) for s in raw_tail]
    return SemanticNode(
        id=props["id"],
        aliases=props.get("aliases") or [],
        summary=props["summary"],
        total_log_count=props.get("total_log_count", 0),
        log=log,
    )


def _build_outgoing_edges(
    node_id: str,
    raw_edges: list[list[Any] | None],
    log_tail: int | None = None,
) -> list[SemanticEdge]:
    edges: list[SemanticEdge] = []
    for edge_data in raw_edges:
        if edge_data is None:
            continue
        edge_id, label, to_id, fact, raw_log, total_log_count, source_id = edge_data
        if edge_id is None or to_id is None:
            continue
        full_log = raw_log or []
        tail = full_log[-log_tail:] if log_tail is not None else full_log
        edges.append(
            SemanticEdge(
                id=edge_id,
                from_node_id=node_id,
                to_node_id=to_id,
                label=label,
                fact=fact or "",
                total_log_count=total_log_count or 0,
                log=[_deserialize_log_entry(s) for s in tail],
                source_id=source_id,
            )
        )
    return edges


def create_node(
    graph: Graph,
    aliases: list[str],
    summary: str,
    note: str,
    provenance: ProvenanceInput | None,
    node_id: str,
    embedding: list[float] | None = None,
    event_time: datetime | None = None,
    happened_at: datetime | None = None,
) -> OperationResult:
    entry = _make_log_entry(provenance, note, event_time, happened_at)
    entry_json = _serialize_log_entry(entry)
    embedding_clause = ""
    if embedding is not None:
        embedding_clause = f", embedding: {_vecf32_literal(embedding)}"
    # `name` is not read by app code; it's the FalkorDB Browser display
    # caption (see store/system.py:init_system).
    graph.query(
        f"""
        CREATE (n:SemanticNode {{
            id: $id,
            name: $name,
            aliases: $aliases,
            summary: $summary,
            total_log_count: 1,
            log: [$entry_json]{embedding_clause}
        }})
        """,
        {
            "id": node_id,
            "name": aliases[0],
            "aliases": aliases,
            "summary": summary,
            "entry_json": entry_json,
        },
    )
    return OperationResult(ok=True, id=node_id)


def update_node(
    graph: Graph,
    node_id: str,
    note: str,
    new_summary: str | None,
    new_aliases: list[str] | None,
    provenance: ProvenanceInput | None,
    embedding: list[float] | None = None,
    current_summary: str | None = None,
    event_time: datetime | None = None,
    happened_at: datetime | None = None,
) -> OperationResult:
    effective_note = note
    if new_summary is not None:
        if current_summary is None:
            result = graph.query(
                "MATCH (n:SemanticNode {id: $id}) RETURN n.summary",
                {"id": node_id},
            )
            if not result.result_set:
                raise ValueError(f"SemanticNode '{node_id}' not found.")
            current_summary = result.result_set[0][0]
        effective_note = f"[archived summary] {current_summary} | Agent note: {note}"

    entry = _make_log_entry(provenance, effective_note, event_time, happened_at)
    entry_json = _serialize_log_entry(entry)

    set_parts = [
        "n.log = n.log + [$entry_json]",
        "n.total_log_count = n.total_log_count + 1",
    ]
    params: dict[str, Any] = {"id": node_id, "entry_json": entry_json}

    if new_summary is not None:
        set_parts.append("n.summary = $new_summary")
        params["new_summary"] = new_summary
    if new_aliases is not None:
        set_parts.append("n.aliases = $new_aliases")
        set_parts.append("n.name = $new_name")
        params["new_aliases"] = new_aliases
        params["new_name"] = new_aliases[0]
    if embedding is not None:
        set_parts.append(f"n.embedding = {_vecf32_literal(embedding)}")

    result = graph.query(
        f"MATCH (n:SemanticNode {{id: $id}}) SET {', '.join(set_parts)} RETURN n",
        params,
    )
    if not result.result_set:
        raise ValueError(f"SemanticNode '{node_id}' not found.")

    return OperationResult(ok=True, id=node_id)


def link_mention(graph: Graph, session_id: str, node_id: str) -> None:
    """Record that an episode touched a node: (:Session)-[:MENTIONS]->(:SemanticNode).

    Idempotent via MERGE — repeated touches in the same session do not duplicate.
    """
    graph.query(
        """
        MATCH (s:Session {id: $sid}), (n:SemanticNode {id: $nid})
        MERGE (s)-[:MENTIONS]->(n)
        """,
        {"sid": session_id, "nid": node_id},
    )


def delete_node(graph: Graph, node_id: str) -> OperationResult:
    result = graph.query(
        "MATCH (n:SemanticNode {id: $id}) DETACH DELETE n RETURN count(n)",
        {"id": node_id},
    )
    deleted = result.result_set[0][0] if result.result_set else 0
    if deleted == 0:
        raise ValueError(f"SemanticNode '{node_id}' not found.")
    return OperationResult(ok=True, id=node_id)


# ---------------------------------------------------------------------------
# Edge CRUD
# ---------------------------------------------------------------------------


def create_edge(
    graph: Graph,
    from_node_id: str,
    to_node_id: str,
    label: str,
    fact: str,
    note: str,
    edge_id: str,
    provenance: ProvenanceInput | None,
    embedding: list[float] | None = None,
    event_time: datetime | None = None,
    happened_at: datetime | None = None,
    source_id: str | None = None,
) -> OperationResult:
    entry = _make_log_entry(provenance, note, event_time, happened_at)
    entry_json = _serialize_log_entry(entry)
    embedding_clause = (
        f", embedding: {_vecf32_literal(embedding)}" if embedding is not None else ""
    )
    result = graph.query(
        f"""
        MATCH (a:SemanticNode {{id: $from_id}}), (b:SemanticNode {{id: $to_id}})
        CREATE (a)-[r:RELATES {{
            id: $edge_id,
            label: $label,
            fact: $fact,
            total_log_count: 1,
            log: [$entry_json],
            source_id: $source_id{embedding_clause}
        }}]->(b)
        RETURN r
        """,
        {
            "from_id": from_node_id,
            "to_id": to_node_id,
            "edge_id": edge_id,
            "label": label,
            "fact": fact,
            "entry_json": entry_json,
            "source_id": source_id,
        },
    )
    if not result.result_set:
        # Domain refusal, not a system fault: report it as a result the agent
        # can react to (look up real ids and retry) instead of raising.
        return OperationResult(
            ok=False,
            message=(
                f"Not created — one or both nodes do not exist "
                f"(from='{from_node_id}', to='{to_node_id}'). Use node ids "
                f"returned by create_node/search_graph, then retry."
            ),
        )
    return OperationResult(ok=True, id=edge_id)


def update_edge(
    graph: Graph,
    edge_id: str,
    note: str,
    new_label: str | None = None,
    new_fact: str | None = None,
    provenance: ProvenanceInput | None = None,
    embedding: list[float] | None = None,
    event_time: datetime | None = None,
    happened_at: datetime | None = None,
) -> OperationResult:
    entry = _make_log_entry(provenance, note, event_time, happened_at)
    entry_json = _serialize_log_entry(entry)

    set_parts = [
        "r.log = r.log + [$entry_json]",
        "r.total_log_count = r.total_log_count + 1",
    ]
    params: dict[str, Any] = {"id": edge_id, "entry_json": entry_json}

    if new_label is not None:
        set_parts.append("r.label = $new_label")
        params["new_label"] = new_label
    if new_fact is not None:
        set_parts.append("r.fact = $new_fact")
        params["new_fact"] = new_fact
    if embedding is not None:
        set_parts.append(f"r.embedding = {_vecf32_literal(embedding)}")

    result = graph.query(
        f"MATCH ()-[r:RELATES {{id: $id}}]->() SET {', '.join(set_parts)} RETURN r",
        params,
    )
    if not result.result_set:
        raise ValueError(f"Edge '{edge_id}' not found.")
    return OperationResult(ok=True, id=edge_id)


def get_edge(graph: Graph, edge_id: str) -> SemanticEdge:
    result = graph.query(
        """
        MATCH (a:SemanticNode)-[r:RELATES {id: $id}]->(b:SemanticNode)
        RETURN r.id, r.label, r.fact, r.log, r.total_log_count, r.source_id, a.id, b.id
        """,
        {"id": edge_id},
    )
    if not result.result_set:
        raise ValueError(f"Edge '{edge_id}' not found.")
    eid, label, fact, raw_log, total_log_count, source_id, from_id, to_id = (
        result.result_set[0]
    )
    return SemanticEdge(
        id=eid,
        from_node_id=from_id,
        to_node_id=to_id,
        label=label,
        fact=fact or "",
        total_log_count=total_log_count or 0,
        log=[_deserialize_log_entry(s) for s in (raw_log or [])],
        source_id=source_id,
    )


def delete_edge(graph: Graph, edge_id: str) -> OperationResult:
    result = graph.query(
        "MATCH ()-[r:RELATES {id: $id}]->() DELETE r RETURN count(r)",
        {"id": edge_id},
    )
    deleted = result.result_set[0][0] if result.result_set else 0
    if deleted == 0:
        raise ValueError(f"Edge '{edge_id}' not found.")
    return OperationResult(ok=True, id=edge_id)


# ---------------------------------------------------------------------------
# Read operations
# ---------------------------------------------------------------------------


def _node_row(
    node_id, aliases, summary, edge_count, distance: float | None = None
) -> SearchResult:
    return SearchResult(
        kind="node",
        id=node_id,
        label=aliases[0] if aliases else node_id,
        text=summary.splitlines()[0] if summary else "",
        distance=distance,
        edge_count=int(edge_count),
    )


def _edge_row(
    edge_id, label, fact, from_id, to_id, distance: float | None = None
) -> SearchResult:
    return SearchResult(
        kind="edge",
        id=edge_id,
        label=label or "",
        text=(fact or "").splitlines()[0] if fact else "",
        distance=distance,
        endpoints=[from_id, to_id],
    )


def _log_when(entry: LogEntry) -> datetime:
    """Valid time of a log entry, falling back to record time for legacy entries."""
    return entry.happened_at or entry.timestamp


class _MemoryInfo(NamedTuple):
    """Log + provenance for one node or edge, used for time filtering and output."""

    log: list[LogEntry]
    sources: list[str]  # s:N ids, most recently recorded first, de-duplicated


def _ordered_sources(log: list[LogEntry], extra: list[str] | None = None) -> list[str]:
    by_recency = sorted(log, key=lambda e: e.timestamp, reverse=True)
    candidates = [e.source_id for e in by_recency] + sorted(extra or [])
    return [
        sid
        for sid in dict.fromkeys(candidates)
        if isinstance(sid, str) and _SESSION_ID_RE.match(sid)
    ]


def _node_infos(graph: Graph, node_ids: list[str]) -> dict[str, _MemoryInfo]:
    if not node_ids:
        return {}
    result = graph.query(
        """
        MATCH (n:SemanticNode) WHERE n.id IN $ids
        OPTIONAL MATCH (s:Session)-[:MENTIONS]->(n)
        RETURN n.id, n.log, collect(DISTINCT s.id)
        """,
        {"ids": node_ids},
    )
    infos: dict[str, _MemoryInfo] = {}
    for node_id, raw_log, mentioned_by in result.result_set:
        log = [_deserialize_log_entry(s) for s in raw_log or []]
        infos[node_id] = _MemoryInfo(log, _ordered_sources(log, mentioned_by))
    return infos


def _edge_infos(graph: Graph, edge_ids: list[str]) -> dict[str, _MemoryInfo]:
    if not edge_ids:
        return {}
    result = graph.query(
        """
        MATCH ()-[r:RELATES]->() WHERE r.id IN $ids
        RETURN r.id, r.log, r.source_id
        """,
        {"ids": edge_ids},
    )
    infos: dict[str, _MemoryInfo] = {}
    for edge_id, raw_log, source_id in result.result_set:
        log = [_deserialize_log_entry(s) for s in raw_log or []]
        extra = [source_id] if source_id else []
        infos[edge_id] = _MemoryInfo(log, _ordered_sources(log, extra))
    return infos


def _memory_infos(graph: Graph, rows: list[SearchResult]) -> dict[str, _MemoryInfo]:
    infos = _node_infos(graph, [r.id for r in rows if r.kind == "node"])
    infos.update(_edge_infos(graph, [r.id for r in rows if r.kind == "edge"]))
    return infos


def _in_time_range(
    info: _MemoryInfo | None, bounds: tuple[datetime | None, datetime | None]
) -> bool:
    """True when at least one log entry's valid time falls inside the bounds."""
    if info is None:
        return False
    return any(in_range(_log_when(e), *bounds) for e in info.log)


def _attach_provenance(
    row: SearchResult, info: _MemoryInfo | None, max_sources: int
) -> None:
    if info is None:
        return
    row.source_sessions = info.sources[:max_sources]
    row.source_count = len(info.sources)
    if info.log:
        valid_times = [_log_when(e) for e in info.log]
        row.earliest_happened_at = min(valid_times)
        row.latest_happened_at = max(valid_times)
        row.last_recorded_at = max(e.timestamp for e in info.log)


def normalize_query(query: str) -> str:
    """Unicode-normalize and collapse whitespace. Content (dates, numbers,
    names) is never removed, so temporal cues in the query are preserved."""
    return " ".join(unicodedata.normalize("NFKC", query).split())


def _mention_haystack(query_lower: str) -> str:
    """Space-padded word sequence of the query, for whole-word alias mentions."""
    return " " + " ".join(_WORD_RE.findall(query_lower)) + " "


# Reciprocal Rank Fusion constant (standard choice; see Cormack et al. 2009).
# RRF combines the vector-distance ranking and the keyword match-tier ranking
# by rank position rather than raw magnitude, so a top keyword hit (e.g. an
# exact alias/title match) can outrank a middling vector hit instead of being
# structurally capped below every embedding result.
_RRF_K = 60

# Shortest alias that counts as "mentioned" when it appears inside a longer
# query ("When did I adopt Biscuit?" → alias "Biscuit"). Shorter aliases such
# as "me" or "AI" occur inside ordinary questions too often to be a signal.
_MIN_ALIAS_MENTION_LEN = 3


def _rrf_scores(ranked_ids: list[str]) -> dict[str, float]:
    return {rid: 1.0 / (_RRF_K + rank) for rank, rid in enumerate(ranked_ids, start=1)}


def search_graph(
    graph: Graph,
    *,
    query: str,
    limit: int = 10,
    query_embedding: list[float] | None = None,
    time_range: list[datetime | None] | None = None,
    max_distance: float | None = None,
    candidate_pool: int | None = None,
    expand_depth: int = 0,
    expand_limit: int = 0,
    max_sources: int | None = None,
) -> SearchGraphOutput:
    """Hybrid (vector + keyword) search over nodes and edges.

    Pipeline: normalize query → gather up to ``candidate_pool`` candidates per
    pass → drop vector hits farther than ``max_distance`` → **filter by
    ``time_range`` before ranking** → RRF fusion → top ``limit`` → optional
    bounded neighbour expansion → attach provenance (source sessions, time span).

    Defaults keep the store layer policy-free (no distance cut-off, no
    expansion); ``services.memory.search_graph`` applies configured policy.
    """
    if limit < 1:
        raise ValueError("limit must be >= 1")
    pool = max(
        limit,
        candidate_pool
        if candidate_pool is not None
        else settings.SEARCH_CANDIDATE_POOL,
    )
    sources_cap = (
        max_sources if max_sources is not None else settings.SEARCH_MAX_SOURCES
    )
    bounds = normalize_range(time_range)
    q = normalize_query(query)
    q_lower = q.lower()

    rows: dict[str, SearchResult] = {}
    vector_hits: list[tuple[float, str]] = []
    keyword_hits: list[tuple[int, str]] = []
    dropped_by_distance: set[str] = set()

    if query_embedding is not None:
        vec_literal = _vecf32_literal(query_embedding)
        node_vec = graph.query(
            f"""
            CALL db.idx.vector.queryNodes(
                'SemanticNode', 'embedding', $limit, {vec_literal}
            ) YIELD node, score
            OPTIONAL MATCH (node)-[r:RELATES]-()
            WITH node, score, count(r) AS edge_count
            RETURN node.id, node.aliases, node.summary, edge_count, score
            """,
            {"limit": pool},
        )
        for nid, aliases, summary, edge_count, distance in node_vec.result_set:
            if max_distance is not None and float(distance) > max_distance:
                dropped_by_distance.add(nid)
                continue
            rows[nid] = _node_row(nid, aliases, summary, edge_count, float(distance))
            vector_hits.append((float(distance), nid))

        # FalkorDB indexes relationships separately; surface edge-facts in the same ranking.
        # Read endpoints via startNode()/endNode() rather than re-MATCHing `relationship` —
        # FalkorDB doesn't treat a YIELDed relationship as bound in a later MATCH pattern,
        # which silently collapses every row's `score` to the first row's value.
        edge_vec = graph.query(
            f"""
            CALL db.idx.vector.queryRelationships(
                'RELATES', 'embedding', $limit, {vec_literal}
            ) YIELD relationship, score
            RETURN relationship.id, relationship.label, relationship.fact,
                   startNode(relationship).id, endNode(relationship).id, score
            """,
            {"limit": pool},
        )
        for eid, label, fact, from_id, to_id, distance in edge_vec.result_set:
            if max_distance is not None and float(distance) > max_distance:
                dropped_by_distance.add(eid)
                continue
            rows[eid] = _edge_row(eid, label, fact, from_id, to_id, float(distance))
            vector_hits.append((float(distance), eid))

    if q_lower:
        # match_tier 0 = the query hit the entity's identity: the query occurs
        # in an alias, or an alias is mentioned as whole words inside the
        # query; 1 = it only hit free text (summary). Ordering by tier gives
        # keyword hits a real rank to feed into RRF.
        node_kw = graph.query(
            """
            MATCH (n:SemanticNode)
            WITH n,
                 CASE
                   WHEN ANY(a IN n.aliases WHERE toLower(a) CONTAINS $query) THEN 0
                   WHEN ANY(a IN n.aliases WHERE size(a) >= $min_len
                            AND $haystack CONTAINS (' ' + toLower(a) + ' ')) THEN 0
                   WHEN toLower(n.summary) CONTAINS $query THEN 1
                   ELSE 2
                 END AS match_tier
            WHERE match_tier < 2
            OPTIONAL MATCH (n)-[r:RELATES]-()
            WITH n, match_tier, count(r) AS edge_count
            RETURN n.id, n.aliases, n.summary, edge_count, match_tier
            ORDER BY match_tier ASC, edge_count DESC
            LIMIT $limit
            """,
            {
                "query": q_lower,
                "haystack": _mention_haystack(q_lower),
                "min_len": _MIN_ALIAS_MENTION_LEN,
                "limit": pool,
            },
        )
        for nid, aliases, summary, edge_count, match_tier in node_kw.result_set:
            if nid not in rows:
                rows[nid] = _node_row(nid, aliases, summary, edge_count)
            keyword_hits.append((match_tier, nid))

        edge_kw = graph.query(
            """
            MATCH (a:SemanticNode)-[r:RELATES]->(b:SemanticNode)
            WHERE toLower(r.label) CONTAINS $query
               OR toLower(r.fact) CONTAINS $query
            WITH a, b, r,
                 CASE WHEN toLower(r.label) CONTAINS $query
                      THEN 0 ELSE 1 END AS match_tier
            RETURN r.id, r.label, r.fact, a.id, b.id, match_tier
            ORDER BY match_tier ASC
            LIMIT $limit
            """,
            {"query": q_lower, "limit": pool},
        )
        for eid, label, fact, from_id, to_id, match_tier in edge_kw.result_set:
            if eid not in rows:
                rows[eid] = _edge_row(eid, label, fact, from_id, to_id)
            keyword_hits.append((match_tier, eid))

    infos = _memory_infos(graph, list(rows.values()))
    if bounds is not None:
        kept = {rid for rid in rows if _in_time_range(infos.get(rid), bounds)}
    else:
        kept = set(rows)

    # Rank only the survivors, so the time filter never empties a top-k that
    # in-range candidates further down the list could have filled.
    vector_rank = [rid for _, rid in sorted(vector_hits) if rid in kept]
    keyword_rank = [
        rid for _, rid in sorted(keyword_hits, key=lambda t: t[0]) if rid in kept
    ]
    vector_scores = _rrf_scores(vector_rank)
    keyword_scores = _rrf_scores(keyword_rank)
    fused = {
        rid: vector_scores.get(rid, 0.0) + keyword_scores.get(rid, 0.0) for rid in kept
    }
    ranked = sorted(kept, key=lambda rid: (-fused[rid], rid))
    results = [rows[rid] for rid in ranked[:limit]]

    expanded: list[SearchResult] = []
    if expand_depth > 0 and expand_limit > 0 and results:
        expanded, expanded_infos = _expand_neighbors(
            graph,
            results,
            depth=expand_depth,
            limit=expand_limit,
            fetch_cap=pool,
            bounds=bounds,
        )
        infos.update(expanded_infos)

    for row in results + expanded:
        _attach_provenance(row, infos.get(row.id), sources_cap)

    return SearchGraphOutput(
        results=results + expanded,
        diagnostics=SearchDiagnostics(
            candidates=len(set(rows) | dropped_by_distance),
            dropped_by_distance=len(dropped_by_distance - set(rows)),
            dropped_by_time=len(rows) - len(kept),
            expanded=len(expanded),
            max_distance=max_distance,
            time_range=list(bounds) if bounds is not None else None,
        ),
    )


def _expand_neighbors(
    graph: Graph,
    seeds: list[SearchResult],
    *,
    depth: int,
    limit: int,
    fetch_cap: int,
    bounds: tuple[datetime | None, datetime | None] | None,
) -> tuple[list[SearchResult], dict[str, _MemoryInfo]]:
    """Bounded breadth-first expansion from the direct hits.

    Seeds are visited in rank order, so neighbours of better hits come first.
    At most ``limit`` rows are added across all ``depth`` hops; each row names
    the hit it hangs off (``via``), the connecting edge, and its hop count.
    Edge hits seed from their endpoints, which the agent already has as ids.
    """
    seen = {r.id for r in seeds}
    frontier: list[str] = []
    for r in seeds:
        frontier.extend([r.id] if r.kind == "node" else (r.endpoints or []))
    frontier = list(dict.fromkeys(frontier))
    seen.update(frontier)

    added: list[SearchResult] = []
    infos: dict[str, _MemoryInfo] = {}
    for hop in range(1, depth + 1):
        next_frontier: list[str] = []
        for seed_id in frontier:
            if len(added) >= limit:
                break
            result = graph.query(
                """
                MATCH (s:SemanticNode {id: $seed})-[r:RELATES]-(m:SemanticNode)
                WHERE NOT m.id IN $seen
                OPTIONAL MATCH (m)-[r2:RELATES]-()
                WITH m, r, count(r2) AS edge_count
                RETURN m.id, m.aliases, m.summary, edge_count, r.id
                ORDER BY edge_count DESC, m.id ASC
                LIMIT $cap
                """,
                {"seed": seed_id, "seen": sorted(seen), "cap": fetch_cap},
            )
            candidates: dict[str, tuple[SearchResult, str]] = {}
            for mid, aliases, summary, edge_count, edge_id in result.result_set:
                if mid not in candidates:
                    candidates[mid] = (
                        _node_row(mid, aliases, summary, edge_count),
                        edge_id,
                    )
            cand_infos = _node_infos(graph, list(candidates))
            for mid, (row, edge_id) in candidates.items():
                if len(added) >= limit:
                    break
                if bounds is not None and not _in_time_range(
                    cand_infos.get(mid), bounds
                ):
                    continue
                row.via, row.via_edge, row.hops = seed_id, edge_id, hop
                added.append(row)
                infos[mid] = cand_infos[mid]
                seen.add(mid)
                next_frontier.append(mid)
        frontier = next_frontier
        if not frontier or len(added) >= limit:
            break
    return added, infos


def find_similar_nodes(
    graph: Graph,
    *,
    aliases: list[str],
    embedding: list[float],
    max_distance: float,
    limit: int = 3,
) -> list[tuple[str, str, float]]:
    """Existing nodes that likely denote the same entity as a prospective write.

    A node matches if it shares an alias (case-insensitive) or its embedding is
    within *max_distance* (cosine). Returns [(node_id, canonical_alias,
    distance)] ordered nearest-first; alias matches without a close vector
    report distance as found by the vector query or ``max_distance`` if
    outside its top-k.

    Queries the single ``embedding`` index — every node's embedding is aliases
    + current summary only, so the candidate's own aliases+summary vector
    always compares apples-to-apples against it.
    """
    candidates: dict[str, tuple[str, float]] = {}

    rows = graph.query(
        f"""
        CALL db.idx.vector.queryNodes(
            'SemanticNode', 'embedding', $limit, {_vecf32_literal(embedding)}
        ) YIELD node, score
        RETURN node.id, node.aliases, score
        """,
        {"limit": limit},
    )
    lowered = {a.lower() for a in aliases}
    for nid, node_aliases, score in rows.result_set:
        alias_hit = any(a.lower() in lowered for a in node_aliases or [])
        if float(score) <= max_distance or alias_hit:
            canonical = (node_aliases or [nid])[0]
            candidates[nid] = (canonical, float(score))

    alias_rows = graph.query(
        """
        MATCH (n:SemanticNode)
        WHERE ANY(a IN n.aliases WHERE toLower(a) IN $aliases)
        RETURN n.id, n.aliases
        LIMIT $limit
        """,
        {"aliases": list(lowered), "limit": limit},
    )
    for nid, node_aliases in alias_rows.result_set:
        if nid not in candidates:
            candidates[nid] = ((node_aliases or [nid])[0], max_distance)

    return sorted(
        [(nid, alias, dist) for nid, (alias, dist) in candidates.items()],
        key=lambda t: t[2],
    )[:limit]


def get_node(graph: Graph, node_id: str) -> GetNodeResult:
    nodes_by_id = get_nodes(graph, [node_id])
    try:
        return nodes_by_id[node_id]
    except KeyError as exc:
        raise ValueError(f"SemanticNode '{node_id}' not found.") from exc


def get_nodes(graph: Graph, node_ids: list[str]) -> dict[str, GetNodeResult]:
    if not node_ids:
        return {}

    # Fetch node properties, outgoing edges, and source episodes in one query.
    result = graph.query(
        """
        MATCH (n:SemanticNode)
        WHERE n.id IN $ids
        OPTIONAL MATCH (n)-[r:RELATES]->(m:SemanticNode)
        WITH n, collect(CASE WHEN r IS NOT NULL THEN [r.id, r.label, m.id, r.fact, r.log, r.total_log_count, r.source_id] ELSE null END) AS edges
        OPTIONAL MATCH (s:Session)-[:MENTIONS]->(n)
        RETURN n.id, n, edges, collect(DISTINCT s.id) AS mentioned_by
        """,
        {"ids": node_ids},
    )

    nodes_by_id: dict[str, GetNodeResult] = {}
    for found_node_id, raw_node, raw_edges, mentioned_by in result.result_set:
        node_props = raw_node.properties
        nodes_by_id[found_node_id] = GetNodeResult(
            node=_node_from_props(node_props),
            edges=_build_outgoing_edges(found_node_id, raw_edges or [], log_tail=2),
            mentioned_by=[sid for sid in (mentioned_by or []) if sid is not None],
        )
    return nodes_by_id


def get_node_neighborhood(
    graph: Graph, node_id: str, limit: int | None = None
) -> GetNodeNeighborhoodOutput:
    """A node plus its neighbours, capped at *limit* (default NEIGHBORHOOD_LIMIT).

    Hub nodes can have hundreds of neighbours; the cap keeps one tool result
    from flooding the agent's context. ``total_neighbors`` and ``truncated``
    report what was left out. Neighbours are ordered by edge id (creation order).
    """
    cap = limit if limit is not None else settings.NEIGHBORHOOD_LIMIT
    result = graph.query(
        """
        MATCH (n:SemanticNode {id: $id})
        OPTIONAL MATCH (n)-[r:RELATES]-(m:SemanticNode)
        RETURN n, collect(CASE WHEN r IS NOT NULL THEN [r.id, r.label, m.id, m.aliases, m.summary, r.fact] ELSE null END) AS neighbor_data
        """,
        {"id": node_id},
    )
    if not result.result_set:
        raise ValueError(f"SemanticNode '{node_id}' not found.")

    row = result.result_set[0]
    node_props = row[0].properties
    raw_neighbors = row[1] or []

    node = _node_from_props(node_props, log_tail=2)

    neighbors = []
    seen_edge_ids = set()
    for nd in raw_neighbors:
        if nd is None:
            continue
        (
            edge_id,
            edge_label,
            neighbor_id,
            neighbor_aliases,
            neighbor_summary,
            edge_fact,
        ) = nd
        if edge_id is None or edge_id in seen_edge_ids:
            continue
        seen_edge_ids.add(edge_id)
        canonical = neighbor_aliases[0] if neighbor_aliases else neighbor_id
        summary_line = neighbor_summary.splitlines()[0] if neighbor_summary else ""
        neighbors.append(
            NeighborSummary(
                edge_id=edge_id,
                edge_label=edge_label,
                edge_fact=edge_fact,
                node_id=neighbor_id,
                canonical_alias=canonical,
                summary_line=summary_line,
            )
        )

    neighbors.sort(key=lambda n: _id_number(n.edge_id))
    return GetNodeNeighborhoodOutput(
        node=node,
        neighbors=neighbors[:cap],
        total_neighbors=len(neighbors),
        truncated=len(neighbors) > cap,
    )


def _id_number(value: str) -> int:
    _, _, suffix = value.partition(":")
    return int(suffix) if suffix.isdigit() else -1
