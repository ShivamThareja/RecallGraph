from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Callable

from falkordb import Graph
from pydantic import BaseModel

from weavy.application.contracts import GetThemeOutput, OperationResult
from weavy.models.graph import ProvenanceInput
from weavy.harness.tool_models import (
    CompleteInput,
    CompleteThemeUpdateInput,
    CreateEdgeInput,
    CreateNodeInput,
    CreateThemeInput,
    DeleteEdgeInput,
    DeleteNodeInput,
    GetNodeInput,
    GetNodeNeighborhoodInput,
    GetSessionInput,
    GetThemeInput,
    ListSessionsInput,
    RetireThemeInput,
    SearchGraphInput,
    SetPrefaceInput,
    UpdateEdgeInput,
    UpdateNodeInput,
    UpdateThemeInput,
)
from weavy.models.traces import RunTrace
from weavy.services import memory
from weavy.store import canonical as store_canonical
from weavy.store import graph as store_graph
from weavy.store import system as store_system
from weavy.store import themes as store_themes


@dataclass
class ActionContext:
    graph: Graph
    trace: RunTrace
    event_time: datetime | None = None
    session_id: str | None = None


@dataclass
class Action:
    name: str
    description: str
    input_model: type[BaseModel]
    fn: Callable[[Any, ActionContext], Any]
    is_completion: bool = False


def _edge_provenance(ctx: ActionContext) -> ProvenanceInput:
    """Provenance for an edge log entry, sourced from the running session."""
    if ctx.session_id is None:
        raise ValueError("Edge write requires a session_id for provenance.")
    return ProvenanceInput(source_id=ctx.session_id)


def _provenance_mismatch(
    provenance: ProvenanceInput, ctx: ActionContext
) -> OperationResult | None:
    """Node writes must cite the session that is running.

    The agent supplies node provenance itself; without this check it could
    attribute a fact to any session (or a non-existent one), which would make
    ``mentioned_by`` and citation verification untrustworthy. Runs without a
    session (low-level harness use) keep the agent-supplied value.
    """
    if ctx.session_id is None or provenance.source_id == ctx.session_id:
        return None
    return OperationResult(
        ok=False,
        message=(
            f"Not written — provenance.source_id must be the current session "
            f"{ctx.session_id!r}, got {provenance.source_id!r}. Retry with "
            f"source_id={ctx.session_id!r}."
        ),
    )


def _create_node(p: Any, ctx: ActionContext) -> OperationResult:
    refusal = _provenance_mismatch(p.provenance, ctx)
    if refusal is not None:
        return refusal
    return memory.create_node(
        ctx.graph,
        aliases=p.aliases,
        summary=p.summary,
        note=p.note,
        provenance=p.provenance,
        trace=ctx.trace,
        event_time=ctx.event_time,
        happened_at=p.happened_at,
        session_id=ctx.session_id,
        force=p.force,
    )


def _update_node(p: Any, ctx: ActionContext) -> OperationResult:
    refusal = _provenance_mismatch(p.provenance, ctx)
    if refusal is not None:
        return refusal
    return memory.update_node(
        ctx.graph,
        node_id=p.node_id,
        note=p.note,
        provenance=p.provenance,
        trace=ctx.trace,
        new_summary=p.new_summary,
        new_aliases=p.new_aliases,
        event_time=ctx.event_time,
        happened_at=p.happened_at,
        session_id=ctx.session_id,
    )


def _complete(params: Any, ctx: ActionContext) -> OperationResult:
    ctx.trace.completion_payload = params.model_dump()
    return OperationResult(ok=True)


ACTIONS: dict[str, Action] = {
    "search_graph": Action(
        "search_graph",
        "Hybrid search over the semantic graph — the sole search surface. Semantic similarity (embedding) plus keyword matching (including entity names mentioned inside your query), returning a unified ranked list of entities (kind='node') and relationship facts (kind='edge'). Every result carries provenance: `source_sessions` (the s:N episodes behind it, most recent first; `source_count` is the total) and its valid-time span (`earliest_happened_at`/`latest_happened_at`, `last_recorded_at`). Rows with `via` set are not direct hits but graph neighbours of the hit named in `via` (connected by `via_edge`). Vector hits too far from the query are dropped, so an empty list means nothing relevant was found. Optional time_range [start, end] keeps only memories whose happened_at falls in the window and is applied before the top-`limit` cut; use null for an open bound (e.g. [start, null]). To read ground truth, call get_session on a source session.",
        SearchGraphInput,
        lambda p, ctx: memory.search_graph(
            ctx.graph, query=p.query, limit=p.limit, time_range=p.time_range
        ),
    ),
    "get_node": Action(
        "get_node",
        "Get one or more semantic nodes by id — returns each node's summary, log, outgoing edges (with their facts), and `mentioned_by` (the s:N episodes that touched it).",
        GetNodeInput,
        lambda p, ctx: memory.get_node(ctx.graph, node_ids=p.node_ids),
    ),
    "get_session": Action(
        "get_session",
        "Get an episode (session) by s:N id — returns its raw text and metadata. Use this to re-read the original source behind a node's facts; reach a node's source ids via get_node's `mentioned_by`.",
        GetSessionInput,
        lambda p, ctx: store_canonical.get_session_output(ctx.graph, p.session_id),
    ),
    "get_node_neighborhood": Action(
        "get_node_neighborhood",
        "Get a node with its neighbors (capped; `total_neighbors` and `truncated` report what was left out — pass a larger `limit` to see more).",
        GetNodeNeighborhoodInput,
        lambda p, ctx: store_graph.get_node_neighborhood(
            ctx.graph, p.node_id, limit=p.limit
        ),
    ),
    "list_sessions": Action(
        "list_sessions",
        "List sessions (transcripts and chats), most recent first. Optional date_range [start, end]; either bound may be null.",
        ListSessionsInput,
        lambda p, ctx: store_canonical.list_sessions(
            ctx.graph, limit=p.limit, date_range=p.date_range
        ),
    ),
    "get_theme": Action(
        "get_theme",
        "Get a theme by name.",
        GetThemeInput,
        lambda p, ctx: GetThemeOutput(theme=store_themes.get_theme(ctx.graph, p.name)),
    ),
    "create_node": Action(
        "create_node",
        "Create a semantic node. Creation is refused (ok=false, with candidates) when an existing node likely denotes the same entity — update that node instead, or retry with force=true only if the entity is genuinely distinct. provenance.source_id must be the current session.",
        CreateNodeInput,
        _create_node,
    ),
    "update_node": Action(
        "update_node",
        "Update a semantic node. provenance.source_id must be the current session.",
        UpdateNodeInput,
        _update_node,
    ),
    "delete_node": Action(
        "delete_node",
        "Delete a semantic node.",
        DeleteNodeInput,
        lambda p, ctx: memory.delete_node(
            ctx.graph, node_id=p.node_id, trace=ctx.trace
        ),
    ),
    "create_edge": Action(
        "create_edge",
        "Create a semantic edge between two nodes. Both node IDs must already exist in the graph — use only IDs returned by prior create_node calls, never IDs of nodes being created in the same batch. Requires a label (short directional relationship type), a fact (natural-language statement of the relationship — this is embedded and directly searchable), and a note (why this edge is written now — logged).",
        CreateEdgeInput,
        lambda p, ctx: memory.create_edge(
            ctx.graph,
            from_node_id=p.from_node_id,
            to_node_id=p.to_node_id,
            label=p.label,
            fact=p.fact,
            note=p.note,
            provenance=_edge_provenance(ctx),
            trace=ctx.trace,
            event_time=ctx.event_time,
            happened_at=p.happened_at,
            source_id=ctx.session_id,
        ),
    ),
    "update_edge": Action(
        "update_edge",
        "Update a semantic edge — append a log entry and optionally revise its label and/or fact. Use this when a relationship changes instead of creating a parallel edge; the fact should state the current truth and the note records what changed.",
        UpdateEdgeInput,
        lambda p, ctx: memory.update_edge(
            ctx.graph,
            edge_id=p.edge_id,
            note=p.note,
            provenance=_edge_provenance(ctx),
            trace=ctx.trace,
            new_label=p.new_label,
            new_fact=p.new_fact,
            event_time=ctx.event_time,
            happened_at=p.happened_at,
            session_id=ctx.session_id,
        ),
    ),
    "delete_edge": Action(
        "delete_edge",
        "Delete a semantic edge.",
        DeleteEdgeInput,
        lambda p, ctx: memory.delete_edge(
            ctx.graph, edge_id=p.edge_id, trace=ctx.trace
        ),
    ),
    "create_theme": Action(
        "create_theme",
        "Create a theme.",
        CreateThemeInput,
        lambda p, ctx: store_themes.create_theme(ctx.graph, p.name, p.state, p.anchors),
    ),
    "update_theme": Action(
        "update_theme",
        "Update a theme.",
        UpdateThemeInput,
        lambda p, ctx: store_themes.update_theme(
            ctx.graph, p.name, p.new_state, p.new_anchors
        ),
    ),
    "retire_theme": Action(
        "retire_theme",
        "Retire a theme.",
        RetireThemeInput,
        lambda p, ctx: store_themes.retire_theme(ctx.graph, p.name),
    ),
    "set_preface": Action(
        "set_preface",
        "Set or update the graph preface — a short description of what this graph is about and whose it is.",
        SetPrefaceInput,
        lambda p, ctx: store_system.set_preface(ctx.graph, p.preface),
    ),
    "complete": Action(
        "complete",
        "Finish a session run.",
        CompleteInput,
        _complete,
        is_completion=True,
    ),
    "complete_theme_update": Action(
        "complete_theme_update",
        "Finish a theme update run.",
        CompleteThemeUpdateInput,
        _complete,
        is_completion=True,
    ),
}

GRAPH_READ_ACTIONS = [
    "search_graph",
    "get_node",
    "get_node_neighborhood",
    "get_session",
    "list_sessions",
]

GRAPH_WRITE_ACTIONS = [
    "create_node",
    "update_node",
    "delete_node",
    "create_edge",
    "update_edge",
    "delete_edge",
]

SESSION_ACTIONS = GRAPH_READ_ACTIONS + GRAPH_WRITE_ACTIONS + ["get_theme", "complete"]

THEME_ACTIONS = GRAPH_READ_ACTIONS + [
    "get_theme",
    "create_theme",
    "update_theme",
    "retire_theme",
    "set_preface",
    "complete_theme_update",
]


def _slim_schema(schema: dict) -> dict:
    """Compact a Pydantic JSON schema for LLM tool definitions.

    Resolves $ref pointers inline, strips title fields, and simplifies
    Optional (anyOf-with-null) wrappers so the schema is shorter in tokens.
    """
    defs = schema.get("$defs", {})

    def _resolve(node: object) -> object:
        if isinstance(node, list):
            return [_resolve(item) for item in node]
        if not isinstance(node, dict):
            return node

        if "$ref" in node:
            ref = node["$ref"].rsplit("/", 1)[-1]
            return _resolve(dict(defs[ref])) if ref in defs else node

        if "anyOf" in node:
            non_null = [v for v in node["anyOf"] if v != {"type": "null"}]
            has_null = any(v == {"type": "null"} for v in node["anyOf"])
            if len(non_null) == 1 and has_null:
                merged = _resolve(non_null[0])
                if isinstance(merged, dict):
                    for k, v in node.items():
                        if k not in ("anyOf", "title"):
                            merged.setdefault(k, v)
                return merged

        return {k: _resolve(v) for k, v in node.items() if k not in ("title", "$defs")}

    return _resolve(schema)  # type: ignore[return-value]


def get_action_definitions(names: list[str]) -> list[dict]:
    return [
        {
            "type": "function",
            "function": {
                "name": ACTIONS[name].name,
                "description": ACTIONS[name].description,
                "parameters": _slim_schema(
                    ACTIONS[name].input_model.model_json_schema()
                ),
            },
        }
        for name in names
    ]
