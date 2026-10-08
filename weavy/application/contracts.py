from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field

from weavy.models.graph import SemanticEdge, SemanticNode
from weavy.models.themes import Theme


class OperationResult(BaseModel):
    ok: bool
    id: str | None = None
    message: str | None = None


class SearchResult(BaseModel):
    """A single hit in the unified ranked search.

    ``kind="node"`` rows describe an entity (``id`` is ``node:N``, ``text`` is
    the entity's summary line). ``kind="edge"`` rows describe a relationship fact
    (``id`` is ``edge:N``, ``text`` is the fact, and ``endpoints`` names the two
    nodes it connects so the agent can traverse from a fact to its entities).
    The graph is the sole search surface; ground-truth episodes are reached by
    navigation (``get_node``'s ``mentioned_by`` or an edge's ``source_id`` →
    ``get_session``), never returned as a search result kind here.
    """

    kind: Literal["node", "edge"]
    id: str
    label: str  # node: canonical alias; edge: relationship label
    text: str  # node: summary line; edge: the fact
    distance: (
        float | None
    )  # vector distance (lower = closer); None = keyword-only hit, no embedding comparison made
    edge_count: int | None = None  # node only: degree, for hub identification
    endpoints: list[str] | None = None  # edge only: [from_node_id, to_node_id]
    # Provenance: the s:N episodes behind this memory, most recent first
    # (capped; ``source_count`` is the full number). Navigate with get_session.
    source_sessions: list[str] = Field(default_factory=list)
    source_count: int = 0
    # Valid-time span of the memory's log (happened_at, falling back to record
    # time) and when it was last recorded.
    earliest_happened_at: datetime | None = None
    latest_happened_at: datetime | None = None
    last_recorded_at: datetime | None = None
    # Graph-aware expansion: set when this row was not a direct hit but a
    # neighbour of one. ``via`` is the direct hit it hangs off, ``via_edge``
    # the connecting relationship, ``hops`` the distance from it.
    via: str | None = None
    via_edge: str | None = None
    hops: int | None = None


class SearchDiagnostics(BaseModel):
    """How the result list was produced — makes filtering visible to the agent."""

    candidates: int = 0  # distinct candidates before any filtering
    dropped_by_distance: int = 0
    dropped_by_time: int = 0
    expanded: int = 0  # neighbour rows added by graph expansion
    max_distance: float | None = None
    time_range: list[datetime | None] | None = None


class SearchGraphOutput(BaseModel):
    results: list[SearchResult]
    diagnostics: SearchDiagnostics | None = None


class NeighborSummary(BaseModel):
    edge_id: str
    edge_label: str
    edge_fact: str | None = None
    node_id: str
    canonical_alias: str
    summary_line: str


class GetNodeNeighborhoodOutput(BaseModel):
    node: SemanticNode
    neighbors: list[NeighborSummary]
    total_neighbors: int | None = None  # before the limit was applied
    truncated: bool = False


class GetNodeResult(BaseModel):
    node: SemanticNode
    edges: list[SemanticEdge] = Field(default_factory=list)
    mentioned_by: list[str] = Field(
        default_factory=list
    )  # s:N episodes that touched this node


class GetSessionOutput(BaseModel):
    id: str
    timestamp: datetime
    text: str  # raw episode content (concatenated user messages)
    summary: str | None = None


class GetNodeOutput(BaseModel):
    results: list[GetNodeResult]
    not_found: list[str] = Field(default_factory=list)


class SessionSummary(BaseModel):
    id: str
    timestamp: datetime
    summary: str | None = None


class ListSessionsOutput(BaseModel):
    sessions: list[SessionSummary]


class GetThemeOutput(BaseModel):
    theme: Theme


class CompletedSessionRow(BaseModel):
    id: str
    summary: str
    graph_changes: dict
    completed_at: str
