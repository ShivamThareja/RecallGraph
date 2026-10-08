"""
Agent tool input contracts.

These schemas belong to the harness boundary only. They define what the model is
allowed to call, not the general application contract for lower layers.
"""

from typing import Annotated

from pydantic import AfterValidator, BaseModel, Field, StringConstraints

from weavy.models.graph import ProvenanceInput
from weavy.timeutil import UTCDatetime, normalize_range

NodeId = Annotated[str, StringConstraints(pattern=r"^node:\d+$")]
EdgeId = Annotated[str, StringConstraints(pattern=r"^edge:\d+$")]


# Resolved event/valid time of this fact — when it became true in the world (not when
# it was discussed). Optional; defaults to the episode time when no temporal cue exists.
# Naive values are read as UTC.
HappenedAt = UTCDatetime | None


# [start, end] with either bound null for an open range; naive values read as UTC.
# A list (not a tuple) avoids Gemini's prefixItems rejection.
def _check_range(value: list) -> list:
    normalize_range(value)  # raises ValueError on malformed / inverted ranges
    return value


TimeRange = Annotated[list[UTCDatetime | None], AfterValidator(_check_range)] | None


class CreateNodeInput(BaseModel):
    aliases: list[str]
    summary: str
    note: str
    provenance: ProvenanceInput
    happened_at: HappenedAt = None
    # Creation is refused when an existing node likely denotes the same entity;
    # set force=true only after confirming the entity is genuinely distinct.
    force: bool = False


class UpdateNodeInput(BaseModel):
    node_id: NodeId
    note: str
    new_summary: str | None = None
    new_aliases: list[str] | None = None
    provenance: ProvenanceInput
    happened_at: HappenedAt = None


class CreateEdgeInput(BaseModel):
    from_node_id: NodeId
    to_node_id: NodeId
    label: str  # short directional relationship type, e.g. "works at"
    fact: str  # natural-language statement of the relationship; searchable
    note: str  # why this edge is being written now (logged)
    happened_at: HappenedAt = None


class UpdateEdgeInput(BaseModel):
    edge_id: EdgeId
    note: str  # why this edge is changing (logged)
    new_label: str | None = None
    new_fact: str | None = None
    happened_at: HappenedAt = None


class DeleteNodeInput(BaseModel):
    node_id: NodeId
    reason: str


class DeleteEdgeInput(BaseModel):
    edge_id: EdgeId
    reason: str


class SearchGraphInput(BaseModel):
    query: str
    limit: int = Field(default=10, ge=1)
    # [start, end] — filter by happened_at (valid time), falling back to record
    # time. Applied before the top-`limit` cut. Either bound may be null.
    time_range: TimeRange = None


class GetNodeNeighborhoodInput(BaseModel):
    node_id: NodeId
    # Max neighbours returned (default NEIGHBORHOOD_LIMIT); the output reports
    # the total so truncation is visible.
    limit: int | None = Field(default=None, ge=1)


class GetNodeInput(BaseModel):
    node_ids: list[NodeId]


class GetSessionInput(BaseModel):
    session_id: Annotated[str, StringConstraints(pattern=r"^s:\d+$")]


class ListSessionsInput(BaseModel):
    date_range: TimeRange = None  # [start, end]; either bound may be null
    limit: int = Field(default=20, ge=1)


class GetThemeInput(BaseModel):
    name: str


class CreateThemeInput(BaseModel):
    name: str
    state: str
    anchors: list[NodeId]


class UpdateThemeInput(BaseModel):
    name: str
    new_state: str | None = None
    new_anchors: list[NodeId] | None = None


class RetireThemeInput(BaseModel):
    name: str


class SetPrefaceInput(BaseModel):
    preface: str


class CompleteInput(BaseModel):
    summary: str
    answer: str | None = None
    cited_sources: list[str] = Field(default_factory=list)
    consulted_nodes: list[str] = Field(default_factory=list)


class CompleteThemeUpdateInput(BaseModel):
    updated_themes: list[str]
