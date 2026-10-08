from pydantic import BaseModel, ConfigDict, Field, field_validator

from weavy.timeutil import UTCDatetime


class LogEntry(BaseModel):
    # All times are UTC-aware; legacy naive values are read as UTC.
    source_id: str  # s:N
    timestamp: UTCDatetime  # transaction time — when this fact was recorded/discussed
    happened_at: UTCDatetime | None = (
        None  # valid time — when the fact became true in the world
    )
    note: str


class ProvenanceInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source_id: str  # s:N

    @field_validator("source_id")
    @classmethod
    def validate_source_id(cls, v: str) -> str:
        if not v.startswith("s:"):
            raise ValueError("source_id must be s:N")
        return v


class SemanticNode(BaseModel):
    id: str  # node:N
    aliases: list[str]  # aliases[0] is canonical name; min 1
    summary: str
    total_log_count: int = 0
    log: list[LogEntry] = Field(default_factory=list)

    @field_validator("aliases")
    @classmethod
    def aliases_nonempty(cls, v: list[str]) -> list[str]:
        if not v:
            raise ValueError("aliases must have at least one entry")
        return v


class SemanticEdge(BaseModel):
    id: str  # edge:N
    from_node_id: str
    to_node_id: str
    label: str  # short directional relationship type, e.g. "works at"
    fact: str  # natural-language statement of the relationship; embedded & searchable
    total_log_count: int = 0
    log: list[LogEntry] = Field(default_factory=list)
    source_id: str | None = None  # s:N — which session first created this edge
