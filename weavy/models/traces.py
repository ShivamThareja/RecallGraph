import uuid
from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field

RunMode = Literal["ingestion", "query", "theme"]


class TouchedNode(BaseModel):
    node_id: str
    action: Literal["created", "updated", "deleted"]


class TouchedEdge(BaseModel):
    edge_id: str
    action: Literal["created", "updated", "deleted"]


class ToolCall(BaseModel):
    tool_name: str
    args: dict[str, Any]
    result: str | None = None
    error: str | None = None
    called_at: datetime


class TurnUsage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    reasoning_tokens: int = 0
    cached_tokens: int = 0
    total_tokens: int = 0


class Turn(BaseModel):
    turn_number: int
    input_messages: list[dict[str, Any]] = Field(default_factory=list)
    reasoning_content: str | None = None
    text_content: str | None = None
    tool_calls: list[ToolCall] = Field(default_factory=list)
    usage: TurnUsage = Field(default_factory=TurnUsage)
    timestamp: datetime


def graph_changes(
    touched_nodes: list["TouchedNode"],
    touched_edges: list["TouchedEdge"],
) -> dict[str, list[str]]:
    """Return ID-lists grouped by action, e.g. {"nodes_created": ["node:1"]}."""
    result: dict[str, list[str]] = {}
    for action in ("created", "updated", "deleted"):
        node_ids = [n.node_id for n in touched_nodes if n.action == action]
        edge_ids = [e.edge_id for e in touched_edges if e.action == action]
        if node_ids:
            result[f"nodes_{action}"] = node_ids
        if edge_ids:
            result[f"edges_{action}"] = edge_ids
    return result


def graph_delta(
    touched_nodes: list["TouchedNode"],
    touched_edges: list["TouchedEdge"],
) -> dict[str, int]:
    """Summarise write activity as action-count pairs, e.g. nodes_created=3."""
    return {k: len(v) for k, v in graph_changes(touched_nodes, touched_edges).items()}


class RunTrace(BaseModel):
    run_id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    mode: RunMode
    started_at: datetime
    ended_at: datetime | None = None
    input_summary: str
    turns: list[Turn] = Field(default_factory=list)
    total_usage: TurnUsage = Field(default_factory=TurnUsage)
    completion_payload: dict[str, Any] | None = None
    touched_nodes: list[TouchedNode] = Field(default_factory=list)
    touched_edges: list[TouchedEdge] = Field(default_factory=list)
    status: Literal["running", "completed", "failed"] = "running"
    error: str | None = None
    conversation: list[dict] | None = None  # full messages excluding system prompt
    session_id: str | None = None
