"""What did a run actually retrieve, and are its citations backed by that?

A model can cite any ``s:N`` / ``node:N`` string it likes. This module derives
the *evidence* a run really had in context from its recorded tool calls and
checks the completion's ``cited_sources`` / ``consulted_nodes`` against it.

Evidence sources (successful read-tool calls only — ``list_sessions`` lists
sessions without retrieving their content, so it does not count):

* **surfaced sessions** — provenance attached to retrieved memories:
  ``source_id``, ``mentioned_by`` and ``source_sessions`` values anywhere in
  a read-tool result, plus every session opened with ``get_session``.
* **read sessions** — sessions whose raw text was opened with ``get_session``.
* **seen memories** — every ``node:N`` / ``edge:N`` id present in a read-tool
  result (or requested by id in ``get_node`` / ``get_node_neighborhood``).

A citation is **verified** when it is in this evidence and **unsupported**
otherwise. The run's own session is never evidence for itself.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Iterable

from weavy.models.traces import RunTrace, ToolCall

READ_TOOLS = frozenset(
    {"search_graph", "get_node", "get_node_neighborhood", "get_session", "get_theme"}
)
SESSION_ID_RE = re.compile(r"^s:\d+$")
MEMORY_ID_RE = re.compile(r"^(?:node|edge):\d+$")
_SESSION_LIST_KEYS = frozenset({"mentioned_by", "source_sessions"})


def collect_session_ids(value: Any, out: set[str]) -> None:
    """Add provenance session ids found anywhere in a decoded tool result."""
    if isinstance(value, dict):
        for key, child in value.items():
            if key == "source_id" and isinstance(child, str):
                if SESSION_ID_RE.match(child):
                    out.add(child)
            elif key in _SESSION_LIST_KEYS and isinstance(child, list):
                out.update(
                    s for s in child if isinstance(s, str) and SESSION_ID_RE.match(s)
                )
            else:
                collect_session_ids(child, out)
    elif isinstance(value, list):
        for child in value:
            collect_session_ids(child, out)


def collect_memory_ids(value: Any, out: set[str]) -> None:
    """Add every node:N / edge:N string value found in a decoded tool result."""
    if isinstance(value, dict):
        for child in value.values():
            collect_memory_ids(child, out)
    elif isinstance(value, list):
        for child in value:
            collect_memory_ids(child, out)
    elif isinstance(value, str) and MEMORY_ID_RE.match(value):
        out.add(value)


@dataclass
class Evidence:
    surfaced_sessions: set[str] = field(default_factory=set)
    read_sessions: set[str] = field(default_factory=set)
    seen_memories: set[str] = field(default_factory=set)
    retrieved_text: list[str] = field(default_factory=list)


def successful_read_calls(calls: Iterable[ToolCall]) -> Iterable[ToolCall]:
    for call in calls:
        if call.tool_name in READ_TOOLS and call.error is None and call.result:
            yield call


def trace_tool_calls(trace: RunTrace) -> list[ToolCall]:
    return [call for turn in trace.turns for call in turn.tool_calls]


def gather_evidence(
    calls: Iterable[ToolCall], exclude_session: str | None = None
) -> Evidence:
    evidence = Evidence()
    for call in successful_read_calls(calls):
        evidence.retrieved_text.append(call.result or "")
        try:
            decoded = json.loads(call.result or "")
        except json.JSONDecodeError:
            decoded = None
        collect_session_ids(decoded, evidence.surfaced_sessions)
        collect_memory_ids(decoded, evidence.seen_memories)
        if call.tool_name == "get_session":
            sid = call.args.get("session_id")
            if isinstance(sid, str):
                evidence.surfaced_sessions.add(sid)
                evidence.read_sessions.add(sid)
        elif call.tool_name == "get_node":
            evidence.seen_memories.update(
                n for n in call.args.get("node_ids") or [] if isinstance(n, str)
            )
        elif call.tool_name == "get_node_neighborhood":
            nid = call.args.get("node_id")
            if isinstance(nid, str):
                evidence.seen_memories.add(nid)
    if exclude_session:
        evidence.surfaced_sessions.discard(exclude_session)
        evidence.read_sessions.discard(exclude_session)
    return evidence


def _unique_strings(values: Any) -> list[str]:
    if not isinstance(values, list):
        return []
    return [v for v in dict.fromkeys(values) if isinstance(v, str)]


def verify_citations(trace: RunTrace) -> dict[str, list[str]] | None:
    """Check the completion's citations against the run's evidence.

    Returns None when the completion makes no citations at all.
    """
    payload = trace.completion_payload or {}
    cited = _unique_strings(payload.get("cited_sources"))
    consulted = _unique_strings(payload.get("consulted_nodes"))
    if not cited and not consulted:
        return None
    evidence = gather_evidence(trace_tool_calls(trace), trace.session_id)
    return {
        "verified_sources": [s for s in cited if s in evidence.surfaced_sessions],
        "unsupported_sources": [
            s for s in cited if s not in evidence.surfaced_sessions
        ],
        "read_sources": [s for s in cited if s in evidence.read_sessions],
        "verified_nodes": [n for n in consulted if n in evidence.seen_memories],
        "unsupported_nodes": [n for n in consulted if n not in evidence.seen_memories],
    }


def annotate_citations(trace: RunTrace) -> None:
    """Attach ``citation_check`` to the completion payload (no-op without citations)."""
    check = verify_citations(trace)
    if check is not None and trace.completion_payload is not None:
        trace.completion_payload["citation_check"] = check
