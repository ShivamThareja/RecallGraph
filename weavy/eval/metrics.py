"""Per-question metrics computed from a query RunTrace.

Definitions (all session ids are Weavy ``s:N`` ids, after mapping the dataset's
local evidence ids):

* **surfaced sessions** — every session id that reached the agent's context
  as provenance of a retrieved memory (``source_id`` / ``mentioned_by`` /
  ``source_sessions`` anywhere in a successful read-tool result) plus the
  sessions opened with ``get_session``. ``list_sessions`` is excluded. The
  definition is shared with runtime citation checks — see
  ``weavy.services.provenance``.
* **read sessions** — sessions whose raw text the agent actually opened with
  ``get_session``.
* ``session_recall``     = |gold ∩ surfaced| / |gold|           (None if no gold)
* ``session_precision``  = |gold ∩ read| / |read|               (None if none read)
* ``fact_recall``        = evidence keywords present in successful read-tool
  results / evidence keywords                                  (None if none)
* ``citation_precision`` = |cited ∩ gold| / |cited|             (None if none cited)
* ``citation_support``   = |cited ∩ surfaced| / |cited|         (None if none cited)

The query's own session is excluded everywhere: a query run may write nodes
whose provenance is the query session itself, which is not evidence.
"""

from __future__ import annotations

from typing import Any

from weavy.eval.judges import keyword_group_matches
from weavy.models.traces import RunTrace
from weavy.services.provenance import gather_evidence, trace_tool_calls


def _evidence(trace: RunTrace):
    return gather_evidence(trace_tool_calls(trace), trace.session_id)


def surfaced_sessions(trace: RunTrace) -> set[str]:
    return _evidence(trace).surfaced_sessions


def read_sessions(trace: RunTrace) -> set[str]:
    return _evidence(trace).read_sessions


def retrieved_text(trace: RunTrace) -> str:
    """Concatenated successful read-tool results — what entered the agent's context."""
    return "\n".join(_evidence(trace).retrieved_text)


def tool_call_count(trace: RunTrace) -> int:
    return sum(len(turn.tool_calls) for turn in trace.turns)


def cited_sessions(trace: RunTrace) -> list[str]:
    cited = (trace.completion_payload or {}).get("cited_sources") or []
    return [c for c in dict.fromkeys(cited) if isinstance(c, str)]


def _ratio(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def retrieval_metrics(
    trace: RunTrace,
    gold_sessions: set[str],
    evidence_keywords: list[str],
) -> dict[str, Any]:
    evidence = _evidence(trace)
    surfaced = evidence.surfaced_sessions
    read = evidence.read_sessions
    cited = cited_sessions(trace)
    cited_set = set(cited)
    text = "\n".join(evidence.retrieved_text)
    facts_found = [k for k in evidence_keywords if keyword_group_matches(k, text)]
    return {
        "surfaced_sessions": sorted(surfaced),
        "read_sessions": sorted(read),
        "cited_sessions": cited,
        "session_recall": _ratio(len(gold_sessions & surfaced), len(gold_sessions)),
        "session_precision": _ratio(len(gold_sessions & read), len(read)),
        "fact_recall": _ratio(len(facts_found), len(evidence_keywords)),
        "citation_precision": _ratio(len(cited_set & gold_sessions), len(cited_set)),
        "citation_support": _ratio(len(cited_set & surfaced), len(cited_set)),
        "unsupported_citations": sorted(cited_set - surfaced),
    }
