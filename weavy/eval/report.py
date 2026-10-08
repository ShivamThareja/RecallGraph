"""Evaluation result models, aggregation, and rendering."""

from __future__ import annotations

import json
import math
import statistics
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

from pydantic import BaseModel, Field

REPORT_SCHEMA_VERSION = 1

STANDARD_NOTES = [
    "Embedding calls are not recorded in RunTraces, so estimated_cost_usd "
    "covers LLM completions only.",
    "LLM outputs are non-deterministic; repeat runs can differ even with an "
    "identical dataset, models, and configuration.",
]


class Usage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    reasoning_tokens: int = 0
    cached_tokens: int = 0
    total_tokens: int = 0
    estimated_cost_usd: float | None = 0.0


class IngestionResult(BaseModel):
    conversation_id: str
    local_session_id: str
    session_id: str | None = None
    status: str
    error: str | None = None
    latency_s: float
    theme_latency_s: float | None = None
    theme_status: str | None = None
    tool_calls: int = 0
    nodes_created: int = 0
    edges_created: int = 0
    usage: Usage = Field(default_factory=Usage)


class QuestionResult(BaseModel):
    conversation_id: str
    question_id: str
    category: str
    abstain: bool
    question: str
    gold_answer: str
    answer: str = ""
    status: str
    error: str | None = None
    correct: bool = False
    judge_reason: str = ""
    query_time: datetime
    gold_sessions: list[str] = Field(default_factory=list)
    surfaced_sessions: list[str] = Field(default_factory=list)
    read_sessions: list[str] = Field(default_factory=list)
    cited_sessions: list[str] = Field(default_factory=list)
    unsupported_citations: list[str] = Field(default_factory=list)
    session_recall: float | None = None
    session_precision: float | None = None
    fact_recall: float | None = None
    citation_precision: float | None = None
    citation_support: float | None = None
    latency_s: float
    tool_calls: int = 0
    turns: int = 0
    usage: Usage = Field(default_factory=Usage)
    judge_usage: Usage = Field(default_factory=Usage)
    run_id: str | None = None


class EvalReport(BaseModel):
    schema_version: int = REPORT_SCHEMA_VERSION
    started_at: datetime
    ended_at: datetime
    config: dict[str, Any]
    dataset: dict[str, Any]
    ingestion: list[IngestionResult]
    questions: list[QuestionResult]
    summary: dict[str, Any] = Field(default_factory=dict)
    notes: list[str] = Field(default_factory=lambda: list(STANDARD_NOTES))


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


def _mean(values: Iterable[float | None]) -> dict[str, Any]:
    present = [v for v in values if v is not None]
    return {
        "mean": statistics.fmean(present) if present else None,
        "n": len(present),
    }


def _percentile(sorted_values: list[float], pct: float) -> float:
    """Nearest-rank percentile over an already-sorted, non-empty list."""
    rank = max(1, math.ceil(pct / 100 * len(sorted_values)))
    return sorted_values[min(rank, len(sorted_values)) - 1]


def _latency(values: list[float]) -> dict[str, Any]:
    if not values:
        return {"mean": None, "p50": None, "p95": None, "max": None, "n": 0}
    ordered = sorted(values)
    return {
        "mean": statistics.fmean(ordered),
        "p50": statistics.median(ordered),
        "p95": _percentile(ordered, 95),
        "max": ordered[-1],
        "n": len(ordered),
    }


def _sum_usage(usages: Iterable[Usage]) -> dict[str, Any]:
    total = Usage()
    cost_known = True
    for u in usages:
        total.prompt_tokens += u.prompt_tokens
        total.completion_tokens += u.completion_tokens
        total.reasoning_tokens += u.reasoning_tokens
        total.cached_tokens += u.cached_tokens
        total.total_tokens += u.total_tokens
        if u.estimated_cost_usd is None:
            cost_known = False
        else:
            total.estimated_cost_usd = (total.estimated_cost_usd or 0.0) + (
                u.estimated_cost_usd
            )
    data = total.model_dump()
    data["cost_known"] = cost_known
    if not cost_known:
        # A partial sum would understate cost; report the known part separately.
        data["known_cost_usd"] = data["estimated_cost_usd"]
        data["estimated_cost_usd"] = None
    return data


def _question_block(rows: list[QuestionResult]) -> dict[str, Any]:
    n = len(rows)
    correct = sum(r.correct for r in rows)
    answerable = [r for r in rows if not r.abstain]
    return {
        "questions": n,
        "correct": correct,
        "accuracy": correct / n if n else None,
        "failed_runs": sum(r.status != "completed" for r in rows),
        "session_recall": _mean(r.session_recall for r in answerable),
        "session_precision": _mean(r.session_precision for r in answerable),
        "fact_recall": _mean(r.fact_recall for r in answerable),
        "citation_precision": _mean(r.citation_precision for r in rows),
        "citation_support": _mean(r.citation_support for r in rows),
        "unsupported_citations": sum(len(r.unsupported_citations) for r in rows),
        "latency_s": _latency([r.latency_s for r in rows]),
        "tool_calls": _mean(float(r.tool_calls) for r in rows),
        "turns": _mean(float(r.turns) for r in rows),
    }


def summarize(
    ingestion: list[IngestionResult], questions: list[QuestionResult]
) -> dict[str, Any]:
    by_category: dict[str, list[QuestionResult]] = defaultdict(list)
    for row in questions:
        by_category[row.category].append(row)

    return {
        "overall": _question_block(questions),
        "by_category": {
            cat: _question_block(rows) for cat, rows in sorted(by_category.items())
        },
        "ingestion": {
            "sessions": len(ingestion),
            "failed": sum(r.status != "completed" for r in ingestion),
            "theme_failures": sum(
                r.theme_status is not None and r.theme_status != "completed"
                for r in ingestion
            ),
            "latency_s": _latency([r.latency_s for r in ingestion]),
            "theme_latency_s": _latency(
                [r.theme_latency_s for r in ingestion if r.theme_latency_s is not None]
            ),
            "tool_calls": sum(r.tool_calls for r in ingestion),
            "nodes_created": sum(r.nodes_created for r in ingestion),
            "edges_created": sum(r.edges_created for r in ingestion),
        },
        "usage": {
            "ingestion": _sum_usage(r.usage for r in ingestion),
            "query": _sum_usage(r.usage for r in questions),
            "judge": _sum_usage(r.judge_usage for r in questions),
        },
    }


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------


def write_json(report: EvalReport, path: str | Path) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report.model_dump(mode="json"), indent=2) + "\n")
    return target


def _pct(value: float | None) -> str:
    return "  n/a" if value is None else f"{value * 100:5.1f}%"


def _metric(block: dict[str, Any]) -> str:
    return _pct(block["mean"]) + f" (n={block['n']})"


def _secs(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.2f}s"


def _cost(usage: dict[str, Any]) -> str:
    if usage["cost_known"]:
        return f"${usage['estimated_cost_usd']:.4f}"
    return f"unknown (priced part ${usage['known_cost_usd']:.4f})"


def render_text(report: EvalReport) -> str:
    s = report.summary
    o = s["overall"]
    cfg = report.config
    lines = [
        f"Weavy eval — dataset {report.dataset['name']} v{report.dataset['version']} "
        f"({report.dataset['questions']} questions, sha256 {str(report.dataset['sha256'])[:12]})",
        f"LLM {cfg['llm_model']} | embeddings {cfg['embedding_model']} | "
        f"judge {cfg['judge']}{' (' + cfg['judge_model'] + ')' if cfg.get('judge_model') else ''} | "
        f"themes {'on' if cfg['update_themes'] else 'off'}",
        "",
        f"Accuracy            {_pct(o['accuracy'])}  ({o['correct']}/{o['questions']}, "
        f"{o['failed_runs']} failed runs)",
        f"Session recall      {_metric(o['session_recall'])}",
        f"Session precision   {_metric(o['session_precision'])}",
        f"Fact recall         {_metric(o['fact_recall'])}",
        f"Citation precision  {_metric(o['citation_precision'])}",
        f"Citation support    {_metric(o['citation_support'])}  "
        f"({o['unsupported_citations']} unsupported citations)",
        f"Query latency       mean {_secs(o['latency_s']['mean'])}  "
        f"p50 {_secs(o['latency_s']['p50'])}  p95 {_secs(o['latency_s']['p95'])}",
        f"Tool calls / query  {o['tool_calls']['mean'] or 0:.1f}",
        "",
        "By category:",
    ]
    for cat, block in s["by_category"].items():
        lines.append(
            f"  {cat:<18} acc {_pct(block['accuracy'])} ({block['correct']}/{block['questions']})"
            f"  recall {_pct(block['session_recall']['mean'])}"
            f"  fact {_pct(block['fact_recall']['mean'])}"
        )
    ing = s["ingestion"]
    lines += [
        "",
        f"Ingestion: {ing['sessions']} sessions, {ing['failed']} failed, "
        f"{ing['nodes_created']} nodes / {ing['edges_created']} edges created, "
        f"mean {_secs(ing['latency_s']['mean'])}/session",
        "",
        "Usage:",
    ]
    for phase in ("ingestion", "query", "judge"):
        u = s["usage"][phase]
        lines.append(
            f"  {phase:<10} prompt {u['prompt_tokens']:>9,}  completion "
            f"{u['completion_tokens']:>8,}  cost {_cost(u)}"
        )
    lines += ["", "Notes:"] + [f"  - {n}" for n in report.notes]
    return "\n".join(lines)
