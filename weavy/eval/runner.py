"""Evaluation orchestration.

For each conversation the runner:

1. opens an isolated graph ``<prefix>_<dataset>_<conversation>`` and resets it,
2. ingests the sessions in timestamp order (theme pass timed separately),
3. maps the dataset's local session ids to the ``s:N`` ids Weavy minted,
4. asks every question at its fixed ``query_time`` and scores the RunTrace.

The memory system is reached only through ``MemoryClient`` — the ``Weavy``
SDK satisfies it, and tests substitute a fake so no FalkorDB or LLM is needed.
A failing run or exception is recorded on its row; the evaluation continues.
"""

from __future__ import annotations

import re
import time
from datetime import datetime, timezone
from typing import Callable, Protocol

from weavy.config import settings
from weavy.eval.dataset import EvalConversation, EvalDataset
from weavy.eval.judges import Judge
from weavy.eval.metrics import retrieval_metrics, tool_call_count
from weavy.eval.report import (
    EvalReport,
    IngestionResult,
    QuestionResult,
    Usage,
    summarize,
)
from weavy.models.traces import RunTrace
from weavy.services.cost import estimate_cost


class MemoryClient(Protocol):
    def reset(self) -> None: ...

    def add(
        self,
        text: str,
        *,
        timestamp: datetime | None = None,
        context: str | None = None,
        update_themes: bool = True,
    ) -> RunTrace: ...

    def update_themes(self) -> RunTrace: ...

    def query(
        self,
        question: str,
        *,
        context: str | None = None,
        query_time: datetime | None = None,
    ) -> RunTrace: ...


ClientFactory = Callable[[str], MemoryClient]
Progress = Callable[[str], None]


def eval_graph_name(prefix: str, dataset_name: str, conversation_id: str) -> str:
    raw = f"{prefix}_{dataset_name}_{conversation_id}"
    return re.sub(r"[^A-Za-z0-9_-]", "_", raw)


def check_graph_names(names: list[str], protected: str) -> None:
    """Refuse to evaluate in (and therefore reset) the user's real memory graph."""
    if protected in names:
        raise ValueError(
            f"refusing to run: eval graph {protected!r} is the configured "
            f"GRAPH_NAME and would be wiped. Choose a different --graph-prefix."
        )
    if len(set(names)) != len(names):
        raise ValueError("eval graph names collide after sanitizing conversation ids")


def _usage(trace: RunTrace | None, model: str) -> Usage:
    if trace is None:
        return Usage()
    u = trace.total_usage
    return Usage(
        prompt_tokens=u.prompt_tokens,
        completion_tokens=u.completion_tokens,
        reasoning_tokens=u.reasoning_tokens,
        cached_tokens=u.cached_tokens,
        total_tokens=u.total_tokens,
        estimated_cost_usd=estimate_cost(model, u.prompt_tokens, u.completion_tokens),
    )


def _add_usage(a: Usage, b: Usage) -> Usage:
    cost = (
        None
        if a.estimated_cost_usd is None or b.estimated_cost_usd is None
        else a.estimated_cost_usd + b.estimated_cost_usd
    )
    return Usage(
        prompt_tokens=a.prompt_tokens + b.prompt_tokens,
        completion_tokens=a.completion_tokens + b.completion_tokens,
        reasoning_tokens=a.reasoning_tokens + b.reasoning_tokens,
        cached_tokens=a.cached_tokens + b.cached_tokens,
        total_tokens=a.total_tokens + b.total_tokens,
        estimated_cost_usd=cost,
    )


def _created(trace: RunTrace) -> tuple[int, int]:
    nodes = sum(n.action == "created" for n in trace.touched_nodes)
    edges = sum(e.action == "created" for e in trace.touched_edges)
    return nodes, edges


def _ingest(
    client: MemoryClient,
    conversation: EvalConversation,
    *,
    update_themes: bool,
    model: str,
    progress: Progress,
) -> tuple[list[IngestionResult], dict[str, str]]:
    results: list[IngestionResult] = []
    session_map: dict[str, str] = {}
    for session in conversation.sessions_in_order():
        progress(f"  ingest {conversation.id}/{session.id}")
        t0 = time.perf_counter()
        try:
            trace = client.add(
                session.text,
                # Weavy renders session times as UTC ("...Z"); normalize first.
                timestamp=session.timestamp.astimezone(timezone.utc),
                context=conversation.context,
                update_themes=False,
            )
        except Exception as exc:  # recorded, evaluation continues
            results.append(
                IngestionResult(
                    conversation_id=conversation.id,
                    local_session_id=session.id,
                    status="error",
                    error=f"{type(exc).__name__}: {exc}",
                    latency_s=time.perf_counter() - t0,
                )
            )
            continue
        latency = time.perf_counter() - t0
        if trace.session_id:
            session_map[session.id] = trace.session_id
        nodes, edges = _created(trace)
        row = IngestionResult(
            conversation_id=conversation.id,
            local_session_id=session.id,
            session_id=trace.session_id,
            status=trace.status,
            error=trace.error,
            latency_s=latency,
            tool_calls=tool_call_count(trace),
            nodes_created=nodes,
            edges_created=edges,
            usage=_usage(trace, model),
        )
        # Mirror Weavy.add(): themes only after a successful ingestion, but
        # timed separately so ingestion latency is not conflated with themes.
        if update_themes and trace.status == "completed":
            t1 = time.perf_counter()
            try:
                theme_trace = client.update_themes()
                row.theme_status = theme_trace.status
                row.usage = _add_usage(row.usage, _usage(theme_trace, model))
            except Exception as exc:
                row.theme_status = "error"
                row.error = f"theme update: {type(exc).__name__}: {exc}"
            row.theme_latency_s = time.perf_counter() - t1
        results.append(row)
    return results, session_map


def _ask(
    client: MemoryClient,
    conversation: EvalConversation,
    session_map: dict[str, str],
    judge: Judge,
    *,
    model: str,
    progress: Progress,
) -> list[QuestionResult]:
    rows: list[QuestionResult] = []
    for q in conversation.questions:
        progress(f"  query  {conversation.id}/{q.id}")
        query_time = conversation.resolve_query_time(q)
        # An evidence session that failed to ingest keeps a placeholder so it
        # still counts in recall's denominator (it is genuinely unretrievable).
        gold = {session_map.get(sid, f"unmapped:{sid}") for sid in q.evidence_sessions}
        row = QuestionResult(
            conversation_id=conversation.id,
            question_id=q.id,
            category=q.category,
            abstain=q.abstain,
            question=q.question,
            gold_answer=q.answer,
            status="error",
            query_time=query_time,
            gold_sessions=sorted(gold),
            latency_s=0.0,
        )
        t0 = time.perf_counter()
        try:
            trace = client.query(
                q.question, context=conversation.context, query_time=query_time
            )
        except Exception as exc:
            row.latency_s = time.perf_counter() - t0
            row.error = f"{type(exc).__name__}: {exc}"
            row.judge_reason = "query raised; not judged"
            # A crashed run retrieved nothing: score it as zero rather than
            # dropping it from the averages, which would inflate recall.
            if gold:
                row.session_recall = 0.0
            if q.evidence_keywords:
                row.fact_recall = 0.0
            rows.append(row)
            continue
        row.latency_s = time.perf_counter() - t0
        row.status = trace.status
        row.error = trace.error
        row.run_id = trace.run_id
        row.turns = len(trace.turns)
        row.tool_calls = tool_call_count(trace)
        row.usage = _usage(trace, model)
        row.answer = str((trace.completion_payload or {}).get("answer") or "")
        for key, value in retrieval_metrics(trace, gold, q.evidence_keywords).items():
            setattr(row, key, value)

        if trace.status != "completed":
            row.judge_reason = "run failed; not judged"
        else:
            try:
                verdict = judge.judge(q, row.answer)
                row.correct = verdict.correct
                row.judge_reason = verdict.reason
                row.judge_usage = Usage(
                    prompt_tokens=verdict.prompt_tokens,
                    completion_tokens=verdict.completion_tokens,
                    total_tokens=verdict.prompt_tokens + verdict.completion_tokens,
                    estimated_cost_usd=verdict.cost_usd,
                )
            except Exception as exc:
                row.judge_reason = f"judge error: {type(exc).__name__}: {exc}"
        rows.append(row)
    return rows


def run_eval(
    dataset: EvalDataset,
    *,
    client_factory: ClientFactory,
    judge: Judge,
    graph_prefix: str = "weavy_eval",
    update_themes: bool = True,
    conversation_ids: list[str] | None = None,
    limit: int | None = None,
    cleanup: Callable[[str], None] | None = None,
    judge_model: str | None = None,
    progress: Progress = lambda _msg: None,
) -> EvalReport:
    """Run *dataset* end to end and return a fully aggregated report.

    Args:
        conversation_ids: Restrict to these conversations (default: all).
        limit: Maximum questions per conversation (ingestion is never limited,
            so every question sees the full memory).
        cleanup: Called with each graph name after its conversation finishes.
    """
    conversations = dataset.conversations
    if conversation_ids:
        unknown = set(conversation_ids) - {c.id for c in conversations}
        if unknown:
            raise ValueError(f"unknown conversation ids: {sorted(unknown)}")
        conversations = [c for c in conversations if c.id in conversation_ids]
    if limit is not None and limit < 1:
        raise ValueError("limit must be >= 1")

    names = [eval_graph_name(graph_prefix, dataset.name, c.id) for c in conversations]
    check_graph_names(names, settings.GRAPH_NAME)

    model = settings.LLM_MODEL
    started = datetime.now(tz=timezone.utc)
    ingestion: list[IngestionResult] = []
    questions: list[QuestionResult] = []

    for conversation, graph_name in zip(conversations, names):
        progress(f"conversation {conversation.id} → graph {graph_name}")
        client = client_factory(graph_name)
        client.reset()
        try:
            ing_rows, session_map = _ingest(
                client,
                conversation,
                update_themes=update_themes,
                model=model,
                progress=progress,
            )
            ingestion.extend(ing_rows)
            subset = conversation.model_copy(
                update={"questions": conversation.questions[:limit]}
            )
            questions.extend(
                _ask(
                    client,
                    subset,
                    session_map,
                    judge,
                    model=model,
                    progress=progress,
                )
            )
        finally:
            if cleanup is not None:
                cleanup(graph_name)

    report = EvalReport(
        started_at=started,
        ended_at=datetime.now(tz=timezone.utc),
        config={
            "llm_model": model,
            "embedding_model": settings.EMBEDDING_MODEL,
            "reasoning_effort": settings.REASONING_EFFORT,
            "judge": judge.name,
            "judge_model": judge_model,
            "update_themes": update_themes,
            "graph_prefix": graph_prefix,
            "graphs": names,
            "conversation_filter": conversation_ids,
            "question_limit_per_conversation": limit,
        },
        dataset={
            "name": dataset.name,
            "version": dataset.version,
            "description": dataset.description,
            "sha256": dataset.source_sha256,
            "conversations": len(conversations),
            "questions": len(questions),
        },
        ingestion=ingestion,
        questions=questions,
    )
    if judge.name == "keyword":
        report.notes.append(
            "Keyword judge: deterministic substring matching on required "
            "keywords. It can mark paraphrased correct answers wrong; use "
            "--judge llm for semantic grading."
        )
    report.summary = summarize(ingestion, questions)
    return report
