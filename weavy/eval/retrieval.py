"""Retrieval-level evaluation: score ``search_graph`` on a known graph, no LLM.

The QA eval (``runner.py``) measures the whole agent; its retrieval numbers
depend on what the LLM chose to call. This module isolates the retrieval
primitive: it builds the graph *directly* from a fixture (sessions, nodes with
dated observations, edges), runs each query through
``services.memory.search_graph`` — the exact function the agent's tool calls —
and compares results against labelled expectations.

Dataset format ``weavy-retrieval/v1``::

    {
      "format": "weavy-retrieval/v1", "name": "...", "version": "1",
      "description": "...",
      "sessions":  [{"id": "s-job", "timestamp": "...+00:00", "text": "..."}],
      "nodes": [{"id": "n-x", "aliases": ["X"], "summary": "...",
                 "observations": [{"session": "s-job", "note": "...",
                                   "happened_at": "...+00:00 | null"}]}],
      "edges": [{"id": "e-x", "from": "n-x", "to": "n-y", "label": "...",
                 "fact": "...", "session": "s-job", "happened_at": null}],
      "queries": [{"id": "q1", "query": "...", "category": "...",
                   "limit": 5, "time_range": [start|null, end|null] | null,
                   "expected": ["n-x", "e-x"],     # should be direct hits
                   "related": ["n-y"],             # may arrive via expansion
                   "expected_sessions": ["s-job"], # provenance to expose
                   "expect_empty": false}]
    }

``time_range`` bounds may be naive (exercising naive/aware handling) or null
(open-ended). Metrics per query:

* ``recall_at_k``    = expected ∩ direct results / expected
* ``precision_at_k`` = expected ∩ direct results / direct results
* ``mrr``            = 1 / rank of the first expected id among direct results
* ``related_recall`` = related ∩ all results (incl. graph-expanded) / related
* ``source_recall``  = expected_sessions ∩ sessions exposed on results / expected
* ``empty_correct``  = for ``expect_empty`` queries, whether nothing came back
"""

from __future__ import annotations

import hashlib
import json
import statistics
import time
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any, Callable

from falkordb import Graph
from pydantic import AwareDatetime, BaseModel, Field, model_validator

from weavy.config import settings
from weavy.eval.dataset import resolve_dataset_path
from weavy.models.canonical import ChatMessage, Session
from weavy.models.graph import ProvenanceInput
from weavy.models.traces import RunTrace
from weavy.services import embedding, memory
from weavy.store import canonical as store_canonical
from weavy.store.client import delete_graph_if_exists
from weavy.store.system import increment_counter, init_system

RETRIEVAL_FORMAT = "weavy-retrieval/v1"
RETRIEVAL_SCHEMA_VERSION = 1


class RSession(BaseModel):
    id: str
    timestamp: AwareDatetime
    text: str


class RObservation(BaseModel):
    session: str
    note: str = "observed"
    happened_at: AwareDatetime | None = None


class RNode(BaseModel):
    id: str
    aliases: list[str]
    summary: str
    observations: list[RObservation] = Field(min_length=1)


class REdge(BaseModel):
    model_config = {"populate_by_name": True}

    id: str
    from_: str = Field(alias="from")
    to: str
    label: str
    fact: str
    session: str
    happened_at: AwareDatetime | None = None


class RQuery(BaseModel):
    id: str
    query: str
    category: str = "uncategorized"
    limit: int = 5
    # Deliberately plain datetime: naive bounds are part of what is tested.
    time_range: list[datetime | None] | None = None
    expected: list[str] = Field(default_factory=list)
    related: list[str] = Field(default_factory=list)
    expected_sessions: list[str] = Field(default_factory=list)
    expect_empty: bool = False

    @model_validator(mode="after")
    def _consistent(self) -> RQuery:
        if self.time_range is not None and len(self.time_range) != 2:
            raise ValueError(f"query {self.id!r}: time_range must have 2 entries")
        if self.expect_empty and (self.expected or self.related):
            raise ValueError(f"query {self.id!r}: expect_empty excludes expectations")
        return self


class RetrievalDataset(BaseModel):
    name: str
    version: str = "1"
    description: str = ""
    sessions: list[RSession]
    nodes: list[RNode]
    edges: list[REdge] = Field(default_factory=list)
    queries: list[RQuery]
    source_sha256: str | None = None

    @model_validator(mode="after")
    def _references(self) -> RetrievalDataset:
        sessions = {s.id for s in self.sessions}
        nodes = {n.id for n in self.nodes}
        edges = {e.id for e in self.edges}
        if len(sessions) != len(self.sessions) or len(nodes) != len(self.nodes):
            raise ValueError("duplicate session or node ids")
        if len(edges) != len(self.edges):
            raise ValueError("duplicate edge ids")
        for n in self.nodes:
            for obs in n.observations:
                if obs.session not in sessions:
                    raise ValueError(f"node {n.id!r}: unknown session {obs.session!r}")
        for e in self.edges:
            if e.from_ not in nodes or e.to not in nodes:
                raise ValueError(f"edge {e.id!r}: unknown endpoint")
            if e.session not in sessions:
                raise ValueError(f"edge {e.id!r}: unknown session {e.session!r}")
        for q in self.queries:
            unknown = [i for i in q.expected + q.related if i not in nodes | edges]
            unknown += [s for s in q.expected_sessions if s not in sessions]
            if unknown:
                raise ValueError(f"query {q.id!r}: unknown references {unknown}")
        return self


def load_retrieval_dataset(name_or_path: str) -> RetrievalDataset:
    path = resolve_dataset_path(name_or_path)
    raw = path.read_bytes()
    data = json.loads(raw)
    fmt = data.get("format")
    if fmt != RETRIEVAL_FORMAT:
        raise ValueError(f"expected format {RETRIEVAL_FORMAT!r}, got {fmt!r}")
    dataset = RetrievalDataset.model_validate(
        {k: v for k, v in data.items() if k != "format"}
    )
    dataset.source_sha256 = hashlib.sha256(raw).hexdigest()
    return dataset


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------


def build_graph(graph: Graph, dataset: RetrievalDataset) -> dict[str, str]:
    """Write the fixture through the same services the agent uses.

    Returns a map from dataset-local ids (sessions, nodes, edges) to Weavy ids.
    """
    trace = RunTrace(
        mode="ingestion",
        started_at=datetime.now(tz=timezone.utc),
        input_summary="retrieval-eval fixture",
    )
    ids: dict[str, str] = {}
    session_time: dict[str, datetime] = {}
    for s in sorted(dataset.sessions, key=lambda s: s.timestamp):
        sid = increment_counter(graph, "session")
        store_canonical.create_session(
            graph,
            Session(
                id=sid,
                timestamp=s.timestamp,
                messages=[ChatMessage(role="user", content=s.text)],
            ),
        )
        ids[s.id] = sid
        session_time[s.id] = s.timestamp

    for node in dataset.nodes:
        first, *rest = sorted(node.observations, key=lambda o: session_time[o.session])
        result = memory.create_node(
            graph,
            aliases=node.aliases,
            summary=node.summary,
            note=first.note,
            provenance=ProvenanceInput(source_id=ids[first.session]),
            trace=trace,
            event_time=session_time[first.session],
            happened_at=first.happened_at,
            session_id=ids[first.session],
            force=True,  # the fixture defines identity; skip the dedup guard
        )
        if not result.ok or result.id is None:
            raise RuntimeError(
                f"fixture node {node.id!r} not created: {result.message}"
            )
        ids[node.id] = result.id
        for obs in rest:
            memory.update_node(
                graph,
                node_id=result.id,
                note=obs.note,
                provenance=ProvenanceInput(source_id=ids[obs.session]),
                trace=trace,
                event_time=session_time[obs.session],
                happened_at=obs.happened_at,
                session_id=ids[obs.session],
            )

    for edge in dataset.edges:
        result = memory.create_edge(
            graph,
            from_node_id=ids[edge.from_],
            to_node_id=ids[edge.to],
            label=edge.label,
            fact=edge.fact,
            note="fixture",
            provenance=ProvenanceInput(source_id=ids[edge.session]),
            trace=trace,
            event_time=session_time[edge.session],
            happened_at=edge.happened_at,
            source_id=ids[edge.session],
        )
        if not result.ok or result.id is None:
            raise RuntimeError(
                f"fixture edge {edge.id!r} not created: {result.message}"
            )
        ids[edge.id] = result.id
    return ids


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


class RetrievalQueryResult(BaseModel):
    query_id: str
    category: str
    query: str
    time_range: list[datetime | None] | None
    limit: int
    status: str  # "ok" | "error"
    error: str | None = None
    returned: list[str] = Field(default_factory=list)
    expanded: list[str] = Field(default_factory=list)
    exposed_sessions: list[str] = Field(default_factory=list)
    recall_at_k: float | None = None
    precision_at_k: float | None = None
    mrr: float | None = None
    related_recall: float | None = None
    source_recall: float | None = None
    empty_correct: bool | None = None
    latency_ms: float


class RetrievalReport(BaseModel):
    schema_version: int = RETRIEVAL_SCHEMA_VERSION
    started_at: datetime
    ended_at: datetime
    config: dict[str, Any]
    dataset: dict[str, Any]
    queries: list[RetrievalQueryResult]
    summary: dict[str, Any] = Field(default_factory=dict)
    notes: list[str] = Field(default_factory=list)


def _ratio(num: int, den: int) -> float | None:
    return num / den if den else None


def score_query(
    q: RQuery, results: list[Any], ids: dict[str, str], latency_ms: float
) -> RetrievalQueryResult:
    direct = [r.id for r in results if getattr(r, "via", None) is None]
    expanded = [r.id for r in results if getattr(r, "via", None) is not None]
    exposed: set[str] = set()
    for r in results:
        exposed.update(getattr(r, "source_sessions", None) or [])
    expected = {ids[i] for i in q.expected}
    related = {ids[i] for i in q.related}
    exp_sessions = {ids[s] for s in q.expected_sessions}
    hits = expected & set(direct)
    first_rank = next((i for i, rid in enumerate(direct, 1) if rid in expected), None)
    return RetrievalQueryResult(
        query_id=q.id,
        category=q.category,
        query=q.query,
        time_range=q.time_range,
        limit=q.limit,
        status="ok",
        returned=direct,
        expanded=expanded,
        exposed_sessions=sorted(exposed),
        recall_at_k=_ratio(len(hits), len(expected)),
        precision_at_k=_ratio(len(hits), len(direct)) if expected else None,
        mrr=(1 / first_rank if first_rank else 0.0) if expected else None,
        related_recall=_ratio(len(related & set(direct + expanded)), len(related)),
        source_recall=_ratio(len(exp_sessions & exposed), len(exp_sessions)),
        empty_correct=(len(results) == 0) if q.expect_empty else None,
        latency_ms=latency_ms,
    )


def _mean(values) -> dict[str, Any]:
    present = [float(v) for v in values if v is not None]
    return {"mean": statistics.fmean(present) if present else None, "n": len(present)}


def _block(rows: list[RetrievalQueryResult]) -> dict[str, Any]:
    return {
        "queries": len(rows),
        "errors": sum(r.status != "ok" for r in rows),
        "recall_at_k": _mean(r.recall_at_k for r in rows),
        "precision_at_k": _mean(r.precision_at_k for r in rows),
        "mrr": _mean(r.mrr for r in rows),
        "related_recall": _mean(r.related_recall for r in rows),
        "source_recall": _mean(r.source_recall for r in rows),
        "empty_correct": _mean(r.empty_correct for r in rows),
        "latency_ms": _mean(r.latency_ms for r in rows),
    }


def summarize_retrieval(rows: list[RetrievalQueryResult]) -> dict[str, Any]:
    by_cat: dict[str, list[RetrievalQueryResult]] = defaultdict(list)
    for r in rows:
        by_cat[r.category].append(r)
    return {
        "overall": _block(rows),
        "by_category": {c: _block(rs) for c, rs in sorted(by_cat.items())},
    }


def run_retrieval_eval(
    dataset: RetrievalDataset,
    *,
    graph_factory: Callable[[str], Graph],
    graph_prefix: str = "weavy_eval",
    cleanup: bool = False,
    search_overrides: dict[str, Any] | None = None,
    progress: Callable[[str], None] = lambda _m: None,
) -> RetrievalReport:
    """Build the fixture graph, run every query, and score it.

    Errors raised by ``search_graph`` are recorded on the query row (status
    ``error``, all metrics zero/false) rather than aborting the run.
    """
    from weavy.eval.runner import check_graph_names, eval_graph_name

    graph_name = eval_graph_name(graph_prefix, dataset.name, "retrieval")
    check_graph_names([graph_name], settings.GRAPH_NAME)
    graph = graph_factory(graph_name)
    delete_graph_if_exists(graph)
    init_system(graph, embedding_dim=embedding.get_dimension())

    started = datetime.now(tz=timezone.utc)
    rows: list[RetrievalQueryResult] = []
    try:
        progress(f"building fixture graph {graph_name}")
        ids = build_graph(graph, dataset)
        for q in dataset.queries:
            progress(f"  search {q.id}")
            t0 = time.perf_counter()
            try:
                out = memory.search_graph(
                    graph,
                    query=q.query,
                    limit=q.limit,
                    time_range=q.time_range,
                    **(search_overrides or {}),
                )
            except Exception as exc:
                rows.append(
                    RetrievalQueryResult(
                        query_id=q.id,
                        category=q.category,
                        query=q.query,
                        time_range=q.time_range,
                        limit=q.limit,
                        status="error",
                        error=f"{type(exc).__name__}: {exc}",
                        recall_at_k=0.0 if q.expected else None,
                        precision_at_k=None,
                        mrr=0.0 if q.expected else None,
                        related_recall=0.0 if q.related else None,
                        source_recall=0.0 if q.expected_sessions else None,
                        empty_correct=False if q.expect_empty else None,
                        latency_ms=(time.perf_counter() - t0) * 1000,
                    )
                )
                continue
            latency = (time.perf_counter() - t0) * 1000
            rows.append(score_query(q, out.results, ids, latency))
    finally:
        if cleanup:
            delete_graph_if_exists(graph)

    notes = [
        "Retrieval-only evaluation: the graph is built directly from the "
        "fixture and search_graph is called without an LLM, so this measures "
        "the retrieval primitive, not answer quality.",
    ]
    if settings.EMBEDDING_MODEL.startswith(embedding.LOCAL_HASH_PREFIX):
        notes.append(
            "Embeddings came from the local bag-of-words hash embedder: vector "
            "similarity reflects shared word tokens only, not meaning. Results "
            "measure retrieval mechanics (filtering, thresholds, provenance, "
            "expansion), not semantic retrieval quality."
        )
    return RetrievalReport(
        started_at=started,
        ended_at=datetime.now(tz=timezone.utc),
        config={
            "embedding_model": settings.EMBEDDING_MODEL,
            "graph": graph_name,
            "search_overrides": search_overrides or {},
            "search_settings": {
                k: getattr(settings, k)
                for k in dir(settings)
                if k.startswith("SEARCH_") or k == "NEIGHBORHOOD_LIMIT"
            },
        },
        dataset={
            "name": dataset.name,
            "version": dataset.version,
            "description": dataset.description,
            "sha256": dataset.source_sha256,
            "queries": len(dataset.queries),
        },
        queries=rows,
        summary=summarize_retrieval(rows),
        notes=notes,
    )


def _pct(block: dict[str, Any]) -> str:
    m = block["mean"]
    return ("  n/a" if m is None else f"{m * 100:5.1f}%") + f" (n={block['n']})"


def render_retrieval_text(report: RetrievalReport) -> str:
    o = report.summary["overall"]
    lines = [
        f"Weavy retrieval eval — dataset {report.dataset['name']} "
        f"v{report.dataset['version']} ({report.dataset['queries']} queries, "
        f"sha256 {str(report.dataset['sha256'])[:12]})",
        f"embeddings {report.config['embedding_model']} | "
        f"settings {report.config['search_settings']}",
        "",
        f"Recall@k        {_pct(o['recall_at_k'])}",
        f"Precision@k     {_pct(o['precision_at_k'])}",
        f"MRR             {_pct(o['mrr'])}",
        f"Related recall  {_pct(o['related_recall'])}",
        f"Source recall   {_pct(o['source_recall'])}",
        f"Empty correct   {_pct(o['empty_correct'])}",
        f"Errors          {o['errors']}/{o['queries']}",
        f"Latency         {o['latency_ms']['mean'] or 0:.1f} ms mean",
        "",
        "By category:",
    ]
    for cat, b in report.summary["by_category"].items():
        lines.append(
            f"  {cat:<16} recall {_pct(b['recall_at_k'])}  prec {_pct(b['precision_at_k'])}"
            f"  empty {_pct(b['empty_correct'])}  errors {b['errors']}"
        )
    errors = [r for r in report.queries if r.status != "ok"]
    if errors:
        lines += ["", "Errors:"] + [f"  {r.query_id}: {r.error}" for r in errors]
    lines += ["", "Notes:"] + [f"  - {n}" for n in report.notes]
    return "\n".join(lines)


def write_retrieval_json(report: RetrievalReport, path: str) -> str:
    from pathlib import Path

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report.model_dump(mode="json"), indent=2) + "\n")
    return str(target)
