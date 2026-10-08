"""Retrieval-level eval: dataset validation, scoring, and an end-to-end run
against FalkorDB (deterministic conftest embeddings, no LLM)."""

import json
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from weavy import cli
from weavy.application.contracts import SearchResult
from weavy.eval.retrieval import (
    RetrievalDataset,
    RQuery,
    load_retrieval_dataset,
    run_retrieval_eval,
    score_query,
)
from weavy.services.embedding import hash_embed
from weavy.store.client import _get_db, get_graph


def test_builtin_retrieval_dataset_loads() -> None:
    ds = load_retrieval_dataset("retrieval-smoke")
    assert ds.name == "retrieval-smoke"
    assert len(ds.queries) == 13
    assert "NOT a benchmark" in ds.description
    assert ds.source_sha256


def test_dataset_rejects_unknown_references() -> None:
    base = {
        "name": "x",
        "sessions": [
            {"id": "s", "timestamp": "2024-01-01T00:00:00+00:00", "text": "t"}
        ],
        "nodes": [
            {
                "id": "n",
                "aliases": ["a"],
                "summary": "s",
                "observations": [{"session": "s"}],
            }
        ],
        "queries": [{"id": "q", "query": "a", "expected": ["missing"]}],
    }
    with pytest.raises(ValidationError, match="unknown references"):
        RetrievalDataset.model_validate(base)


def test_qa_format_rejected_by_retrieval_loader() -> None:
    with pytest.raises(ValueError, match="expected format"):
        load_retrieval_dataset("smoke")


def _row(rid, via=None, sources=()):
    kind = "edge" if rid.startswith("edge") else "node"
    return SearchResult(
        kind=kind,
        id=rid,
        label=rid,
        text="",
        distance=None,
        via=via,
        source_sessions=list(sources),
    )


def test_score_query_metrics() -> None:
    ids = {
        "n-a": "node:1",
        "n-b": "node:2",
        "n-c": "node:3",
        "s-1": "s:1",
        "s-2": "s:2",
    }
    q = RQuery(
        id="q",
        query="x",
        expected=["n-a", "n-b"],
        related=["n-c"],
        expected_sessions=["s-1", "s-2"],
    )
    results = [
        _row("node:9", sources=["s:1"]),
        _row("node:1"),
        _row("node:3", via="node:1"),
    ]
    r = score_query(q, results, ids, latency_ms=1.0)
    assert r.returned == ["node:9", "node:1"] and r.expanded == ["node:3"]
    assert r.recall_at_k == 0.5
    assert r.precision_at_k == 0.5
    assert r.mrr == 0.5
    assert r.related_recall == 1.0
    assert r.source_recall == 0.5
    assert r.empty_correct is None

    empty_q = RQuery(id="e", query="x", expect_empty=True)
    assert score_query(empty_q, [], ids, 1.0).empty_correct is True
    assert score_query(empty_q, results, ids, 1.0).empty_correct is False


@pytest.fixture
def hash_embeddings(monkeypatch):
    """Swap conftest's random 8-dim fakes for the offline hash embedder.

    Random low-dimensional vectors are often within any sensible distance
    threshold of each other, which would make threshold behaviour untestable.
    """
    from weavy.services import embedding

    dim = 512
    monkeypatch.setattr(embedding, "embed", lambda t: hash_embed(t, dim))
    monkeypatch.setattr(
        embedding,
        "embed_node",
        lambda aliases, summary: hash_embed(" | ".join(aliases) + " — " + summary, dim),
    )
    monkeypatch.setattr(
        embedding,
        "embed_edge",
        lambda label, fact: hash_embed(f"{label} — {fact}", dim),
    )
    monkeypatch.setattr(embedding, "get_dimension", lambda: dim)


def test_retrieval_eval_end_to_end_on_falkordb(hash_embeddings) -> None:
    ds = load_retrieval_dataset("retrieval-smoke")
    report = run_retrieval_eval(
        ds, graph_factory=get_graph, graph_prefix="weavy_test_eval", cleanup=True
    )
    o = report.summary["overall"]
    assert o["errors"] == 0, [q.error for q in report.queries if q.error]
    by_cat = report.summary["by_category"]
    # Mechanics the fixture is designed to exercise, under the default
    # SEARCH_MAX_DISTANCE with lexical hash embeddings.
    assert by_cat["temporal"]["recall_at_k"]["mean"] == 1.0
    assert by_cat["temporal-open"]["recall_at_k"]["mean"] == 1.0
    assert by_cat["temporal-naive"]["recall_at_k"]["mean"] == 1.0
    assert by_cat["entity-in-question"]["recall_at_k"]["mean"] == 1.0
    assert by_cat["no-match"]["empty_correct"]["mean"] == 1.0
    assert o["source_recall"]["mean"] == 1.0
    assert "weavy_test_eval_retrieval-smoke_retrieval" not in _get_db().list_graphs()


def test_cli_retrieval_mode_writes_report(tmp_path, capsys, hash_embeddings) -> None:
    out = tmp_path / "r.json"
    args = SimpleNamespace(
        dataset=None,
        format="weavy-eval/v1",
        judge="keyword",
        judge_model=None,
        output=str(out),
        limit=None,
        conversation=[],
        graph_prefix="weavy_test_eval",
        no_themes=False,
        cleanup=True,
        verbose=False,
        list_datasets=False,
        mode="retrieval",
        embedding_model=None,
        max_distance=None,
    )
    cli.cmd_eval(args)
    data = json.loads(out.read_text())
    assert data["dataset"]["name"] == "retrieval-smoke"
    assert "Recall@k" in capsys.readouterr().out
