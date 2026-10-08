"""Eval runner + report with a fake MemoryClient. No FalkorDB or LLM required."""

import json
from datetime import datetime, timezone

import pytest

from weavy.config import settings
from weavy.eval.dataset import EvalDataset
from weavy.eval.judges import KeywordJudge
from weavy.eval.report import render_text, write_json
from weavy.eval.runner import eval_graph_name, run_eval
from weavy.models.traces import RunTrace, ToolCall, TouchedNode, Turn, TurnUsage

NOW = datetime(2024, 1, 1, tzinfo=timezone.utc)

DATASET = EvalDataset.model_validate(
    {
        "name": "fake",
        "conversations": [
            {
                "id": "c1",
                "context": "notes",
                "query_time": "2024-03-01T00:00:00+05:30",
                "sessions": [
                    # Deliberately out of order: ingestion must sort by time.
                    {
                        "id": "b",
                        "timestamp": "2024-01-02T00:00:00+00:00",
                        "text": "Lena Berlin",
                    },
                    {
                        "id": "a",
                        "timestamp": "2024-01-01T00:00:00+00:00",
                        "text": "Biscuit",
                    },
                ],
                "questions": [
                    {
                        "id": "q-dog",
                        "category": "single-hop",
                        "question": "dog?",
                        "answer": "Biscuit",
                        "answer_keywords": ["Biscuit"],
                        "evidence_sessions": ["a"],
                        "evidence_keywords": ["Biscuit"],
                    },
                    {
                        "id": "q-sister",
                        "category": "multi-hop",
                        "question": "sister?",
                        "answer": "Lena, Berlin",
                        "answer_keywords": ["Lena", "Berlin"],
                        "evidence_sessions": ["a", "b"],
                    },
                    {
                        "id": "q-cat",
                        "category": "abstention",
                        "question": "cat?",
                        "answer": "none",
                        "abstain": True,
                    },
                    {
                        "id": "q-boom",
                        "category": "single-hop",
                        "question": "boom?",
                        "answer": "x",
                        "evidence_sessions": ["b"],
                        "evidence_keywords": ["Lena"],
                    },
                ],
            }
        ],
    }
)


def _trace(mode, *, session_id, calls=(), payload=None, status="completed", usage=None):
    return RunTrace(
        mode=mode,
        started_at=NOW,
        input_summary="t",
        session_id=session_id,
        status=status,
        turns=[Turn(turn_number=1, tool_calls=list(calls), timestamp=NOW)],
        total_usage=usage
        or TurnUsage(prompt_tokens=100, completion_tokens=10, total_tokens=110),
        completion_payload=payload,
        touched_nodes=[TouchedNode(node_id="node:1", action="created")]
        if mode == "ingestion"
        else [],
    )


def _call(name, args, result):
    return ToolCall(tool_name=name, args=args, result=json.dumps(result), called_at=NOW)


class FakeClient:
    """Mimics Weavy: mints s:N per add/query and answers from a script."""

    instances: list["FakeClient"] = []

    def __init__(self, graph_name: str) -> None:
        self.graph_name = graph_name
        self.next_session = 1
        self.added: list[tuple[str, datetime, str | None, bool]] = []
        self.queries: list[tuple[str, datetime | None]] = []
        self.theme_runs = 0
        self.was_reset = False
        FakeClient.instances.append(self)

    def _mint(self) -> str:
        sid = f"s:{self.next_session}"
        self.next_session += 1
        return sid

    def reset(self) -> None:
        self.was_reset = True

    def add(self, text, *, timestamp=None, context=None, update_themes=True):
        self.added.append((text, timestamp, context, update_themes))
        return _trace("ingestion", session_id=self._mint(), payload={"summary": "ok"})

    def update_themes(self):
        self.theme_runs += 1
        return _trace("theme", session_id=None)

    def query(self, question, *, context=None, query_time=None):
        self.queries.append((question, query_time))
        sid = self._mint()
        if question == "boom?":
            raise RuntimeError("db down")
        if question == "dog?":
            calls = [
                _call(
                    "get_node",
                    {"node_ids": ["node:1"]},
                    {"results": [{"mentioned_by": ["s:1"]}]},
                ),
                _call(
                    "get_session",
                    {"session_id": "s:1"},
                    {"id": "s:1", "text": "Biscuit"},
                ),
            ]
            return _trace(
                "query",
                session_id=sid,
                calls=calls,
                payload={"answer": "Your dog is Biscuit.", "cited_sources": ["s:1"]},
            )
        if question == "sister?":
            calls = [
                _call(
                    "get_node",
                    {"node_ids": ["node:2"]},
                    {"results": [{"mentioned_by": ["s:2"]}]},
                )
            ]
            return _trace(
                "query",
                session_id=sid,
                calls=calls,
                payload={"answer": "Lena.", "cited_sources": ["s:2", "s:7"]},
            )
        return _trace(
            "query",
            session_id=sid,
            payload={"answer": "The graph has no record of this."},
        )


@pytest.fixture(autouse=True)
def _reset_instances():
    FakeClient.instances = []


def _run(**kw):
    return run_eval(DATASET, client_factory=FakeClient, judge=KeywordJudge(), **kw)


def test_run_eval_end_to_end_metrics() -> None:
    report = _run(graph_prefix="t")
    client = FakeClient.instances[0]

    assert client.was_reset
    assert client.graph_name == eval_graph_name("t", "fake", "c1") == "t_fake_c1"
    # Sessions ingested in timestamp order, UTC-normalized, with the context.
    assert [a[0] for a in client.added] == ["Biscuit", "Lena Berlin"]
    assert all(a[1].tzinfo == timezone.utc and a[2] == "notes" for a in client.added)
    # Themes run separately (add is always called with update_themes=False).
    assert all(a[3] is False for a in client.added)
    assert client.theme_runs == 2
    # query_time normalized to UTC.
    assert client.queries[0][1] == datetime(2024, 2, 29, 18, 30, tzinfo=timezone.utc)

    rows = {r.question_id: r for r in report.questions}
    dog = rows["q-dog"]
    assert dog.correct and dog.status == "completed"
    assert dog.gold_sessions == ["s:1"]
    assert dog.session_recall == 1.0 and dog.session_precision == 1.0
    assert dog.fact_recall == 1.0
    assert dog.citation_precision == 1.0 and dog.citation_support == 1.0
    assert dog.tool_calls == 2 and dog.usage.prompt_tokens == 100

    sister = rows["q-sister"]
    assert not sister.correct  # missing "Berlin"
    assert sister.session_recall == 0.5
    assert sister.unsupported_citations == ["s:7"]

    assert rows["q-cat"].correct  # abstained

    boom = rows["q-boom"]
    assert boom.status == "error" and "db down" in boom.error and not boom.correct
    # A crashed query counts as zero recall instead of vanishing from the mean.
    assert boom.session_recall == 0.0 and boom.fact_recall == 0.0

    overall = report.summary["overall"]
    assert overall["questions"] == 4
    assert overall["correct"] == 2
    assert overall["accuracy"] == 0.5
    assert overall["failed_runs"] == 1
    # Recall is averaged over answerable questions with gold evidence.
    assert overall["session_recall"] == {"mean": pytest.approx(0.5), "n": 3}
    assert report.summary["by_category"]["abstention"]["accuracy"] == 1.0
    ing = report.summary["ingestion"]
    assert ing["sessions"] == 2 and ing["failed"] == 0 and ing["nodes_created"] == 2


def test_no_themes_skips_theme_pass() -> None:
    _run(update_themes=False)
    assert FakeClient.instances[0].theme_runs == 0


def test_limit_and_conversation_filter() -> None:
    report = _run(limit=1, conversation_ids=["c1"])
    assert [r.question_id for r in report.questions] == ["q-dog"]
    # Ingestion is never limited.
    assert len(report.ingestion) == 2
    with pytest.raises(ValueError, match="unknown conversation"):
        _run(conversation_ids=["nope"])
    with pytest.raises(ValueError, match="limit"):
        _run(limit=0)


def test_refuses_to_wipe_configured_graph(monkeypatch) -> None:
    monkeypatch.setattr(settings, "GRAPH_NAME", "t_fake_c1")
    with pytest.raises(ValueError, match="refusing"):
        _run(graph_prefix="t")
    assert FakeClient.instances == []  # nothing was opened or reset


def test_cleanup_called_per_graph_even_on_failure() -> None:
    cleaned: list[str] = []

    class Exploding(FakeClient):
        def add(self, *a, **kw):
            raise RuntimeError("ingest failed")

    report = run_eval(
        DATASET,
        client_factory=Exploding,
        judge=KeywordJudge(),
        cleanup=cleaned.append,
    )
    assert cleaned == ["weavy_eval_fake_c1"]
    assert all(r.status == "error" for r in report.ingestion)
    # Unmapped evidence is still counted in recall's denominator.
    dog = next(r for r in report.questions if r.question_id == "q-dog")
    assert dog.gold_sessions == ["unmapped:a"]


def test_unknown_model_cost_is_reported_unknown(monkeypatch) -> None:
    monkeypatch.setattr(settings, "LLM_MODEL", "nobody/unpriced-model")
    report = _run()
    usage = report.summary["usage"]["query"]
    assert usage["cost_known"] is False
    assert usage["estimated_cost_usd"] is None


def test_report_json_and_text(tmp_path) -> None:
    report = _run()
    path = write_json(report, tmp_path / "out" / "r.json")
    data = json.loads(path.read_text())
    assert data["schema_version"] == 1
    assert data["dataset"]["name"] == "fake"
    assert len(data["questions"]) == 4
    assert data["summary"]["overall"]["accuracy"] == 0.5
    assert any("Keyword judge" in n for n in data["notes"])

    text = render_text(report)
    assert "Accuracy" in text and "50.0%" in text
    assert "abstention" in text
    assert "Notes:" in text
