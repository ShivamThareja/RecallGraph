"""`weavy eval` CLI wiring and a FalkorDB-backed end-to-end eval run.

The CLI tests mock the runner. The integration test drives the real Weavy SDK
against FalkorDB with a scripted (mocked) LLM, so it verifies the session-id
mapping and provenance-based recall on real graph data.
"""

import json
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from weavy import cli
from weavy.client import Weavy
from weavy.eval.dataset import EvalDataset
from weavy.eval.judges import KeywordJudge
from weavy.eval.runner import run_eval
from weavy.store.client import delete_graph_if_exists, get_graph
from tests.conftest import TEST_EMBEDDING_DIM
from tests.helpers import mock_tool_response


def _args(**kw) -> SimpleNamespace:
    base = dict(
        dataset=None,
        format="weavy-eval/v1",
        judge="keyword",
        judge_model=None,
        output=None,
        limit=None,
        conversation=[],
        graph_prefix="weavy_eval",
        no_themes=False,
        cleanup=False,
        verbose=False,
        list_datasets=False,
        mode="qa",
        embedding_model=None,
        max_distance=None,
    )
    base.update(kw)
    return SimpleNamespace(**base)


def test_cli_list_datasets(capsys) -> None:
    cli.cmd_eval(_args(list_datasets=True))
    assert "smoke" in capsys.readouterr().out.split()


def test_cli_eval_bad_dataset_exits(capsys) -> None:
    with pytest.raises(SystemExit) as exc:
        cli.cmd_eval(_args(dataset="no-such-dataset"))
    assert exc.value.code == 1
    assert "Error" in capsys.readouterr().err


def test_cli_eval_writes_json_and_summary(tmp_path, capsys) -> None:
    from weavy.eval import runner as eval_runner

    captured = {}

    def fake_run_eval(dataset, **kwargs):
        captured.update(kwargs)
        # Re-dispatch to the real runner with a fake client: no DB / LLM.
        from tests.test_eval_runner import FakeClient

        kwargs["client_factory"] = FakeClient
        kwargs["cleanup"] = None
        return eval_runner.run_eval(dataset, **kwargs)

    out = tmp_path / "res.json"
    with patch("weavy.eval.run_eval", side_effect=fake_run_eval):
        cli.cmd_eval(
            _args(
                output=str(out),
                no_themes=True,
                limit=2,
                conversation=["journal"],
                cleanup=True,
            )
        )

    assert captured["update_themes"] is False
    assert captured["limit"] == 2
    assert captured["conversation_ids"] == ["journal"]
    assert captured["client_factory"] is Weavy
    assert callable(captured["cleanup"])
    data = json.loads(out.read_text())
    assert data["dataset"]["name"] == "smoke"
    assert len(data["questions"]) == 2
    stdout = capsys.readouterr().out
    assert "Accuracy" in stdout and f"Results written to {out}" in stdout


def test_cli_eval_parser_registered() -> None:
    # main() builds its dispatch dict at call time, so patching the module
    # global is enough.
    with (
        patch("sys.argv", ["weavy", "eval", "--list-datasets"]),
        patch.object(cli, "cmd_eval") as cmd,
    ):
        cli.main()
    assert cmd.call_args.args[0].list_datasets is True


# ---------------------------------------------------------------------------
# Integration: real Weavy SDK + FalkorDB, scripted LLM
# ---------------------------------------------------------------------------

INTEGRATION_DATASET = EvalDataset.model_validate(
    {
        "name": "it",
        "conversations": [
            {
                "id": "c1",
                "query_time": "2024-03-25T12:00:00+00:00",
                "sessions": [
                    {
                        "id": "job",
                        "timestamp": "2024-03-01T09:00:00+00:00",
                        "text": "I started at Brightline Robotics today.",
                    }
                ],
                "questions": [
                    {
                        "id": "q1",
                        "question": "Where do I work?",
                        "answer": "Brightline Robotics",
                        "answer_keywords": ["Brightline"],
                        "evidence_sessions": ["job"],
                        "evidence_keywords": ["Brightline"],
                    }
                ],
            }
        ],
    }
)


def test_eval_against_real_graph_maps_sessions_and_measures_recall() -> None:
    prefix = "weavy_test_eval"
    usage = {"prompt_tokens": 50, "completion_tokens": 5, "total_tokens": 55}
    script = [
        # Ingestion of the only session (fresh graph → s:1).
        mock_tool_response(
            "create_node",
            {
                "aliases": ["Brightline Robotics"],
                "summary": "Company where the user started working in March 2024.",
                "note": "new employer",
                "provenance": {"source_id": "s:1"},
            },
            "tc-1",
            usage=usage,
        ),
        mock_tool_response(
            "complete", {"summary": "added employer"}, "tc-2", usage=usage
        ),
        # Query (its own session is s:2): navigate node → episode, then answer.
        mock_tool_response("get_node", {"node_ids": ["node:1"]}, "tc-3", usage=usage),
        mock_tool_response("get_session", {"session_id": "s:1"}, "tc-4", usage=usage),
        mock_tool_response(
            "complete",
            {
                "summary": "no changes",
                "answer": "You work at Brightline Robotics.",
                "cited_sources": ["s:1"],
                "consulted_nodes": ["node:1"],
            },
            "tc-5",
            usage=usage,
        ),
    ]
    cleaned: list[str] = []

    def cleanup(name: str) -> None:
        cleaned.append(name)
        delete_graph_if_exists(get_graph(name))

    # client.py binds get_dimension at import, so conftest's embedding mock does
    # not reach Weavy(); align the vector index with the 8-dim fake embeddings.
    with (
        patch("litellm.completion", side_effect=script),
        patch("weavy.client.get_dimension", return_value=TEST_EMBEDDING_DIM),
    ):
        report = run_eval(
            INTEGRATION_DATASET,
            client_factory=Weavy,
            judge=KeywordJudge(),
            graph_prefix=prefix,
            update_themes=False,
            cleanup=cleanup,
        )

    assert cleaned == ["weavy_test_eval_it_c1"]
    assert [r.session_id for r in report.ingestion] == ["s:1"]
    assert report.ingestion[0].nodes_created == 1
    row = report.questions[0]
    assert row.status == "completed", row.error
    assert row.correct
    assert row.gold_sessions == ["s:1"]
    assert row.surfaced_sessions == ["s:1"]
    assert row.read_sessions == ["s:1"]
    assert row.session_recall == 1.0
    assert row.fact_recall == 1.0
    assert row.citation_support == 1.0
    assert row.tool_calls == 3
    assert row.usage.prompt_tokens == 150
