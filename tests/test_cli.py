from datetime import datetime, timezone
from unittest.mock import MagicMock

from weavy import cli
from weavy.models.traces import RunTrace
from weavy.services.backup import BackupSummary


def _trace(status: str, *, summary: str = "done") -> RunTrace:
    return RunTrace(
        mode="ingestion",
        started_at=datetime.now(tz=timezone.utc),
        input_summary="test",
        status=status,
        completion_payload={"summary": summary},
    )


def test_cli_add_runs_theme_update_after_success(monkeypatch, tmp_path) -> None:
    source = tmp_path / "note.txt"
    source.write_text("hello")
    graph = MagicMock()

    add_trace = _trace("completed", summary="ingested")
    theme_trace = RunTrace(
        mode="theme",
        started_at=datetime.now(tz=timezone.utc),
        input_summary="theme",
        status="completed",
    )

    run_add = MagicMock(return_value=add_trace)
    run_theme_update = MagicMock(return_value=theme_trace)
    monkeypatch.setattr(cli, "get_graph", MagicMock(return_value=graph))
    monkeypatch.setattr(cli.session_runs, "run_add", run_add)
    monkeypatch.setattr(cli.theme_runs, "run_theme_update", run_theme_update)

    args = MagicMock(source=str(source), timestamp=None, context=None)
    cli.cmd_add(args)

    run_add.assert_called_once_with(
        "hello",
        graph,
        timestamp=None,
        context=None,
    )
    run_theme_update.assert_called_once_with(graph)


def test_cli_add_skips_theme_update_after_failed_ingestion(
    monkeypatch, tmp_path
) -> None:
    source = tmp_path / "note.txt"
    source.write_text("hello")
    graph = MagicMock()

    add_trace = _trace("failed")
    add_trace.error = "model failed"

    run_theme_update = MagicMock()
    monkeypatch.setattr(cli, "get_graph", MagicMock(return_value=graph))
    monkeypatch.setattr(cli.session_runs, "run_add", MagicMock(return_value=add_trace))
    monkeypatch.setattr(cli.theme_runs, "run_theme_update", run_theme_update)

    args = MagicMock(source=str(source), timestamp=None, context=None)
    cli.cmd_add(args)

    run_theme_update.assert_not_called()


def test_cli_export_writes_backup(monkeypatch, tmp_path) -> None:
    backup_path = tmp_path / "backup.json"
    graph = MagicMock()
    summary = BackupSummary(
        path=str(backup_path),
        graph_name="weavy",
        sessions=1,
        semantic_nodes=2,
        semantic_edges=1,
        themes=1,
        run_traces=1,
    )
    export_backup = MagicMock(return_value=summary)
    monkeypatch.setattr(cli.settings, "GRAPH_NAME", "weavy")
    monkeypatch.setattr(cli, "get_graph", MagicMock(return_value=graph))
    monkeypatch.setattr(cli, "export_graph_backup", export_backup)

    cli.cmd_export(MagicMock(path=str(backup_path)))

    export_backup.assert_called_once_with(graph, str(backup_path), graph_name="weavy")


def test_cli_import_requires_replace_flag(monkeypatch, tmp_path) -> None:
    backup_path = tmp_path / "backup.json"
    graph = MagicMock()
    import_backup = MagicMock(side_effect=RuntimeError("Target graph is not empty."))
    monkeypatch.setattr(cli.settings, "GRAPH_NAME", "weavy")
    monkeypatch.setattr(cli, "get_graph", MagicMock(return_value=graph))
    monkeypatch.setattr(cli, "import_graph_backup", import_backup)

    try:
        cli.cmd_import(MagicMock(path=str(backup_path), replace=False))
    except SystemExit as exc:
        assert exc.code == 1
    else:
        raise AssertionError(
            "cmd_import should exit on non-empty graph without replace"
        )

    import_backup.assert_called_once_with(
        graph,
        str(backup_path),
        replace=False,
        graph_name="weavy",
    )
