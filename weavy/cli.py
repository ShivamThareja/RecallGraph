"""
Weavy CLI — entry point for operational commands.
Run as: python -m weavy.cli <command>
"""

import argparse
import sys
from datetime import datetime

from falkordb import Graph

from weavy.application import session_runs, theme_runs
from weavy.config import settings
from weavy.models.traces import RunTrace
from weavy.services.backup import BackupSummary
from weavy.services.backup import export_backup as export_graph_backup
from weavy.services.backup import import_backup as import_graph_backup
from weavy.store import themes as store_themes
from weavy.store.canonical import list_sessions
from weavy.store.client import get_graph
from weavy.store.system import (
    SystemState,
    get_system,
    init_system,
    set_preface,
)


def _print_system_state(header: str, state: SystemState, graph: Graph) -> None:
    theme_names = [t.name for t in store_themes.list_all_themes(graph)]
    print(f"{header}:")
    print(f"  preface           = {state.preface or '(not set)'}")
    print(f"  next_node_id      = {state.next_node_id}")
    print(f"  next_edge_id      = {state.next_edge_id}")
    print(f"  next_session_id   = {state.next_session_id}")
    print(f"  themes            = {theme_names}")
    print(f"  last_theme_run_at = {state.last_theme_run_at}")


def _read_text(source: str) -> str:
    if source == "-":
        return sys.stdin.read()
    with open(source) as f:
        return f.read()


def _parse_timestamp(value: str | None) -> datetime | None:
    if value is None:
        return None
    return datetime.fromisoformat(value)


def _print_trace_status(trace: RunTrace) -> bool:
    print(f"Status: {trace.status}")
    if trace.error:
        print(f"Error: {trace.error}", file=sys.stderr)
        return False
    return True


def _completion_field(trace: RunTrace, field: str) -> str:
    value = (trace.completion_payload or {}).get(field)
    return value if isinstance(value, str) else ""


def _print_backup_summary(action: str, summary: BackupSummary) -> None:
    print(f"{action}: {summary.path}")
    print(f"  graph          = {summary.graph_name or '(default)'}")
    print(f"  sessions       = {summary.sessions}")
    print(f"  semantic_nodes = {summary.semantic_nodes}")
    print(f"  semantic_edges = {summary.semantic_edges}")
    print(f"  themes         = {summary.themes}")
    print(f"  run_traces     = {summary.run_traces}")


def cmd_init_system(_args: argparse.Namespace) -> None:
    from weavy.services.embedding import get_dimension

    graph = get_graph()
    state = init_system(graph, embedding_dim=get_dimension())
    _print_system_state("System node initialised", state, graph)


def cmd_status(_args: argparse.Namespace) -> None:
    graph = get_graph()
    state = get_system(graph)
    _print_system_state("System state", state, graph)


def cmd_list_sessions(args: argparse.Namespace) -> None:
    graph = get_graph()
    output = list_sessions(graph, limit=args.limit)
    if not output.sessions:
        print("No sessions found.")
        return
    for s in output.sessions:
        summary_line = f"  {s.summary}" if s.summary else "  (not yet ingested)"
        print(f"  {s.id}  {s.timestamp.isoformat()}{summary_line}")


def cmd_add(args: argparse.Namespace) -> None:
    text = _read_text(args.source)

    if not text.strip():
        print("Error: empty input.", file=sys.stderr)
        sys.exit(1)

    trace = session_runs.run_add(
        text,
        get_graph(),
        timestamp=_parse_timestamp(args.timestamp),
        context=args.context,
    )
    if not _print_trace_status(trace):
        return

    touched = list(dict.fromkeys(n.node_id for n in trace.touched_nodes))
    print(f"Touched nodes: {touched}")
    print(f"Summary: {_completion_field(trace, 'summary')}")

    theme_trace = theme_runs.run_theme_update(get_graph())
    if theme_trace.status == "failed":
        print("Theme update status: failed", file=sys.stderr)
        if theme_trace.error:
            print(f"Theme update error: {theme_trace.error}", file=sys.stderr)
        return
    print("Theme update complete.")


def cmd_update_themes(_args: argparse.Namespace) -> None:
    trace = theme_runs.run_theme_update(get_graph())
    if _print_trace_status(trace):
        print("Theme update complete.")


def cmd_export(args: argparse.Namespace) -> None:
    summary = export_graph_backup(
        get_graph(), args.path, graph_name=settings.GRAPH_NAME
    )
    _print_backup_summary("Exported backup", summary)


def cmd_import(args: argparse.Namespace) -> None:
    try:
        summary = import_graph_backup(
            get_graph(),
            args.path,
            replace=args.replace,
            graph_name=settings.GRAPH_NAME,
        )
    except (RuntimeError, ValueError, OSError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)
    _print_backup_summary("Imported backup", summary)


def cmd_set_preface(args: argparse.Namespace) -> None:
    graph = get_graph()
    set_preface(graph, args.preface)
    print(f"Preface set: {args.preface}")


def _print_citation_warnings(trace: RunTrace) -> None:
    check = (trace.completion_payload or {}).get("citation_check") or {}
    unsupported = check.get("unsupported_sources", []) + check.get(
        "unsupported_nodes", []
    )
    if unsupported:
        print(
            f"\nUnverified citations (not retrieved during this run): "
            f"{', '.join(unsupported)}",
            file=sys.stderr,
        )


def cmd_continue(args: argparse.Namespace) -> None:
    trace = session_runs.run_session(
        args.session_id, "query", get_graph(), args.question
    )
    if _print_trace_status(trace):
        print(f"\n{_completion_field(trace, 'answer')}")
        _print_citation_warnings(trace)


def cmd_query(args: argparse.Namespace) -> None:
    if args.question is None:
        _run_chat_repl()
        return

    trace = session_runs.run_query(args.question, get_graph())
    if _print_trace_status(trace):
        print(f"\n{_completion_field(trace, 'answer')}")
        _print_citation_warnings(trace)


def _apply_eval_overrides(args: argparse.Namespace) -> None:
    """Process-local setting overrides for an eval run (never persisted)."""
    if args.embedding_model:
        from weavy.services import embedding

        settings.EMBEDDING_MODEL = args.embedding_model
        embedding.get_dimension.cache_clear()
    if args.max_distance is not None:
        value = args.max_distance.strip().lower()
        settings.SEARCH_MAX_DISTANCE = None if value == "none" else float(value)


def _cmd_eval_retrieval(args: argparse.Namespace) -> None:
    from weavy.eval.retrieval import (
        load_retrieval_dataset,
        render_retrieval_text,
        run_retrieval_eval,
        write_retrieval_json,
    )

    try:
        dataset = load_retrieval_dataset(args.dataset or "retrieval-smoke")
        report = run_retrieval_eval(
            dataset,
            graph_factory=get_graph,
            graph_prefix=args.graph_prefix,
            cleanup=args.cleanup,
            progress=lambda m: print(m, file=sys.stderr, flush=True),
        )
    except (OSError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)
    output = args.output or (
        f"weavy-eval-{dataset.name}-{report.started_at:%Y%m%dT%H%M%SZ}.json"
    )
    path = write_retrieval_json(report, output)
    print(render_retrieval_text(report))
    print(f"\nResults written to {path}")


def cmd_eval(args: argparse.Namespace) -> None:
    import contextlib
    import io

    from weavy.client import Weavy
    from weavy.eval import load_dataset, make_judge, render_text, run_eval, write_json
    from weavy.eval.dataset import list_builtin_datasets
    from weavy.store.client import delete_graph_if_exists

    if args.list_datasets:
        for name in list_builtin_datasets():
            print(name)
        return

    _apply_eval_overrides(args)
    if args.mode == "retrieval":
        _cmd_eval_retrieval(args)
        return

    try:
        dataset = load_dataset(args.dataset or "smoke", args.format)
        judge_model = (
            (args.judge_model or settings.LLM_MODEL) if args.judge == "llm" else None
        )
        judge = make_judge(args.judge, judge_model)
    except (OSError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)

    def progress(message: str) -> None:
        print(message, file=sys.stderr, flush=True)

    def cleanup(graph_name: str) -> None:
        delete_graph_if_exists(get_graph(graph_name))

    # Agent runs print per-turn console output; keep it out of the summary
    # unless --verbose asks for it.
    sink = (
        contextlib.nullcontext()
        if args.verbose
        else contextlib.redirect_stdout(io.StringIO())
    )
    try:
        with sink:
            report = run_eval(
                dataset,
                client_factory=Weavy,
                judge=judge,
                graph_prefix=args.graph_prefix,
                update_themes=not args.no_themes,
                conversation_ids=args.conversation or None,
                limit=args.limit,
                cleanup=cleanup if args.cleanup else None,
                judge_model=judge_model,
                progress=progress,
            )
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)

    if judge_model and judge_model == settings.LLM_MODEL:
        report.notes.append(
            "The LLM judge uses the same model as the system under test, which "
            "can bias grading; pass --judge-model to use an independent judge."
        )
    output = args.output or (
        f"weavy-eval-{dataset.name}-{report.started_at:%Y%m%dT%H%M%SZ}.json"
    )
    path = write_json(report, output)
    print(render_text(report))
    print(f"\nResults written to {path}")


def _run_chat_repl() -> None:
    """Interactive REPL: each turn runs the query agent with full session history."""
    from weavy.harness.tracing import ChatSessionTracer

    graph = get_graph()
    session_id = session_runs.create_session("", graph)
    session_tracer = ChatSessionTracer(session_id)
    message_count = 0

    print("Weavy chat — type 'exit' or Ctrl-D to quit.\n")
    while True:
        try:
            question = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not question:
            continue
        if question.lower() in ("exit", "quit"):
            break

        message_count += 1
        trace = session_runs.run_session(
            session_id,
            "query",
            graph,
            question,
            parent_observation=session_tracer.root,
        )

        if trace.status == "failed":
            print(f"[error] {trace.error}\n")
        else:
            answer = (trace.completion_payload or {}).get("answer", "")
            print(f"\nWeavy: {answer}\n")

    session_tracer.finalize(message_count)


def main() -> None:
    parser = argparse.ArgumentParser(prog="weavy", description="Weavy CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("init-system", help="Initialise the System node in FalkorDB")
    subparsers.add_parser("status", help="Print current System node state")

    p = subparsers.add_parser("list-sessions", help="List stored sessions")
    p.add_argument("--limit", type=int, default=20)

    p = subparsers.add_parser("add", help="Add text to the memory graph")
    p.add_argument(
        "source",
        help="Path to a text file, or '-' to read from stdin",
    )
    p.add_argument(
        "--context", default=None, help="Caller context for the ingestion agent"
    )
    p.add_argument("--timestamp", default=None, help="ISO timestamp for the session")

    subparsers.add_parser(
        "update-themes", help="Manually run the theme agent over the current graph"
    )

    p = subparsers.add_parser("export", help="Export a complete JSON backup")
    p.add_argument("path", help="Output JSON backup path")

    p = subparsers.add_parser("import", help="Import a complete JSON backup")
    p.add_argument("path", help="Input JSON backup path")
    p.add_argument(
        "--replace",
        action="store_true",
        help="Replace the target graph if it already contains data",
    )

    p = subparsers.add_parser(
        "set-preface", help="Set the graph preface describing what this graph is about"
    )
    p.add_argument(
        "preface",
        help="Description of this graph (e.g. 'Personal memory graph for Alex')",
    )

    p = subparsers.add_parser(
        "continue", help="Continue an existing session with a new question"
    )
    p.add_argument("session_id", help="Session to continue (e.g. s:1)")
    p.add_argument("question", help="Question to ask in the context of that session")

    p = subparsers.add_parser("query", help="Ask a question against the memory graph")
    p.add_argument(
        "question",
        nargs="?",
        default=None,
        help="Question to ask. Omit to enter interactive chat.",
    )

    p = subparsers.add_parser(
        "eval", help="Evaluate memory quality on a dataset (isolated graphs)"
    )
    p.add_argument(
        "--dataset",
        default=None,
        help="Built-in dataset name or path to a dataset JSON file "
        "(default: smoke for --mode qa, retrieval-smoke for --mode retrieval)",
    )
    p.add_argument(
        "--mode",
        choices=["qa", "retrieval"],
        default="qa",
        help="qa = full agent question answering (needs an LLM); "
        "retrieval = score search_graph on a fixture graph (no LLM)",
    )
    p.add_argument(
        "--embedding-model",
        default=None,
        help="Override EMBEDDING_MODEL for this run (e.g. weavy/hash-512 for "
        "the offline, non-semantic hash embedder)",
    )
    p.add_argument(
        "--max-distance",
        default=None,
        help="Override SEARCH_MAX_DISTANCE for this run (a float, or 'none')",
    )
    p.add_argument(
        "--format",
        default="weavy-eval/v1",
        help="Dataset format (default: weavy-eval/v1)",
    )
    p.add_argument(
        "--judge",
        choices=["keyword", "llm"],
        default="keyword",
        help="keyword = deterministic offline matching; llm = model-graded",
    )
    p.add_argument(
        "--judge-model", default=None, help="Model for --judge llm (default: LLM_MODEL)"
    )
    p.add_argument("--output", default=None, help="Path for the JSON results file")
    p.add_argument(
        "--limit", type=int, default=None, help="Max questions per conversation"
    )
    p.add_argument(
        "--conversation",
        action="append",
        default=[],
        help="Only evaluate this conversation id (repeatable)",
    )
    p.add_argument(
        "--graph-prefix",
        default="weavy_eval",
        help="Prefix for the isolated eval graphs (default: weavy_eval)",
    )
    p.add_argument(
        "--no-themes", action="store_true", help="Skip theme updates during ingestion"
    )
    p.add_argument(
        "--cleanup", action="store_true", help="Delete eval graphs after the run"
    )
    p.add_argument(
        "--verbose", action="store_true", help="Show per-turn agent console output"
    )
    p.add_argument(
        "--list-datasets", action="store_true", help="List built-in datasets and exit"
    )

    args = parser.parse_args()
    dispatch = {
        "init-system": cmd_init_system,
        "status": cmd_status,
        "list-sessions": cmd_list_sessions,
        "add": cmd_add,
        "continue": cmd_continue,
        "update-themes": cmd_update_themes,
        "export": cmd_export,
        "import": cmd_import,
        "set-preface": cmd_set_preface,
        "query": cmd_query,
        "eval": cmd_eval,
    }
    dispatch[args.command](args)


if __name__ == "__main__":
    main()
