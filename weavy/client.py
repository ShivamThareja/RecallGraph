"""
Weavy SDK entry point.

    import weavy

    w = weavy.Weavy()                                      # connects + idempotent init
    trace = w.add("I started a new job today")             # -> RunTrace
    trace = w.query("What did I say about work?")          # -> RunTrace
    trace = w.continue_session("s:1", "Tell me more")      # -> RunTrace
    trace = w.update_themes()                              # -> RunTrace
"""

from __future__ import annotations

from datetime import datetime

from falkordb import Graph

from weavy.application.session_runs import run_add, run_query, run_session
from weavy.application.theme_runs import run_theme_update
from weavy.config import settings
from weavy.models.traces import RunTrace
from weavy.services.backup import BackupSummary
from weavy.services.backup import export_backup as export_graph_backup
from weavy.services.backup import import_backup as import_graph_backup
from weavy.services.embedding import get_dimension
from weavy.store.client import delete_graph_if_exists, get_graph
from weavy.store.system import init_system


class Weavy:
    """Weavy SDK client.

    Instantiation connects to FalkorDB and idempotently initialises the System
    node and vector index. All methods return a ``RunTrace`` — the full audit
    trail of the agent run.

    Args:
        graph_name: Named graph to use. Defaults to the ``GRAPH_NAME`` env var
            (``"weavy"``). Pass an explicit name for benchmark isolation.
    """

    def __init__(self, graph_name: str | None = None) -> None:
        self._graph_name = graph_name or settings.GRAPH_NAME
        self._graph: Graph = get_graph(graph_name)
        init_system(self._graph, embedding_dim=get_dimension())

    def add(
        self,
        text: str,
        *,
        timestamp: datetime | None = None,
        context: str | None = None,
        update_themes: bool = True,
    ) -> RunTrace:
        """Ingest *text* into the memory graph and refresh themes.

        Args:
            text: Raw text to ingest. Pre-processing (transcription, parsing)
                is the caller's responsibility.
            timestamp: Event time for the session. Defaults to wall-clock now.
            context: Optional hint for the ingestion agent, e.g. ``"These are
                Slack chat logs"``.
            update_themes: Refresh themes after a successful ingestion (default).
                Set ``False`` to skip the theme pass — useful for bulk backfills
                or benchmark ablations that isolate raw retrieval from themes.

        Returns:
            RunTrace with ``mode="ingestion"``.
        """
        trace = run_add(text, self._graph, timestamp=timestamp, context=context)
        if update_themes and trace.status == "completed":
            run_theme_update(self._graph)
        return trace

    def query(
        self,
        question: str,
        *,
        context: str | None = None,
        query_time: datetime | None = None,
    ) -> RunTrace:
        """Ask a question against the memory graph.

        Args:
            question: Natural-language question.
            context: Optional framing for the query agent.
            query_time: Override the agent's sense of "now". Useful for
                replaying benchmark scenarios at a fixed point in time.

        Returns:
            RunTrace with ``mode="query"`` and ``completion_payload["answer"]``.
        """
        return run_query(question, self._graph, context=context, query_time=query_time)

    def continue_session(self, session_id: str, message: str) -> RunTrace:
        """Append *message* to an existing session and run the query agent.

        Args:
            session_id: Session identifier, e.g. ``"s:1"``.
            message: Follow-up question or message.

        Returns:
            RunTrace with ``mode="query"``.
        """
        return run_session(session_id, "query", self._graph, message)

    def update_themes(self) -> RunTrace:
        """Run the theme-update agent over the current graph.

        Returns:
            RunTrace with ``mode="theme"``.
        """
        return run_theme_update(self._graph)

    def export_backup(self, path: str) -> BackupSummary:
        """Export the current graph to a complete local JSON backup."""
        return export_graph_backup(self._graph, path, graph_name=self._graph_name)

    def import_backup(self, path: str, *, replace: bool = False) -> BackupSummary:
        """Import a complete local JSON backup into this graph.

        Args:
            path: JSON backup path.
            replace: Required when the target graph is non-empty.
        """
        summary = import_graph_backup(
            self._graph, path, replace=replace, graph_name=self._graph_name
        )
        return summary

    def reset(self) -> None:
        """Drop and reinitialise the graph.

        Deletes all graph data then recreates the System node and vector index.
        Intended for benchmark isolation — not for production use.
        """
        delete_graph_if_exists(self._graph)
        init_system(self._graph, embedding_dim=get_dimension())
