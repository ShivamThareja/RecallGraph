from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from falkordb import Graph

from weavy.application.prompts import (
    fetch_prompt,
    render_full_theme_map,
    render_session_journal,
)
from weavy.harness.actions import THEME_ACTIONS
from weavy.harness.runner import run
from weavy.models.traces import RunTrace
from weavy.store import canonical as store_canonical
from weavy.store import system as store_system
from weavy.store import themes as store_themes


def finalize_theme(graph: Graph, trace: RunTrace) -> RunTrace:
    """Record the run's completion time.

    Previously also reconciled an agent-supplied ``priority_order`` onto a
    stored rank per theme (see git history for `_reconcile_priority`). Themes
    are now surfaced to ingestion/query as a plain name menu (see
    `application/prompts.build_themes_context`), so there is nothing left to
    reconcile here — the theme agent's job ends at create/update/retire.
    """
    if trace.status != "completed":
        return trace

    store_system.update_last_theme_run_at(
        graph, datetime.now(tz=timezone.utc).isoformat()
    )
    return trace


def run_theme_update(
    graph: Graph,
    parent_observation: Any = None,
) -> RunTrace:
    system_state = store_system.get_system(graph)
    all_themes = store_themes.list_all_themes(graph)

    system_prompt = fetch_prompt(
        "weavy-theme",
        {
            "theme_map": render_full_theme_map(all_themes),
            "preface": system_state.preface or "(not set)",
        },
    )

    sessions = store_canonical.get_sessions_since(graph, system_state.last_theme_run_at)
    if sessions:
        user_content = render_session_journal(sessions)
    else:
        user_content = (
            "No new sessions since your last run. Review existing themes for coherence."
        )

    trace = run(
        mode="theme",
        system_prompt=system_prompt,
        initial_messages=[{"role": "user", "content": user_content}],
        allowed_actions=THEME_ACTIONS,
        run_context={"input_summary": f"Theme update ({len(sessions)} sessions)"},
        graph=graph,
        parent_observation=parent_observation,
    )
    return finalize_theme(graph, trace)
