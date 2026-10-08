"""
Theme CRUD and anchor management.
Requires a running FalkorDB instance (provided by devenv up).
Uses the "weavy_test" graph to avoid touching the main graph.
"""

import pytest
from falkordb import Graph

from weavy.application.prompts import build_themes_context
from weavy.store import themes as store_themes
from tests.helpers import reset_test_graph


@pytest.fixture
def graph() -> Graph:
    return reset_test_graph("Theme")


# ---------------------------------------------------------------------------
# create_theme / get_theme
# ---------------------------------------------------------------------------


def test_create_theme_basic(graph: Graph) -> None:
    store_themes.create_theme(
        graph, "career-direction", "Weighing a job change.", ["node:1"]
    )
    theme = store_themes.get_theme(graph, "career-direction")
    assert theme.name == "career-direction"
    assert theme.state == "Weighing a job change."
    assert "node:1" in theme.anchors


def test_create_theme_no_anchors(graph: Graph) -> None:
    store_themes.create_theme(graph, "sleep-routine", "Tracking sleep.", [])
    result = store_themes.get_theme(graph, "sleep-routine")
    assert result.anchors == []


def test_get_theme_not_found(graph: Graph) -> None:
    with pytest.raises(ValueError, match="not found"):
        store_themes.get_theme(graph, "nonexistent")


# ---------------------------------------------------------------------------
# update_theme
# ---------------------------------------------------------------------------


def test_update_theme_state_only(graph: Graph) -> None:
    store_themes.create_theme(graph, "meditation-practice", "Just started.", ["node:1"])
    store_themes.update_theme(
        graph,
        "meditation-practice",
        new_state="Practiced for 2 weeks.",
        new_anchors=None,
    )
    result = store_themes.get_theme(graph, "meditation-practice")
    assert result.state == "Practiced for 2 weeks."
    assert "node:1" in result.anchors  # unchanged


def test_update_theme_anchors_add_remove(graph: Graph) -> None:
    store_themes.create_theme(graph, "mental-health", "Exploring emotions.", ["node:1"])
    store_themes.update_theme(
        graph, "mental-health", new_state=None, new_anchors=["node:2"]
    )
    result = store_themes.get_theme(graph, "mental-health")
    assert "node:1" not in result.anchors
    assert "node:2" in result.anchors


def test_update_theme_not_found(graph: Graph) -> None:
    with pytest.raises(ValueError, match="not found"):
        store_themes.update_theme(graph, "nonexistent", new_state="x", new_anchors=None)


# ---------------------------------------------------------------------------
# retire_theme
# ---------------------------------------------------------------------------


def test_retire_theme(graph: Graph) -> None:
    store_themes.create_theme(graph, "exercise", "Building fitness habit.", ["node:1"])
    store_themes.retire_theme(graph, "exercise")
    with pytest.raises(ValueError, match="not found"):
        store_themes.get_theme(graph, "exercise")


def test_retire_theme_not_found(graph: Graph) -> None:
    with pytest.raises(ValueError, match="not found"):
        store_themes.retire_theme(graph, "nonexistent")


# ---------------------------------------------------------------------------
# list_all_themes
# ---------------------------------------------------------------------------


def test_list_all_themes_empty(graph: Graph) -> None:
    assert store_themes.list_all_themes(graph) == []


def test_list_all_themes_multiple(graph: Graph) -> None:
    store_themes.create_theme(graph, "career", "Career focus.", ["node:1"])
    store_themes.create_theme(graph, "health", "Health matters.", [])
    themes = store_themes.list_all_themes(graph)
    names = {t.name for t in themes}
    assert names == {"career", "health"}


def test_list_all_themes_orders_by_name(graph: Graph) -> None:
    # Previously ordered by an agent-maintained `priority` rank reconciled on
    # every theme-run (see git history). Themes are now a plain name menu
    # (see application/prompts.build_themes_context) with nothing to rank, so
    # listing order is just name order — deterministic, no agent upkeep.
    for name in ("c", "a", "b"):
        store_themes.create_theme(graph, name, f"state {name}", [])
    assert [t.name for t in store_themes.list_all_themes(graph)] == ["a", "b", "c"]


def test_retire_theme_removes_it_from_listing(graph: Graph) -> None:
    for name in ("a", "b"):
        store_themes.create_theme(graph, name, f"state {name}", [])
    store_themes.retire_theme(graph, "a")
    assert [t.name for t in store_themes.list_all_themes(graph)] == ["b"]


# ---------------------------------------------------------------------------
# build_themes_context — the ingestion/query-facing theme menu (names only;
# full `state` is fetched on demand via get_theme, not pre-rendered here)
# ---------------------------------------------------------------------------


def test_build_themes_context_empty_uses_fallback(graph: Graph) -> None:
    assert build_themes_context(graph, empty_msg="(none)") == "(none)"


def test_build_themes_context_lists_names_only(graph: Graph) -> None:
    store_themes.create_theme(
        graph, "career-direction", "A long state description.", ["node:1"]
    )
    store_themes.create_theme(graph, "sleep-routine", "Another long description.", [])
    context = build_themes_context(graph)
    assert "career-direction" in context
    assert "sleep-routine" in context
    # names only — full state text must not leak into the menu
    assert "long state description" not in context.lower()
