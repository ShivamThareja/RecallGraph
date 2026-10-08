"""
System node initialisation and counter minting tests.
Requires a running FalkorDB instance (provided by devenv up).
Uses the "weavy_test" graph to avoid touching the main graph.
"""

import pytest
from falkordb import Graph

from weavy.store.client import get_graph
from weavy.store.system import SystemState, get_system, increment_counter, init_system
from tests.helpers import TEST_GRAPH


@pytest.fixture
def graph() -> Graph:
    g = get_graph(TEST_GRAPH)
    g.query("MATCH (s:System) DELETE s")
    yield g
    g.query("MATCH (s:System) DELETE s")


def test_init_system_creates_node(graph: Graph) -> None:
    state = init_system(graph)

    assert isinstance(state, SystemState)
    assert state.next_node_id == 1
    assert state.next_edge_id == 1
    assert state.next_session_id == 1
    assert state.last_theme_run_at == "1970-01-01T00:00:00+00:00"


def test_init_system_idempotent(graph: Graph) -> None:
    state_a = init_system(graph)
    state_b = init_system(graph)

    assert state_a == state_b


def test_get_system_raises_if_missing(graph: Graph) -> None:
    with pytest.raises(RuntimeError, match="System node not found"):
        get_system(graph)


def test_increment_counter_all_types(graph: Graph) -> None:
    init_system(graph)

    assert increment_counter(graph, "node") == "node:1"
    assert increment_counter(graph, "edge") == "edge:1"
    assert increment_counter(graph, "session") == "s:1"

    assert increment_counter(graph, "node") == "node:2"
    assert increment_counter(graph, "session") == "s:2"


def test_increment_counter_raises_without_init(graph: Graph) -> None:
    with pytest.raises(RuntimeError, match="System node not found"):
        increment_counter(graph, "node")
