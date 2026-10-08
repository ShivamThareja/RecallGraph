"""Step 2 retrieval behaviour: temporal filtering before top-k, open/naive time
ranges, distance thresholds, alias mentions, provenance on results, bounded
graph expansion, neighbourhood caps, and UTC-correct session listing.

Requires FalkorDB. Embeddings are the deterministic fakes from conftest.
"""

from datetime import datetime, timedelta, timezone

import pytest
from falkordb import Graph

from tests.conftest import _fake_embed
from tests.helpers import make_test_trace, reset_test_graph, store_test_session_with_id
from weavy.config import settings
from weavy.models.canonical import ChatMessage, Session
from weavy.models.graph import ProvenanceInput
from weavy.services import memory
from weavy.store import canonical as store_canonical
from weavy.store import graph as store_graph

UTC = timezone.utc


@pytest.fixture
def graph() -> Graph:
    return reset_test_graph("Session")


def _node(graph, alias, summary="x", *, session="s:1", happened=None, recorded=None):
    return memory.create_node(
        graph,
        aliases=[alias],
        summary=summary,
        note="n",
        provenance=ProvenanceInput(source_id=session),
        trace=make_test_trace(),
        event_time=recorded,
        happened_at=happened,
        session_id=session,
        force=True,
    ).id


def _edge(graph, a, b, label="related to", fact="a relates to b", session="s:1"):
    return memory.create_edge(
        graph,
        from_node_id=a,
        to_node_id=b,
        label=label,
        fact=fact,
        note="n",
        provenance=ProvenanceInput(source_id=session),
        trace=make_test_trace(),
        source_id=session,
    ).id


def _trips(graph) -> dict[str, str]:
    """Six 'trip' nodes; the September one has no edges, so it ranks last
    among keyword hits and would fall outside a small top-k."""
    store_test_session_with_id(graph, "s:1")
    hub = _node(graph, "Traveller")
    months = {"jan": 1, "mar": 3, "may": 5, "jul": 7, "sep": 9, "nov": 11}
    ids = {}
    for name, month in months.items():
        ids[name] = _node(
            graph, f"{name} trip", happened=datetime(2023, month, 15, tzinfo=UTC)
        )
        if name != "sep":
            _edge(graph, hub, ids[name])
    return ids


# --- temporal ---------------------------------------------------------------


def test_time_filter_applies_before_top_k(graph) -> None:
    ids = _trips(graph)
    window = [datetime(2023, 9, 1, tzinfo=UTC), datetime(2023, 9, 30, tzinfo=UTC)]

    unfiltered = store_graph.search_graph(graph, query="trip", limit=2)
    assert ids["sep"] not in [r.id for r in unfiltered.results]  # crowded out

    out = store_graph.search_graph(graph, query="trip", limit=2, time_range=window)
    assert [r.id for r in out.results] == [ids["sep"]]
    assert out.diagnostics.dropped_by_time == 5


def test_open_ended_time_ranges(graph) -> None:
    ids = _trips(graph)
    since_oct = store_graph.search_graph(
        graph,
        query="trip",
        limit=5,
        time_range=[datetime(2023, 10, 1, tzinfo=UTC), None],
    )
    assert {r.id for r in since_oct.results} == {ids["nov"]}
    until_feb = store_graph.search_graph(
        graph,
        query="trip",
        limit=5,
        time_range=[None, datetime(2023, 2, 1, tzinfo=UTC)],
    )
    assert {r.id for r in until_feb.results} == {ids["jan"]}


def test_naive_and_offset_ranges_compare_in_utc(graph) -> None:
    ids = _trips(graph)
    # Naive bounds are read as UTC — no naive/aware TypeError.
    naive = store_graph.search_graph(
        graph,
        query="trip",
        limit=5,
        time_range=[datetime(2023, 7, 1), datetime(2023, 7, 31)],
    )
    assert [r.id for r in naive.results] == [ids["jul"]]
    # 2023-07-15T00:00Z is 2023-07-15T05:30+05:30: a +05:30 window ending at
    # 05:00 local excludes it, one ending at 06:00 local includes it.
    ist = timezone(timedelta(hours=5, minutes=30))
    before = [
        datetime(2023, 7, 14, tzinfo=ist),
        datetime(2023, 7, 15, 5, 0, tzinfo=ist),
    ]
    after = [datetime(2023, 7, 14, tzinfo=ist), datetime(2023, 7, 15, 6, 0, tzinfo=ist)]
    assert (
        store_graph.search_graph(graph, query="trip", time_range=before).results == []
    )
    assert [
        r.id
        for r in store_graph.search_graph(graph, query="trip", time_range=after).results
    ] == [ids["jul"]]


def test_inverted_or_malformed_range_rejected(graph) -> None:
    with pytest.raises(ValueError):
        store_graph.search_graph(
            graph,
            query="x",
            time_range=[
                datetime(2024, 2, 1, tzinfo=UTC),
                datetime(2024, 1, 1, tzinfo=UTC),
            ],
        )
    with pytest.raises(ValueError):
        store_graph.search_graph(graph, query="x", time_range=[datetime(2024, 1, 1)])


# --- vector threshold -------------------------------------------------------


def test_max_distance_drops_unrelated_vector_hits(graph) -> None:
    store_test_session_with_id(graph, "s:1")
    target = _node(graph, "alpha", "first")
    _node(graph, "beta", "second")
    _node(graph, "gamma", "third")
    query_vec = _fake_embed("alpha — first")  # == the target's embed_node text

    no_cut = store_graph.search_graph(
        graph, query="zzz-no-keyword", query_embedding=query_vec, limit=5
    )
    assert len(no_cut.results) == 3  # nearest neighbours returned regardless

    cut = store_graph.search_graph(
        graph,
        query="zzz-no-keyword",
        query_embedding=query_vec,
        limit=5,
        max_distance=0.05,
    )
    assert [r.id for r in cut.results] == [target]
    assert cut.diagnostics.dropped_by_distance == 2
    assert cut.diagnostics.max_distance == 0.05


def test_keyword_hits_survive_distance_cut(graph) -> None:
    store_test_session_with_id(graph, "s:1")
    nid = _node(graph, "Biscuit", "a greyhound")
    far = [1.0] + [0.0] * 7
    out = store_graph.search_graph(
        graph, query="Biscuit", query_embedding=far, max_distance=0.0001
    )
    assert [r.id for r in out.results] == [nid]


def test_memory_search_uses_configured_threshold(graph, monkeypatch) -> None:
    store_test_session_with_id(graph, "s:1")
    _node(graph, "alpha", "first")
    monkeypatch.setattr(settings, "SEARCH_MAX_DISTANCE", 0.0001)
    assert memory.search_graph(graph, query="unrelated words").results == []
    # An explicit override wins over the setting; None disables the cut.
    out = memory.search_graph(graph, query="unrelated words", max_distance=None)
    assert len(out.results) == 1


# --- query processing -------------------------------------------------------


def test_alias_mentioned_inside_question_is_a_keyword_hit(graph) -> None:
    store_test_session_with_id(graph, "s:1")
    nid = _node(graph, "Biscuit", "a greyhound")
    out = store_graph.search_graph(graph, query="When did I adopt Biscuit?")
    assert [r.id for r in out.results] == [nid]
    # Word boundaries: "Biscuits" is not a mention of "Biscuit".
    assert store_graph.search_graph(graph, query="I like biscuitsx").results == []


def test_short_aliases_are_not_mention_matched(graph) -> None:
    store_test_session_with_id(graph, "s:1")
    _node(graph, "me", "the user")
    assert store_graph.search_graph(graph, query="tell me about it").results == []


def test_normalize_query_preserves_temporal_content() -> None:
    assert store_graph.normalize_query("  trips in   2023-09 \n ") == "trips in 2023-09"
    assert store_graph.normalize_query("ｆｕｌｌ width") == "full width"


def test_empty_query_returns_nothing(graph) -> None:
    store_test_session_with_id(graph, "s:1")
    _node(graph, "alpha")
    assert store_graph.search_graph(graph, query="   ").results == []


# --- provenance on results --------------------------------------------------


def test_results_expose_sources_and_time_span(graph) -> None:
    for sid in ("s:1", "s:2", "s:3"):
        store_test_session_with_id(graph, sid)
    nid = _node(
        graph,
        "Lena",
        "sister",
        session="s:1",
        recorded=datetime(2024, 3, 5, tzinfo=UTC),
        happened=datetime(2024, 3, 4, tzinfo=UTC),
    )
    memory.update_node(
        graph,
        node_id=nid,
        note="lives in Berlin",
        provenance=ProvenanceInput(source_id="s:2"),
        trace=make_test_trace(),
        event_time=datetime(2024, 3, 20, tzinfo=UTC),
        session_id="s:2",
    )
    row = store_graph.search_graph(graph, query="Lena").results[0]
    assert row.source_sessions == ["s:2", "s:1"]  # most recent first
    assert row.source_count == 2
    assert row.earliest_happened_at == datetime(2024, 3, 4, tzinfo=UTC)
    assert row.latest_happened_at == datetime(2024, 3, 20, tzinfo=UTC)
    assert row.last_recorded_at == datetime(2024, 3, 20, tzinfo=UTC)

    capped = store_graph.search_graph(graph, query="Lena", max_sources=1).results[0]
    assert capped.source_sessions == ["s:2"] and capped.source_count == 2


def test_edge_results_expose_source(graph) -> None:
    store_test_session_with_id(graph, "s:1")
    store_test_session_with_id(graph, "s:4")
    a, b = _node(graph, "Ada"), _node(graph, "Acme")
    eid = _edge(graph, a, b, "works at", "Ada works at Acme", session="s:4")
    row = next(
        r
        for r in store_graph.search_graph(graph, query="works at").results
        if r.id == eid
    )
    assert row.source_sessions == ["s:4"]


def test_edge_writes_link_mentions_to_both_endpoints(graph) -> None:
    store_test_session_with_id(graph, "s:1")
    store_test_session_with_id(graph, "s:2")
    store_test_session_with_id(graph, "s:3")
    a, b = _node(graph, "Ada"), _node(graph, "Acme")
    eid = _edge(graph, a, b, "works at", "Ada works at Acme", session="s:2")
    out = memory.get_node(graph, node_ids=[a, b])
    assert all("s:2" in r.mentioned_by for r in out.results)

    memory.update_edge(
        graph,
        edge_id=eid,
        note="promoted",
        provenance=ProvenanceInput(source_id="s:3"),
        trace=make_test_trace(),
        new_fact="Ada leads a team at Acme",
        session_id="s:3",
    )
    out = memory.get_node(graph, node_ids=[a, b])
    assert all("s:3" in r.mentioned_by for r in out.results)


# --- graph-aware expansion --------------------------------------------------


def _chain(graph) -> dict[str, str]:
    store_test_session_with_id(graph, "s:1")
    ids = {k: _node(graph, k) for k in ("Biscuit", "Lena", "Gaia shelter", "Berlin")}
    ids["e1"] = _edge(
        graph, ids["Lena"], ids["Biscuit"], "helped choose", "Lena helped"
    )
    ids["e2"] = _edge(
        graph, ids["Biscuit"], ids["Gaia shelter"], "from", "came from Gaia"
    )
    ids["e3"] = _edge(
        graph, ids["Lena"], ids["Berlin"], "lives in", "Lena lives in Berlin"
    )
    return ids


def test_expansion_adds_bounded_neighbours_with_via(graph) -> None:
    ids = _chain(graph)
    out = store_graph.search_graph(
        graph, query="Biscuit", limit=1, expand_depth=1, expand_limit=5
    )
    direct = [r for r in out.results if r.via is None]
    related = [r for r in out.results if r.via is not None]
    assert [r.id for r in direct] == [ids["Biscuit"]]
    assert {r.id for r in related} == {ids["Lena"], ids["Gaia shelter"]}
    assert all(r.via == ids["Biscuit"] and r.hops == 1 for r in related)
    assert {r.via_edge for r in related} == {ids["e1"], ids["e2"]}
    assert all(r.source_sessions == ["s:1"] for r in related)
    assert out.diagnostics.expanded == 2


def test_expansion_depth_and_limit(graph) -> None:
    ids = _chain(graph)
    two_hops = store_graph.search_graph(
        graph, query="Biscuit", limit=1, expand_depth=2, expand_limit=5
    )
    berlin = next(r for r in two_hops.results if r.id == ids["Berlin"])
    assert berlin.hops == 2 and berlin.via == ids["Lena"]

    capped = store_graph.search_graph(
        graph, query="Biscuit", limit=1, expand_depth=2, expand_limit=1
    )
    assert len([r for r in capped.results if r.via]) == 1

    off = store_graph.search_graph(graph, query="Biscuit", limit=1)
    assert all(r.via is None for r in off.results)


def test_expansion_respects_time_range(graph) -> None:
    store_test_session_with_id(graph, "s:1")
    a = _node(graph, "Biscuit", happened=datetime(2024, 3, 4, tzinfo=UTC))
    old = _node(graph, "Old friend", happened=datetime(2010, 1, 1, tzinfo=UTC))
    _edge(graph, old, a)
    out = store_graph.search_graph(
        graph,
        query="Biscuit",
        time_range=[datetime(2024, 1, 1, tzinfo=UTC), None],
        expand_depth=1,
        expand_limit=5,
    )
    assert [r.id for r in out.results] == [a]


def test_neighborhood_is_capped(graph) -> None:
    store_test_session_with_id(graph, "s:1")
    hub = _node(graph, "hub")
    for i in range(4):
        _edge(graph, hub, _node(graph, f"leaf {i}"))
    out = store_graph.get_node_neighborhood(graph, hub, limit=2)
    assert len(out.neighbors) == 2
    assert out.total_neighbors == 4 and out.truncated
    full = store_graph.get_node_neighborhood(graph, hub, limit=10)
    assert len(full.neighbors) == 4 and not full.truncated


# --- session listing --------------------------------------------------------


def test_list_sessions_orders_and_filters_in_utc(graph) -> None:
    ist = timezone(timedelta(hours=5, minutes=30))
    for sid, ts in [
        ("s:1", datetime(2024, 1, 1, 23, 0, tzinfo=UTC)),
        # 2024-01-02T03:00+05:30 == 2024-01-01T21:30Z: earlier than s:1.
        ("s:2", datetime(2024, 1, 2, 3, 0, tzinfo=ist)),
        ("s:3", datetime(2024, 1, 3, 12, 0, tzinfo=UTC)),
    ]:
        store_canonical.create_session(
            graph,
            Session(
                id=sid, timestamp=ts, messages=[ChatMessage(role="user", content="t")]
            ),
        )
    ordered = [s.id for s in store_canonical.list_sessions(graph).sessions]
    assert ordered == ["s:3", "s:1", "s:2"]
    jan1 = store_canonical.list_sessions(
        graph, date_range=[datetime(2024, 1, 1), datetime(2024, 1, 1, 23, 59)]
    )
    assert {s.id for s in jan1.sessions} == {"s:1", "s:2"}
    since = store_canonical.list_sessions(
        graph, date_range=[datetime(2024, 1, 2, tzinfo=UTC), None]
    )
    assert [s.id for s in since.sessions] == ["s:3"]
