from weavy.application.session_runs import _merge_graph_changes


def test_merge_graph_changes_deduplicates_existing_and_new_ids() -> None:
    merged = _merge_graph_changes(
        {"nodes_created": ["node:1"]},
        {
            "nodes_created": ["node:1", "node:2", "node:2"],
            "edges_deleted": ["edge:1", "edge:1"],
        },
    )

    assert merged == {
        "nodes_created": ["node:1", "node:2"],
        "edges_deleted": ["edge:1"],
    }
