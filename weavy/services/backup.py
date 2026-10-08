from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from falkordb import Graph

from weavy.store.client import delete_graph_if_exists
from weavy.store.graph import _vecf32_literal
from weavy.store.system import _ensure_vector_index

BACKUP_SCHEMA_VERSION = 1

_TOP_LEVEL_KEYS = {
    "metadata",
    "system",
    "sessions",
    "semantic_nodes",
    "semantic_edges",
    "themes",
    "run_traces",
}


@dataclass(frozen=True)
class BackupSummary:
    path: str
    graph_name: str | None
    sessions: int
    semantic_nodes: int
    semantic_edges: int
    themes: int
    run_traces: int


def export_backup(
    graph: Graph, path: str | Path, graph_name: str | None = None
) -> BackupSummary:
    """Export all durable local Weavy graph state to one JSON file."""
    target = Path(path)
    data = {
        "metadata": {
            "schema_version": BACKUP_SCHEMA_VERSION,
            "exported_at": datetime.now(tz=timezone.utc).isoformat(),
            "source_graph": graph_name,
        },
        "system": _get_single_node_props(graph, "System"),
        "sessions": _all_node_props(graph, "Session", "id"),
        "semantic_nodes": _all_semantic_nodes(graph),
        "semantic_edges": _all_semantic_edges(graph),
        "themes": _all_node_props(graph, "Theme", "name"),
        "run_traces": _all_node_props(graph, "RunTrace", "id"),
    }

    target.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
    return _summary_from_data(data, target, graph_name)


def import_backup(
    graph: Graph,
    path: str | Path,
    *,
    replace: bool = False,
    graph_name: str | None = None,
) -> BackupSummary:
    """Restore a complete backup into a graph, replacing only when explicit."""
    source = Path(path)
    data = json.loads(source.read_text())
    _validate_backup(data)

    if _graph_has_data(graph):
        if not replace:
            raise RuntimeError("Target graph is not empty. Re-run with replace=True.")
        delete_graph_if_exists(graph)

    _restore_system(graph, data["system"])
    _restore_vector_index(graph, data["semantic_nodes"])
    _restore_nodes(graph, "Session", data["sessions"])
    _restore_semantic_nodes(graph, data["semantic_nodes"])
    _restore_edges(graph, data["semantic_edges"])
    _restore_nodes(graph, "Theme", data["themes"])
    _restore_nodes(graph, "RunTrace", data["run_traces"])
    return _summary_from_data(data, source, graph_name)


def _summary_from_data(
    data: dict[str, Any], path: Path, graph_name: str | None
) -> BackupSummary:
    return BackupSummary(
        path=str(path),
        graph_name=graph_name,
        sessions=len(data["sessions"]),
        semantic_nodes=len(data["semantic_nodes"]),
        semantic_edges=len(data["semantic_edges"]),
        themes=len(data["themes"]),
        run_traces=len(data["run_traces"]),
    )


def _validate_backup(data: Any) -> None:
    if not isinstance(data, dict):
        raise ValueError("Backup must be a JSON object.")
    missing = sorted(_TOP_LEVEL_KEYS - set(data))
    if missing:
        raise ValueError(f"Backup missing required keys: {', '.join(missing)}")
    if not isinstance(data["metadata"], dict):
        raise ValueError("Backup key 'metadata' must be an object.")
    version = data["metadata"].get("schema_version")
    if version != BACKUP_SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported backup schema_version {version!r}; expected {BACKUP_SCHEMA_VERSION}."
        )
    for key in _TOP_LEVEL_KEYS - {"metadata", "system"}:
        if not isinstance(data[key], list):
            raise ValueError(f"Backup key {key!r} must be a list.")
    if not isinstance(data["system"], dict):
        raise ValueError("Backup key 'system' must be an object.")


def _graph_has_data(graph: Graph) -> bool:
    result = graph.query("MATCH (n) RETURN count(n)")
    node_count = result.result_set[0][0] if result.result_set else 0
    if node_count:
        return True
    result = graph.query("MATCH ()-[r]->() RETURN count(r)")
    edge_count = result.result_set[0][0] if result.result_set else 0
    return bool(edge_count)


def _get_single_node_props(graph: Graph, label: str) -> dict[str, Any]:
    result = graph.query(f"MATCH (n:{label}) RETURN n LIMIT 1")
    if not result.result_set:
        return {}
    return _jsonable_props(result.result_set[0][0].properties)


def _all_node_props(graph: Graph, label: str, sort_key: str) -> list[dict[str, Any]]:
    result = graph.query(f"MATCH (n:{label}) RETURN n")
    rows = [_jsonable_props(row[0].properties) for row in result.result_set]
    return sorted(rows, key=lambda row: row.get(sort_key) or "")


def _all_semantic_nodes(graph: Graph) -> list[dict[str, Any]]:
    nodes = _all_node_props(graph, "SemanticNode", "id")
    return sorted(nodes, key=lambda row: _id_sort_key(row.get("id", "")))


def _all_semantic_edges(graph: Graph) -> list[dict[str, Any]]:
    result = graph.query(
        """
        MATCH (a:SemanticNode)-[r:RELATES]->(b:SemanticNode)
        RETURN a.id, b.id, r
        """
    )
    edges = []
    for from_id, to_id, rel in result.result_set:
        props = _jsonable_props(rel.properties)
        props["from_node_id"] = from_id
        props["to_node_id"] = to_id
        edges.append(props)
    return sorted(edges, key=lambda row: _id_sort_key(row.get("id", "")))


def _jsonable_props(props: dict[str, Any]) -> dict[str, Any]:
    return {key: _jsonable_value(value) for key, value in props.items()}


def _jsonable_value(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, dict):
        return {str(k): _jsonable_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable_value(v) for v in value]
    try:
        return [_jsonable_value(v) for v in value]
    except TypeError:
        return str(value)


def _id_sort_key(value: Any) -> tuple[str, int]:
    text = str(value)
    prefix, _, suffix = text.partition(":")
    try:
        number = int(suffix)
    except ValueError:
        number = -1
    return (prefix, number)


def _restore_system(graph: Graph, props: dict[str, Any]) -> None:
    if not props:
        return
    graph.query("CREATE (s:System) SET s = $props", {"props": props})


def _restore_vector_index(graph: Graph, semantic_nodes: list[dict[str, Any]]) -> None:
    for node in semantic_nodes:
        embedding = node.get("embedding")
        if isinstance(embedding, list) and embedding:
            _ensure_vector_index(graph, len(embedding))
            return


def _restore_nodes(graph: Graph, label: str, rows: list[dict[str, Any]]) -> None:
    for props in rows:
        graph.query(f"CREATE (n:{label}) SET n = $props", {"props": props})


def _restore_semantic_nodes(graph: Graph, rows: list[dict[str, Any]]) -> None:
    for props in rows:
        embedding = props.get("embedding")
        props_without_embedding = dict(props)
        props_without_embedding.pop("embedding", None)
        embedding_clause = ""
        if isinstance(embedding, list):
            embedding_clause = f", n.embedding = {_vecf32_literal(embedding)}"
        graph.query(
            f"CREATE (n:SemanticNode) SET n = $props{embedding_clause}",
            {"props": props_without_embedding},
        )


def _restore_edges(graph: Graph, rows: list[dict[str, Any]]) -> None:
    for row in rows:
        from_node_id = row["from_node_id"]
        to_node_id = row["to_node_id"]
        props = dict(row)
        props.pop("from_node_id", None)
        props.pop("to_node_id", None)
        result = graph.query(
            """
            MATCH (a:SemanticNode {id: $from_node_id}), (b:SemanticNode {id: $to_node_id})
            CREATE (a)-[r:RELATES]->(b)
            SET r = $props
            RETURN r
            """,
            {
                "from_node_id": from_node_id,
                "to_node_id": to_node_id,
                "props": props,
            },
        )
        if not result.result_set:
            raise ValueError(
                "Cannot restore edge: missing endpoint "
                f"(from={from_node_id!r}, to={to_node_id!r})."
            )
