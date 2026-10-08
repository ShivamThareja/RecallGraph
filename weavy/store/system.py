"""
System node — singleton node in FalkorDB that holds global counters and config.
"""

from typing import Literal

from falkordb import Graph
from pydantic import BaseModel

from weavy.application.contracts import OperationResult

CounterName = Literal["node", "edge", "session"]

_COUNTER_FIELD: dict[str, str] = {
    "node": "next_node_id",
    "edge": "next_edge_id",
    "session": "next_session_id",
}


class SystemState(BaseModel):
    next_node_id: int
    next_edge_id: int
    next_session_id: int
    last_theme_run_at: str
    preface: str | None = None  # what this graph is about


def init_system(graph: Graph, embedding_dim: int | None = None) -> SystemState:
    """Create the System node if it does not exist; return current state.

    When *embedding_dim* is provided, a vector index is created (or verified)
    on SemanticNode.embedding for hybrid search.

    Previously also seeded a ``hot_theme_token_budget`` used to ration how many
    themes got rendered in full per run (see git history). Themes are now
    surfaced as a plain name menu with no budget to tune — see
    `application/prompts.build_themes_context`.
    """
    # `name` is not read by app code; FalkorDB Browser uses it as the node's
    # display caption. Keep in sync with `id`/`aliases[0]` on every node type
    # that sets it (SemanticNode, Session, System).
    result = graph.query(
        """
        MERGE (s:System)
        ON CREATE SET
            s.name = 'System',
            s.next_node_id = 1,
            s.next_edge_id = 1,
            s.next_session_id = 1,
            s.last_theme_run_at = '1970-01-01T00:00:00+00:00'
        RETURN s
        """
    )

    if embedding_dim is not None:
        _ensure_vector_index(graph, embedding_dim)

    return SystemState(**result.result_set[0][0].properties)


def _ensure_vector_index(graph: Graph, dim: int) -> None:
    """Create vector indexes for entity nodes and relationship facts (idempotent)."""
    statements = (
        f"CREATE VECTOR INDEX FOR (n:SemanticNode) ON (n.embedding) "
        f"OPTIONS {{dimension:{dim}, similarityFunction:'cosine'}}",
        f"CREATE VECTOR INDEX FOR ()-[r:RELATES]->() ON (r.embedding) "
        f"OPTIONS {{dimension:{dim}, similarityFunction:'cosine'}}",
    )
    for stmt in statements:
        try:
            graph.query(stmt)
        except Exception as e:
            msg = str(e).lower()
            if "already" in msg or "exists" in msg or "equivalent" in msg:
                continue
            raise


def get_system(graph: Graph) -> SystemState:
    """Read the System node. Raises RuntimeError if it does not exist."""
    result = graph.query("MATCH (s:System) RETURN s")
    if not result.result_set:
        raise RuntimeError(
            "System node not found. Run init_system() or `python -m weavy.cli init-system` first."
        )
    return SystemState(**result.result_set[0][0].properties)


def set_preface(graph: Graph, preface: str) -> OperationResult:
    result = graph.query(
        "MATCH (s:System) SET s.preface = $preface RETURN s",
        {"preface": preface},
    )
    if not result.result_set:
        raise RuntimeError("System node not found. Cannot update preface.")
    return OperationResult(ok=True)


def update_last_theme_run_at(graph: Graph, iso_timestamp: str) -> None:
    result = graph.query(
        "MATCH (s:System) SET s.last_theme_run_at = $ts RETURN s",
        {"ts": iso_timestamp},
    )
    if not result.result_set:
        raise RuntimeError("System node not found. Cannot update last_theme_run_at.")


def increment_counter(graph: Graph, counter: CounterName) -> str:
    """
    Atomically increment the named counter and return the minted token.

    Returns e.g. "node:1" on the first call for counter="node".
    The counter field is left pointing at the next available value.
    """
    field = _COUNTER_FIELD[counter]
    prefix = "s" if counter == "session" else counter
    result = graph.query(
        f"""
        MATCH (s:System)
        SET s.{field} = s.{field} + 1
        RETURN s.{field} - 1
        """
    )
    if not result.result_set:
        raise RuntimeError(
            "System node not found. Cannot mint token — run init_system() first."
        )
    minted_id: int = result.result_set[0][0]
    return f"{prefix}:{minted_id}"
