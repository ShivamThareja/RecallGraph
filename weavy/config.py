import os

import litellm
from dotenv import load_dotenv

load_dotenv()

# Single API key for all LiteLLM provider calls.
_api_key = os.getenv("LITELLM_API_KEY")
if _api_key:
    litellm.api_key = _api_key.strip()

_api_base = os.getenv("LITELLM_API_BASE")
if _api_base:
    litellm.api_base = _api_base.strip()


def _clean_env_value(value: str | None) -> str | None:
    if value is None:
        return None

    cleaned = value.strip()
    if len(cleaned) >= 2 and cleaned[0] == cleaned[-1] and cleaned[0] in {"'", '"'}:
        cleaned = cleaned[1:-1].strip()
    return cleaned


def _getenv(name: str, default: str, *, alias: str | None = None) -> str:
    value = _clean_env_value(os.getenv(name))
    if value is not None:
        return value
    if alias is not None:
        alias_value = _clean_env_value(os.getenv(alias))
        if alias_value is not None:
            return alias_value
    return default


def _langfuse_host() -> str:
    host = _getenv(
        "LANGFUSE_HOST",
        "http://localhost:3100",
        alias="LANGFUSE_BASE_URL",
    )
    if "://" not in host:
        return f"http://{host}"
    return host


def _optional_float(name: str, default: float | None) -> float | None:
    raw = _clean_env_value(os.getenv(name))
    if raw is None or raw == "":
        return default
    if raw.lower() == "none":
        return None
    return float(raw)


def _int(name: str, default: int, *, minimum: int = 0) -> int:
    raw = _clean_env_value(os.getenv(name))
    value = default if raw is None or raw == "" else int(raw)
    if value < minimum:
        raise ValueError(f"{name} must be >= {minimum}, got {value}")
    return value


class Settings:
    FALKORDB_HOST: str = _getenv("FALKORDB_HOST", "localhost")
    FALKORDB_PORT: int = int(_getenv("FALKORDB_PORT", "6379"))
    GRAPH_NAME: str = _getenv("GRAPH_NAME", "weavy")

    LLM_MODEL: str = _getenv("LLM_MODEL", "gemini/gemini-2.5-flash")
    EMBEDDING_MODEL: str = _getenv("EMBEDDING_MODEL", "gemini/gemini-embedding-001")
    REASONING_EFFORT: str = _getenv("REASONING_EFFORT", "medium")

    # --- Retrieval (search_graph) ------------------------------------------
    # Cosine distance above which a vector hit is treated as unrelated and
    # dropped (keyword hits are unaffected). "none" disables the cut-off.
    # This is a judgment call that depends on the embedding model: 0.6 is an
    # uncalibrated default meant to drop only clearly unrelated neighbours;
    # calibrate it with `weavy eval --mode retrieval --max-distance ...`.
    SEARCH_MAX_DISTANCE: float | None = _optional_float("SEARCH_MAX_DISTANCE", 0.6)
    # Candidates fetched per pass (vector/keyword × node/edge) before time
    # filtering and the final top-k cut. Larger = better recall under time
    # filters, more DB work.
    SEARCH_CANDIDATE_POOL: int = _int("SEARCH_CANDIDATE_POOL", 50, minimum=1)
    # Graph-aware expansion: neighbours of the top results, at most
    # SEARCH_EXPAND_LIMIT extra rows within SEARCH_EXPAND_DEPTH hops (0 = off).
    SEARCH_EXPAND_DEPTH: int = _int("SEARCH_EXPAND_DEPTH", 1)
    SEARCH_EXPAND_LIMIT: int = _int("SEARCH_EXPAND_LIMIT", 5)
    # Source sessions listed per search result (most recent first); the full
    # count is always reported.
    SEARCH_MAX_SOURCES: int = _int("SEARCH_MAX_SOURCES", 5, minimum=1)
    # Default cap on get_node_neighborhood neighbours.
    NEIGHBORHOOD_LIMIT: int = _int("NEIGHBORHOOD_LIMIT", 25, minimum=1)

    LANGFUSE_HOST: str = _langfuse_host()
    LANGFUSE_PUBLIC_KEY: str = _getenv("LANGFUSE_PUBLIC_KEY", "")
    LANGFUSE_SECRET_KEY: str = _getenv("LANGFUSE_SECRET_KEY", "")


settings = Settings()
