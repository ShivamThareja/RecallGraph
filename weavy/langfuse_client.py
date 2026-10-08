"""
Shared Langfuse client access.

The client is created lazily so importing tracing or prompt helpers does not
eagerly initialize the SDK during unrelated code paths.

Langfuse is an optional dependency. Install with: uv add weavy[langfuse]
"""

from functools import lru_cache
from typing import Any

from weavy.config import settings


@lru_cache(maxsize=1)
def get_langfuse() -> Any:
    try:
        from langfuse import Langfuse
    except ImportError as e:
        raise ImportError(
            "Langfuse is not installed. Install it with: uv add weavy[langfuse]"
        ) from e
    return Langfuse(
        public_key=settings.LANGFUSE_PUBLIC_KEY,
        secret_key=settings.LANGFUSE_SECRET_KEY,
        base_url=settings.LANGFUSE_HOST,
    )
