"""Embedding utilities — thin wrapper around LiteLLM for vector operations.

``EMBEDDING_MODEL=weavy/hash-<dim>`` selects a local, deterministic
bag-of-words hashing embedder instead of a LiteLLM provider. It needs no
network or API key and is meant for offline development and retrieval-mechanics
evaluation. It is NOT semantic: similarity only reflects shared word tokens.
"""

import hashlib
import math
import re
from functools import lru_cache

import litellm

from weavy.config import settings

LOCAL_HASH_PREFIX = "weavy/hash-"
_TOKEN_RE = re.compile(r"\w+", re.UNICODE)


def _local_hash_dim(model: str) -> int | None:
    if not model.startswith(LOCAL_HASH_PREFIX):
        return None
    suffix = model.removeprefix(LOCAL_HASH_PREFIX)
    if not suffix.isdigit() or int(suffix) < 2:
        raise ValueError(f"invalid local hash embedding model {model!r}")
    return int(suffix)


def hash_embed(text: str, dim: int) -> list[float]:
    """Signed feature hashing of lowercase word tokens, L2-normalized."""
    vec = [0.0] * dim
    for token in _TOKEN_RE.findall(text.lower()):
        digest = hashlib.sha256(token.encode()).digest()
        index = int.from_bytes(digest[:4], "big") % dim
        vec[index] += 1.0 if digest[4] & 1 else -1.0
    norm = math.sqrt(sum(v * v for v in vec))
    if norm == 0.0:
        # No tokens: a fixed unit vector keeps cosine defined.
        vec[0] = 1.0
        return vec
    return [v / norm for v in vec]


@lru_cache(maxsize=1)
def get_dimension() -> int:
    local_dim = _local_hash_dim(settings.EMBEDDING_MODEL)
    if local_dim is not None:
        return local_dim
    info = litellm.get_model_info(settings.EMBEDDING_MODEL)
    dim = info.get("output_vector_size")
    if dim is None:
        raise RuntimeError(
            f"litellm has no output_vector_size for {settings.EMBEDDING_MODEL!r}"
        )
    return int(dim)


def embed(text: str) -> list[float]:
    local_dim = _local_hash_dim(settings.EMBEDDING_MODEL)
    if local_dim is not None:
        return hash_embed(text, local_dim)
    response = litellm.embedding(
        model=settings.EMBEDDING_MODEL,
        input=[text],
    )
    return response.data[0]["embedding"]


def embed_node(aliases: list[str], summary: str) -> list[float]:
    """Embed a node's identity: aliases + current summary only.

    Past facts are recovered by navigating to source episodes, not by
    accumulating note history into this vector — see CLAUDE.md's navigational
    memory model.
    """
    return embed(" | ".join(aliases) + " — " + summary)


def embed_edge(label: str, fact: str) -> list[float]:
    return embed(f"{label} — {fact}")
