"""Vector store interface (Protocol)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np


@dataclass(frozen=True)
class SearchHit:
    """One hit from a vector-store query."""

    item_id: str
    score: float  # cosine similarity in [-1, 1]; 1.0 = identical


class VectorStore(Protocol):
    """Minimal interface for top-K nearest-neighbour search.

    Implementations are responsible for normalization. Callers should not
    normalize before calling `search`; the store will do it consistently.
    """

    def __len__(self) -> int: ...

    def search(self, query: np.ndarray, k: int = 20) -> list[SearchHit]:
        """Return up to k highest-score hits."""

    def search_batch(self, queries: np.ndarray, k: int = 20) -> list[list[SearchHit]]:
        """Vectorized batch search. queries shape (n, dim)."""
