"""In-memory cosine vector store using batched numpy matmul.

Comfortable for up to ~1M vectors at 1536 dim (~6 GB). Above that,
swap in FAISS or pgvector. The implementation pre-normalizes vectors
once so `search` is a single matmul.
"""
from __future__ import annotations

import numpy as np

from .base import SearchHit, VectorStore


class NumpyVectorStore(VectorStore):
    """L2-normalized dense vectors, top-K via argpartition.

    Batch search streams through query batches to keep memory bounded.
    """

    def __init__(self, ids: np.ndarray, vectors: np.ndarray):
        if len(ids) != vectors.shape[0]:
            raise ValueError(
                f"ids/vectors mismatch: {len(ids)} ids vs {vectors.shape[0]} vectors"
            )
        self._ids = np.asarray(ids).astype(str)
        # Pre-normalize for cosine via dot-product.
        v = vectors.astype(np.float32)
        norms = np.linalg.norm(v, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        self._mat = (v / norms).astype(np.float32)

    def __len__(self) -> int:
        return self._mat.shape[0]

    def search(self, query: np.ndarray, k: int = 20) -> list[SearchHit]:
        if len(self) == 0:
            return []
        q = query.astype(np.float32)
        n = np.linalg.norm(q)
        if n == 0:
            return []
        q = q / n
        sims = self._mat @ q
        return self._topk_to_hits(sims, k)

    def search_batch(
        self, queries: np.ndarray, k: int = 20, batch_size: int = 1000
    ) -> list[list[SearchHit]]:
        if len(self) == 0 or queries.size == 0:
            return [[] for _ in range(queries.shape[0] if queries.ndim == 2 else 0)]
        results: list[list[SearchHit]] = []
        for start in range(0, queries.shape[0], batch_size):
            end = min(start + batch_size, queries.shape[0])
            q = queries[start:end].astype(np.float32)
            norms = np.linalg.norm(q, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            q = q / norms
            sims = q @ self._mat.T  # (batch, N)
            for row in sims:
                results.append(self._topk_to_hits(row, k))
        return results

    def _topk_to_hits(self, sims: np.ndarray, k: int) -> list[SearchHit]:
        if k >= sims.size:
            order = np.argsort(-sims)
        else:
            top = np.argpartition(-sims, kth=k - 1)[:k]
            order = top[np.argsort(-sims[top])]
        return [
            SearchHit(item_id=str(self._ids[int(i)]), score=float(sims[int(i)]))
            for i in order[:k]
        ]
