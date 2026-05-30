"""T7: semantic retrieval over OpenAI embeddings.

For each A vector, return the top-K B vectors by cosine. Filters out
cosine < `cosine_floor` (default 0.55 — calibrated against TF-IDF's 0.6
since dense distributions sit slightly lower).
"""
from __future__ import annotations

import numpy as np

from ..store import VectorStore
from .base import Candidate, CandidateRetriever


class SemanticRetriever(CandidateRetriever):
    name = "T7"

    def __init__(
        self,
        a_ids: np.ndarray,
        a_vectors: np.ndarray,
        b_store: VectorStore,
        k: int = 20,
        cosine_floor: float = 0.55,
        batch_size: int = 1000,
    ):
        if len(a_ids) != a_vectors.shape[0]:
            raise ValueError("A ids/vectors mismatch")
        self._a_ids = np.asarray(a_ids).astype(str)
        self._a_vecs = a_vectors.astype(np.float32)
        self._b_store = b_store
        self._k = k
        self._cosine_floor = cosine_floor
        self._batch_size = batch_size

    def retrieve_all(self) -> list[Candidate]:
        out: list[Candidate] = []
        results = self._b_store.search_batch(self._a_vecs, k=self._k, batch_size=self._batch_size)
        for a_id, hits in zip(self._a_ids, results):
            for h in hits:
                if h.score < self._cosine_floor:
                    break  # search returns sorted descending
                out.append(
                    Candidate(
                        item_id_a=str(a_id),
                        item_id_b=h.item_id,
                        source=self.name,
                        score=h.score,
                        features={"semantic_cosine": h.score},
                    )
                )
        return out
