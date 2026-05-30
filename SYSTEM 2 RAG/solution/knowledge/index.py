"""Knowledge retriever: embeds the knowledge corpus once, then answers
per-pair queries by cosine similarity + structured filters.

The retriever is the only consumer of knowledge-base embeddings. It
piggy-backs on the same EmbeddingBank used for the catalog, so re-running
after a bootstrap re-embeds only changed entries.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..embed import EmbeddingBank, OpenAIEmbedder, text_hash
from .entry import EntryType, KnowledgeEntry, read_all


@dataclass(frozen=True)
class RetrievedEntry:
    """A knowledge entry plus its retrieval score."""

    entry: KnowledgeEntry
    score: float


class KnowledgeRetriever:
    """In-memory cosine search over embedded knowledge entries.

    Optimized for tiny corpora (a few hundred to a few thousand entries).
    For larger corpora we'd swap in a sharded vector store - the
    `search` API stays the same.
    """

    def __init__(
        self,
        entries: list[KnowledgeEntry],
        vectors: np.ndarray,
    ):
        if len(entries) != vectors.shape[0]:
            raise ValueError(
                f"entries/vectors mismatch: {len(entries)} entries vs "
                f"{vectors.shape[0]} vectors"
            )
        self._entries = entries
        # Already L2-normalized? Don't assume. Normalize for cosine.
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        self._vectors = (vectors / norms).astype(np.float32)
        self._by_id = {e.id: i for i, e in enumerate(self._entries)}

    def __len__(self) -> int:
        return len(self._entries)

    @property
    def entries(self) -> list[KnowledgeEntry]:
        return self._entries

    def search(
        self,
        query_vector: np.ndarray,
        k: int = 5,
        entry_type: EntryType | None = None,
        match_type: str | None = None,
        min_score: float = 0.0,
    ) -> list[RetrievedEntry]:
        """Return the top-k most similar entries to a query vector.

        Filters applied BEFORE k-truncation so we always return up to k
        eligible entries.
        """
        if len(self._entries) == 0:
            return []
        q = query_vector.astype(np.float32)
        q_norm = np.linalg.norm(q)
        if q_norm == 0:
            return []
        q = q / q_norm
        sims = self._vectors @ q  # (N,)

        # Build the eligible-index mask up front.
        eligible = np.ones(len(self._entries), dtype=bool)
        if entry_type is not None:
            for i, e in enumerate(self._entries):
                if eligible[i] and e.type != entry_type:
                    eligible[i] = False
        if match_type is not None:
            for i, e in enumerate(self._entries):
                if eligible[i] and e.match_type is not None and e.match_type != match_type:
                    eligible[i] = False

        # Mask out by setting to -inf so argsort drops them.
        masked = np.where(eligible, sims, -np.inf)
        # Top-k indices.
        if k >= masked.size:
            order = np.argsort(-masked)
        else:
            top = np.argpartition(-masked, kth=k - 1)[:k]
            order = top[np.argsort(-masked[top])]
        results: list[RetrievedEntry] = []
        for i in order:
            score = float(sims[int(i)])
            if score < min_score or not np.isfinite(masked[int(i)]):
                continue
            results.append(RetrievedEntry(entry=self._entries[int(i)], score=score))
            if len(results) >= k:
                break
        return results

    @classmethod
    def build(
        cls,
        knowledge_dir: Path,
        embedder: OpenAIEmbedder,
        bank: EmbeddingBank,
    ) -> "KnowledgeRetriever":
        """Load all JSONL entries from `knowledge_dir`, embed any missing
        ones, return a populated retriever.
        """
        entries = read_all(knowledge_dir)
        if not entries:
            return cls(entries=[], vectors=np.zeros((0, embedder.dim), dtype=np.float32))

        # Compute hashes; embed misses.
        texts = [e.text_for_embedding() for e in entries]
        hashes = [
            text_hash(t, model=embedder.model, dim=embedder.dim) for t in texts
        ]
        to_embed = [(i, t) for i, (t, h) in enumerate(zip(texts, hashes)) if not bank.has(h)]
        if to_embed:
            print(f"  [knowledge] embedding {len(to_embed)} new entries ...")
            t0 = time.time()
            B = embedder.batch_size
            for start in range(0, len(to_embed), B):
                batch = to_embed[start : start + B]
                batch_texts = [t for _, t in batch]
                vecs = embedder.embed_batch(batch_texts)
                for (i, _), v in zip(batch, vecs):
                    bank.put(hashes[i], v)
            bank.save()
            print(f"  [knowledge] embedded in {time.time()-t0:.1f}s")

        # Materialize vectors in entry order.
        mat = np.zeros((len(entries), embedder.dim), dtype=np.float32)
        for i, h in enumerate(hashes):
            v = bank.get(h)
            if v is None:
                raise RuntimeError(f"knowledge entry {entries[i].id} missing vector")
            mat[i] = v
        return cls(entries=entries, vectors=mat)
