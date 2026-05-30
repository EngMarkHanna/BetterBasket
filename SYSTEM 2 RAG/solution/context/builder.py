"""Build a RAG context bundle for one A row + its top-K B candidates.

We pre-fetch *deterministically* in Phase C - no tool-call loop. The
model receives the bundle in the prompt and decides. This isolates the
lift from "context injection" vs adding agentic tool calls.

A context bundle contains five slots:
  - rules         : top-K rule entries by semantic similarity
  - aliases       : brand-alias decisions for each (A.brand, B.brand)
  - bridge        : A->B category bridge for A.category
  - accepted_ex   : top-K accepted examples by semantic similarity
  - rejected_ex   : top-K rejected examples by semantic similarity
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..embed import OpenAIEmbedder
from ..knowledge import EntryType, KnowledgeRetriever, RetrievedEntry


# Curated brand alias map - duplicated from System 1 so we are explicit
# about what System 2 considers an alias. Keep in sync intentionally.
BRAND_ALIASES: dict[str, str] = {
    "e l f cosmetics": "e l f",
    "l oreal paris": "l oreal",
    "amy s kitchen": "amy s",
    "nature s path organic": "nature s path",
    "bigelow tea": "bigelow",
    "so delicious dairy free": "so delicious",
    "bush s best": "bush s",
    "stonyfield organic": "stonyfield",
    "nestl toll house": "toll house",
    "lindt lindor": "lindt",
    "hero cosmetics": "hero",
    "u by kotex": "kotex",
    "mars wrigley": "mars",
    "rachael ray nutrish": "nutrish",
    "clif bar": "clif",
    "coors light": "coors",
    "voortman bakery": "voortman",
    "mt olive pickle": "mt olive",
    "suave essentials": "suave",
    "c4 energy": "c4",
}


@dataclass
class RAGContext:
    """The evidence packet for one judgment."""

    rules: list[RetrievedEntry] = field(default_factory=list)
    aliases: list[dict[str, Any]] = field(default_factory=list)
    bridge_hint: str | None = None
    accepted_examples: list[RetrievedEntry] = field(default_factory=list)
    rejected_examples: list[RetrievedEntry] = field(default_factory=list)

    def as_prompt_block(self) -> str:
        """Format the bundle as a single human-readable text block."""
        lines: list[str] = []
        if self.rules:
            lines.append("Relevant rules:")
            for r in self.rules:
                lines.append(f"- ({r.entry.title}) {r.entry.content}")
        if self.aliases:
            lines.append("Brand alias checks:")
            for a in self.aliases:
                lines.append(
                    f"- B[{a['b_index']}]: A.brand={a['a_brand']!r} vs B.brand={a['b_brand']!r}"
                    f" -> are_aliases={a['are_aliases']}"
                )
        if self.bridge_hint:
            lines.append(f"Category bridge: {self.bridge_hint}")
        if self.accepted_examples:
            lines.append("Similar accepted examples:")
            for r in self.accepted_examples:
                lines.append(f"- {r.entry.title}: {r.entry.content[:300]}")
        if self.rejected_examples:
            lines.append("Similar rejected examples:")
            for r in self.rejected_examples:
                lines.append(f"- {r.entry.title}: {r.entry.content[:300]}")
        return "\n".join(lines) if lines else ""

    def cache_signature(self) -> str:
        """Stable hash of the context for inclusion in LLM cache keys.

        Bumping the context format or filling different rules/examples
        invalidates cached judgments cleanly.
        """
        parts: list[str] = []
        parts.extend([f"rule:{r.entry.id}" for r in self.rules])
        parts.extend([f"alias:{a['a_brand']}|{a['b_brand']}|{int(a['are_aliases'])}" for a in self.aliases])
        if self.bridge_hint:
            parts.append(f"bridge:{self.bridge_hint}")
        parts.extend([f"acc:{r.entry.id}" for r in self.accepted_examples])
        parts.extend([f"rej:{r.entry.id}" for r in self.rejected_examples])
        h = hashlib.sha256()
        h.update("|".join(parts).encode("utf-8"))
        return h.hexdigest()[:16]


class RAGContextBuilder:
    """Builds RAGContext bundles for A-row + B-candidates groups.

    The builder owns an embedder (for query embedding) and a
    KnowledgeRetriever (for top-K search). Bridge and alias lookups
    are deterministic Python.
    """

    def __init__(
        self,
        embedder: OpenAIEmbedder,
        knowledge: KnowledgeRetriever,
        category_bridge: dict[str, list[tuple[str, float, int]]] | None = None,
        rules_k: int = 3,
        accepted_k: int = 2,
        rejected_k: int = 2,
    ):
        self._embedder = embedder
        self._knowledge = knowledge
        self._bridge = category_bridge or {}
        self._rules_k = rules_k
        self._accepted_k = accepted_k
        self._rejected_k = rejected_k

    def build(self, a_row: dict, b_rows: list[dict]) -> RAGContext:
        """Assemble the context for one A row and its B candidates.

        Returns an empty RAGContext if no signals fired (still safe to
        pass to the judge).

        Single-pair path: embeds the query for this pair on its own.
        For bulk workloads use `build_many` to amortize embedding HTTP
        round-trips across many pairs.
        """
        query_text = self._build_query_text(a_row, b_rows)
        q_vec = None
        if query_text:
            q_vec = self._embedder.embed_batch([query_text])[0]
        return self._build_with_query_vector(a_row, b_rows, q_vec)

    def build_many(
        self,
        items: list[tuple[dict, list[dict]]],
        embed_batch_size: int = 100,
        progress_every: int = 50,
    ) -> list[RAGContext]:
        """Bulk path: embed all per-pair queries in groups, then assemble.

        The serial single-call pattern in `build` was the silent
        bottleneck of the full pipeline (one embedding HTTP round-trip
        per A-row group). Batching by 100 cuts wall clock by ~Bx where
        B is the per-call overhead vs per-text cost ratio.
        """
        import time

        # 1. Compute query texts for all groups (deterministic, no API).
        query_texts: list[str] = []
        for a_row, b_rows in items:
            query_texts.append(self._build_query_text(a_row, b_rows))

        # 2. Embed in batches. Skip empty queries (those get no q_vec).
        n = len(query_texts)
        vectors: list[np.ndarray | None] = [None] * n
        # Group indices that need embedding.
        to_embed_idx = [i for i, t in enumerate(query_texts) if t]
        t0 = time.time()
        for start in range(0, len(to_embed_idx), embed_batch_size):
            chunk = to_embed_idx[start : start + embed_batch_size]
            chunk_texts = [query_texts[i] for i in chunk]
            vecs = self._embedder.embed_batch(chunk_texts)
            for local, global_i in enumerate(chunk):
                vectors[global_i] = vecs[local]
            done = start + len(chunk)
            if (done // embed_batch_size) % progress_every == 0:
                elapsed = time.time() - t0
                rate = done / elapsed if elapsed > 0 else 0.0
                eta_s = (len(to_embed_idx) - done) / rate if rate > 0 else 0
                print(
                    f"    [context] embedded {done}/{len(to_embed_idx)} "
                    f"queries  {rate:.0f}/s  ETA {eta_s/60:.1f} min"
                )

        # 3. Assemble contexts with their precomputed query vectors.
        return [
            self._build_with_query_vector(a_row, b_rows, vectors[i])
            for i, (a_row, b_rows) in enumerate(items)
        ]

    def _build_with_query_vector(
        self,
        a_row: dict,
        b_rows: list[dict],
        q_vec: np.ndarray | None,
    ) -> RAGContext:
        """Shared assembly used by both `build` and `build_many`."""
        ctx = RAGContext()

        if q_vec is not None:
            ctx.rules = self._knowledge.search(q_vec, k=self._rules_k, entry_type=EntryType.RULE)
            ctx.accepted_examples = self._knowledge.search(
                q_vec, k=self._accepted_k, entry_type=EntryType.ACCEPTED_EXAMPLE
            )
            ctx.rejected_examples = self._knowledge.search(
                q_vec, k=self._rejected_k, entry_type=EntryType.REJECTED_EXAMPLE
            )

        # Brand alias check per candidate.
        a_brand = self._safe(a_row.get("brand_canonical")).lower()
        for i, b in enumerate(b_rows):
            b_brand = self._safe(b.get("brand_canonical")).lower()
            if not a_brand or not b_brand:
                continue
            canonical_a = BRAND_ALIASES.get(a_brand, a_brand)
            canonical_b = BRAND_ALIASES.get(b_brand, b_brand)
            ctx.aliases.append(
                {
                    "b_index": i,
                    "a_brand": a_brand,
                    "b_brand": b_brand,
                    "canonical_a": canonical_a,
                    "canonical_b": canonical_b,
                    "are_aliases": bool(canonical_a) and canonical_a == canonical_b,
                }
            )

        # Category bridge.
        a_cat = self._best_a_category(a_row)
        if a_cat and self._bridge:
            targets = self._bridge.get(a_cat.lower(), [])
            if targets:
                top = targets[:3]
                desc = "; ".join(f"{t[0]} (share={t[1]:.2f})" for t in top)
                ctx.bridge_hint = f"A category {a_cat!r} typically maps to B categories: {desc}"

        return ctx

    @staticmethod
    def _safe(value) -> str:
        if value is None:
            return ""
        try:
            if isinstance(value, float) and value != value:  # NaN
                return ""
        except Exception:
            pass
        s = str(value).strip()
        if s.lower() in {"nan", "none", "null"}:
            return ""
        return s

    @staticmethod
    def _build_query_text(a_row: dict, b_rows: list[dict]) -> str:
        safe = RAGContextBuilder._safe
        a_name = safe(a_row.get("name"))
        a_brand = safe(a_row.get("brand_canonical")) or safe(a_row.get("brand_raw"))
        a_size = safe(a_row.get("size_text"))
        b_descs = []
        for b in b_rows:
            bn = safe(b.get("name"))
            bb = safe(b.get("brand_canonical")) or safe(b.get("brand_raw"))
            bs = safe(b.get("size_text"))
            b_descs.append(f"{bn} ({bb}, {bs})")
        return (
            f"A: {a_name} brand {a_brand} size {a_size}. "
            f"B candidates: {' | '.join(b_descs)}"
        )

    @staticmethod
    def _best_a_category(a_row: dict) -> str | None:
        for k in ("info_category_2", "info_category_1", "info_category_0"):
            v = RAGContextBuilder._safe(a_row.get(k))
            if v:
                return v
        return None
