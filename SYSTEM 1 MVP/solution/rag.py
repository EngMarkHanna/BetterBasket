"""Tiny RAG layer for the static judge: retrieve top-K supporting rules
or labeled-pair examples for a given candidate pair.

No vector store. TF-IDF cosine over a small (~200 entry) JSONL/CSV
corpus. System 2 will swap this with OpenAI embeddings + a real KB.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer


# Hand-written rule clauses derived from PLAN.md's "Revised LLM Rubric".
# These are the highest-leverage rules; we re-rank them per pair so the
# model sees the most relevant ones.
DEFAULT_RULES: list[dict] = [
    {
        "id": "rule_multipack",
        "title": "Multipack vs single SKU",
        "content": (
            "When the per-unit SKU is the same (same brand, same product, same per-unit size), "
            "a multipack on one side can be an exact_national_brand match against a single "
            "on the other side. Different pack counts are NOT a match-breaker."
        ),
    },
    {
        "id": "rule_word_order",
        "title": "Word order and punctuation drift",
        "content": (
            "Re-ordered words or differences in punctuation, capitalization, or marketing "
            "phrasing are NOT match-breakers. 'Folgers Black Silk Ground Coffee Dark Roast' "
            "and 'Folgers Coffee Ground Dark Black Silk' are the same SKU."
        ),
    },
    {
        "id": "rule_marketing_copy",
        "title": "Marketing copy drift",
        "content": (
            "Different marketing taglines on the same SKU ('Daily Coconut Hydrate' vs "
            "'for Dry Skin') do not break a match when brand, size, form, and product "
            "family all agree."
        ),
    },
    {
        "id": "rule_organic_food",
        "title": "Organic vs conventional for food",
        "content": (
            "For food, dairy, produce, baby food, and ingredients, organic vs conventional "
            "IS a meaningful mismatch and should be no_match."
        ),
    },
    {
        "id": "rule_flavor",
        "title": "Flavor / scent / shade / formulation",
        "content": (
            "Different flavors, scents, shades, or formulations are no_match unless "
            "explicitly equivalent."
        ),
    },
    {
        "id": "rule_private_label",
        "title": "Private label equivalence",
        "content": (
            "Two private-label products (Great Value, Marketside, Wegmans, Equate, etc.) "
            "of the same specific product, same size, same form qualify as "
            "private_label_equivalent. Different flavors or different products do NOT qualify."
        ),
    },
    {
        "id": "rule_size_tolerance",
        "title": "Manufacturer size revision tolerance",
        "content": (
            "Per-unit size drift up to ~15% can be acceptable for the same SKU when all "
            "other signals (brand, name, form, category) align."
        ),
    },
    {
        "id": "rule_size_dim",
        "title": "Different size dimensions",
        "content": (
            "If the size dimensions differ (one is weight, the other is volume), they are "
            "not directly comparable and should be no_match unless the product family "
            "explicitly comes in both forms."
        ),
    },
    {
        "id": "rule_form_conflict",
        "title": "Product form conflict",
        "content": (
            "Liquid vs powder, ground vs whole bean, K-cups vs ground coffee, lotion vs "
            "spray are no_match. Different forms of the same brand are different SKUs."
        ),
    },
    {
        "id": "rule_strict_default",
        "title": "Strict default",
        "content": (
            "When uncertain, prefer no_match. But do not reject matches over trivial "
            "naming, formatting, or marketing differences if the underlying SKU is the same."
        ),
    },
]


def load_examples(eval_results_csv: Path, max_each: int = 25) -> list[dict]:
    """Load accepted / rejected labeled pairs as examples."""
    if not eval_results_csv.exists():
        return []
    df = pd.read_csv(eval_results_csv)
    if "label_is_match" not in df.columns:
        return []
    out: list[dict] = []
    pos = df[df["label_is_match"] == True].head(max_each)
    neg = df[df["label_is_match"] == False].head(max_each)
    for r in pos.itertuples(index=False):
        out.append(
            {
                "id": f"acc_{getattr(r, 'pair_id', '')}",
                "title": "ACCEPTED example",
                "content": (
                    f"A: {getattr(r, 'name_A', '')} / brand {getattr(r, 'brand_A', '')} / size {getattr(r, 'size_A', '')} | "
                    f"B: {getattr(r, 'name_B', '')} / brand {getattr(r, 'brand_B', '')} / size {getattr(r, 'size_B', '')}. "
                    f"Reason: {getattr(r, 'label_notes', '') or ''}"
                ),
            }
        )
    for r in neg.itertuples(index=False):
        out.append(
            {
                "id": f"rej_{getattr(r, 'pair_id', '')}",
                "title": "REJECTED example",
                "content": (
                    f"A: {getattr(r, 'name_A', '')} / brand {getattr(r, 'brand_A', '')} / size {getattr(r, 'size_A', '')} | "
                    f"B: {getattr(r, 'name_B', '')} / brand {getattr(r, 'brand_B', '')} / size {getattr(r, 'size_B', '')}. "
                    f"Reason: {getattr(r, 'label_notes', '') or ''}"
                ),
            }
        )
    return out


class RAGStore:
    """TF-IDF index over (rules + examples). Tiny corpus, instant search."""

    def __init__(self, entries: list[dict]):
        self.entries = entries
        if not entries:
            self.vec = None
            self.mat = None
            return
        self.vec = TfidfVectorizer(analyzer="word", ngram_range=(1, 2), sublinear_tf=True)
        docs = [f'{e["title"]}. {e["content"]}' for e in entries]
        self.mat = self.vec.fit_transform(docs)

    def search(self, query: str, k: int = 3) -> list[dict]:
        if not self.entries or self.vec is None:
            return []
        q = self.vec.transform([query])
        sims = (q @ self.mat.T).toarray()[0]
        idx = sims.argsort()[::-1][:k]
        return [
            {**self.entries[i], "score": float(sims[i])}
            for i in idx
            if sims[i] > 0
        ]


def build_store(eval_results_csv: Path | None = None, extra_rules: list[dict] | None = None) -> RAGStore:
    entries = list(DEFAULT_RULES)
    if extra_rules:
        entries.extend(extra_rules)
    if eval_results_csv is not None:
        entries.extend(load_examples(eval_results_csv))
    return RAGStore(entries)
