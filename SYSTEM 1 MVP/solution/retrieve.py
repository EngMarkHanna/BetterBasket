"""Candidate generation. Three sources, unioned and deduped:

  T1 = strict brand_canonical + size block, scored by RapidFuzz.WRatio
  T3 = TF-IDF top-K cosine over the joint text vocabulary
  T5 = private-label, category-bridge + size compatibility + name overlap

Each candidate keeps `candidate_source` so we can audit it later.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from rapidfuzz import fuzz, process
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize


@dataclass
class Candidate:
    """Single A->B candidate with provenance.

    `features` is a free-form bag of pre-computed signals each retriever
    chose to expose; the scorer reads them downstream.
    """

    item_id_a: str
    item_id_b: str
    source: str  # 'T1' | 'T3' | 'T5'
    score: float  # source-specific raw score (cosine, rapidfuzz, etc.)
    features: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# T1 - strict brand + size blocks
# ---------------------------------------------------------------------------


def t1_strict_blocks(a: pd.DataFrame, b: pd.DataFrame, top_per_a: int = 3) -> list[Candidate]:
    """For each A row with brand_canonical + (unit_bucket OR total_bucket)
    populated, look up B rows sharing the same key and keep the top-N by
    RapidFuzz WRatio on name_norm.

    We use BOTH unit and total bucket as block keys so that a multipack
    on A can match a single-unit B at the per-unit level.
    """
    # Build B side blocks keyed on (brand_canonical, bucket) for both
    # unit and total buckets.
    blocks: dict[tuple[str, str], list[tuple[str, str]]] = defaultdict(list)
    for row in b.itertuples(index=False):
        brand = row.brand_canonical
        if not brand:
            continue
        for bucket in (row.unit_bucket, row.total_bucket):
            if bucket:
                blocks[(brand, bucket)].append((row.item_id, row.name_norm))

    candidates: list[Candidate] = []
    n_total = 0
    for row in a.itertuples(index=False):
        brand = row.brand_canonical
        if not brand:
            continue
        seen_b: set[str] = set()
        choices: list[tuple[str, str]] = []
        for bucket in (row.unit_bucket, row.total_bucket):
            if not bucket:
                continue
            for bid, bname in blocks.get((brand, bucket), ()):
                if bid not in seen_b:
                    seen_b.add(bid)
                    choices.append((bid, bname))
        if not choices:
            continue

        # Score all B names; take top N.
        names = [c[1] for c in choices]
        results = process.extract(row.name_norm, names, scorer=fuzz.WRatio, limit=top_per_a)
        for _, score, idx in results:
            bid, _ = choices[idx]
            candidates.append(
                Candidate(
                    item_id_a=row.item_id,
                    item_id_b=bid,
                    source="T1",
                    score=float(score),
                    features={
                        "rapidfuzz_wratio_name": float(score),
                        "block_brand": brand,
                    },
                )
            )
            n_total += 1
    return candidates


# ---------------------------------------------------------------------------
# T3 - TF-IDF top-K
# ---------------------------------------------------------------------------


def build_tfidf(
    a_texts: list[str], b_texts: list[str]
) -> tuple[TfidfVectorizer, TfidfVectorizer, csr_matrix, csr_matrix]:
    """Fit two analyzers (word 1-2gram, char_wb 3-5gram) on A+B, return
    the L2-normalized matrices for A and B.
    """
    corpus = a_texts + b_texts
    word_vec = TfidfVectorizer(
        analyzer="word",
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.6,
        sublinear_tf=True,
        norm=None,
    )
    char_vec = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 5),
        min_df=2,
        max_df=0.6,
        sublinear_tf=True,
        norm=None,
    )
    word_vec.fit(corpus)
    char_vec.fit(corpus)

    def transform(texts: list[str]) -> csr_matrix:
        joint = hstack([word_vec.transform(texts), char_vec.transform(texts)]).tocsr()
        return normalize(joint, norm="l2", copy=False)

    a_mat = transform(a_texts)
    b_mat = transform(b_texts)
    return word_vec, char_vec, a_mat, b_mat


def t3_tfidf_topk(
    a: pd.DataFrame,
    b: pd.DataFrame,
    k: int = 20,
    cosine_floor: float = 0.4,
    batch_size: int = 2000,
) -> list[Candidate]:
    """For each A row, return the top-K B rows by joint TF-IDF cosine,
    cosine >= cosine_floor. Streams in batches to keep memory bounded.
    """
    a_texts = a["retrieval_text"].fillna("").tolist()
    b_texts = b["retrieval_text"].fillna("").tolist()
    _, _, a_mat, b_mat = build_tfidf(a_texts, b_texts)
    b_ids = b["item_id"].tolist()
    a_ids = a["item_id"].tolist()

    candidates: list[Candidate] = []
    b_T = b_mat.T.tocsr()

    for start in range(0, a_mat.shape[0], batch_size):
        end = min(start + batch_size, a_mat.shape[0])
        sims = (a_mat[start:end] @ b_T).toarray()  # (batch, n_b)
        # Top-K per row.
        if sims.shape[1] <= k:
            top_idx = np.argsort(-sims, axis=1)
        else:
            part = np.argpartition(-sims, kth=k - 1, axis=1)[:, :k]
            row_idx = np.arange(part.shape[0])[:, None]
            order = np.argsort(-sims[row_idx, part], axis=1)
            top_idx = part[row_idx, order]
        for r in range(top_idx.shape[0]):
            a_id = a_ids[start + r]
            for c in top_idx[r]:
                cos = float(sims[r, c])
                if cos < cosine_floor:
                    break  # remaining are smaller
                candidates.append(
                    Candidate(
                        item_id_a=a_id,
                        item_id_b=b_ids[int(c)],
                        source="T3",
                        score=cos,
                        features={"tfidf_cosine": cos},
                    )
                )
    return candidates


# ---------------------------------------------------------------------------
# T5 - private-label / fresh via category bridge
# ---------------------------------------------------------------------------


def _load_category_bridge(bridge_csv: Path) -> dict[str, list[tuple[str, float, int]]]:
    """{a_category_path -> [(b_category, share, support), ...]} from the
    learned bridge. We only use a_level/b_level rows we trust the most.
    """
    if not bridge_csv.exists():
        return {}
    df = pd.read_csv(bridge_csv)
    # Restrict to deepest level pairings where it exists, fallback to cat2.
    df = df[(df["a_level"].isin({"a_cat2", "a_cat3"})) | (df["a_level"] == "a_cat2")]
    df = df[df["support"] >= 10]
    out: dict[str, list[tuple[str, float, int]]] = defaultdict(list)
    for row in df.itertuples(index=False):
        cat = str(row.a_category).strip().lower()
        for col_cat, col_share in (
            ("b_top1", "b_top1_share"),
            ("b_top2", "b_top2_share"),
            ("b_top3", "b_top3_share"),
        ):
            b_cat = getattr(row, col_cat)
            b_share = getattr(row, col_share)
            if pd.notna(b_cat) and pd.notna(b_share) and float(b_share) >= 0.10:
                out[cat].append((str(b_cat).strip().lower(), float(b_share), int(row.support)))
    return out


def t5_private_label(
    a: pd.DataFrame,
    b: pd.DataFrame,
    bridge_csv: Path,
    top_per_a: int = 5,
    name_min_score: float = 60.0,
) -> list[Candidate]:
    """Private-label A products -> private-label B products with:
       - compatible size (unit OR total bucket equal)
       - bridge-mapped category alignment
       - RapidFuzz WRatio >= name_min_score on name_norm
    """
    bridge = _load_category_bridge(bridge_csv)
    a_pl = a[a["is_private_label_inferred"]].copy()
    b_pl = b[b["is_private_label_inferred"]].copy()
    if a_pl.empty or b_pl.empty:
        return []

    # Index B by (bucket) -> list[(item_id, name_norm, b_categories)].
    # Audit fix #14: also index by info_category_3 so cat3 bridge entries
    # can match.
    b_by_bucket: dict[str, list[tuple[str, str, set[str]]]] = defaultdict(list)
    for row in b_pl.itertuples(index=False):
        b_cats = {
            normalize_str(row.info_category_0),
            normalize_str(row.info_category_1),
            normalize_str(row.info_category_2),
            normalize_str(getattr(row, "info_category_3", "")),
        }
        b_cats.discard("")
        for bucket in (row.unit_bucket, row.total_bucket):
            if bucket:
                b_by_bucket[bucket].append((row.item_id, row.name_norm, b_cats))

    candidates: list[Candidate] = []
    for row in a_pl.itertuples(index=False):
        # Build the set of plausible B categories from the A row's
        # category fields run through the bridge. Audit fix #14: walk
        # all levels including cat3.
        plausible_b_cats: set[str] = set()
        a_cat_fields = (
            getattr(row, "info_category_3", None),
            row.info_category_2,
            row.info_category_1,
            row.info_category_0,
        )
        for a_cat_field in a_cat_fields:
            if a_cat_field and not is_missing_str(a_cat_field):
                for b_cat, _, _ in bridge.get(str(a_cat_field).strip().lower(), ()):
                    plausible_b_cats.add(b_cat)
        if not plausible_b_cats:
            continue

        choices: list[tuple[str, str]] = []
        seen: set[str] = set()
        for bucket in (row.unit_bucket, row.total_bucket):
            if not bucket:
                continue
            for b_id, b_name, b_cats in b_by_bucket.get(bucket, ()):
                if b_id in seen:
                    continue
                if not (b_cats & plausible_b_cats):
                    continue
                seen.add(b_id)
                choices.append((b_id, b_name))

        if not choices:
            continue
        names = [c[1] for c in choices]
        results = process.extract(row.name_norm, names, scorer=fuzz.WRatio, limit=top_per_a)
        for _, score, idx in results:
            if score < name_min_score:
                continue
            b_id, _ = choices[idx]
            candidates.append(
                Candidate(
                    item_id_a=row.item_id,
                    item_id_b=b_id,
                    source="T5",
                    score=float(score),
                    features={
                        "rapidfuzz_wratio_name": float(score),
                        "private_label_both": True,
                    },
                )
            )
    return candidates


def normalize_str(value) -> str:
    if value is None or (isinstance(value, float) and value != value):
        return ""
    return str(value).strip().lower()


def is_missing_str(value) -> bool:
    return normalize_str(value) in {"", "nan", "none", "null"}


# ---------------------------------------------------------------------------
# Union + dedupe
# ---------------------------------------------------------------------------


def union_candidates(*lists: Iterable[Candidate]) -> list[Candidate]:
    """Union three candidate sources by (item_id_a, item_id_b), merging
    source labels and keeping max score per source."""
    merged: dict[tuple[str, str], Candidate] = {}
    for lst in lists:
        for cand in lst:
            key = (cand.item_id_a, cand.item_id_b)
            if key not in merged:
                merged[key] = Candidate(
                    item_id_a=cand.item_id_a,
                    item_id_b=cand.item_id_b,
                    source=cand.source,
                    score=cand.score,
                    features=dict(cand.features),
                )
            else:
                existing = merged[key]
                if cand.source not in existing.source.split("+"):
                    existing.source = "+".join(sorted(set(existing.source.split("+") + [cand.source])))
                existing.features.update(cand.features)
                # keep the larger normalized score (just for sort)
                existing.score = max(existing.score, cand.score)
    return list(merged.values())


def to_dataframe(candidates: list[Candidate]) -> pd.DataFrame:
    rows = []
    for c in candidates:
        row = {
            "item_id_a": c.item_id_a,
            "item_id_b": c.item_id_b,
            "candidate_source": c.source,
            "candidate_score": c.score,
        }
        row.update(c.features)
        rows.append(row)
    return pd.DataFrame(rows)
