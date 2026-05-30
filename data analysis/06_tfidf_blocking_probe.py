"""Test TF-IDF top-K retrieval as a brand-free blocking strategy.

The prior agent showed strict brand+size blocking only covers 4.2% of A.
This script measures whether sparse TF-IDF retrieval over a richer text
field (name + brand + slug + category) can give us O(10) plausible B
candidates for *each* A row, including the 45% with no brand at all.

Outputs:
- outputs/tfidf_blocking_probe.md
- outputs/tfidf_blocking_probe.json
- outputs/tfidf_retrieval_examples.csv
"""
from __future__ import annotations

import json
import time
from typing import Any
from urllib.parse import urlparse

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

from eda_utils import (
    OUTPUT_DIR,
    add_matching_features,
    ensure_output_dir,
    is_missing,
    load_dataset,
    normalize_text,
)


SUMMARY_MD = OUTPUT_DIR / "tfidf_blocking_probe.md"
SUMMARY_JSON = OUTPUT_DIR / "tfidf_blocking_probe.json"
EXAMPLES_CSV = OUTPUT_DIR / "tfidf_retrieval_examples.csv"


def slug(url: Any) -> str:
    if is_missing(url):
        return ""
    try:
        parts = [seg for seg in urlparse(str(url)).path.split("/") if seg]
    except Exception:
        return ""
    cleaned = []
    for seg in parts:
        if seg in {"ip", "shop", "product"}:
            continue
        if seg.isdigit():
            continue
        # strip numeric prefix like 940814-Strawberry-...
        if "-" in seg and seg.split("-", 1)[0].isdigit():
            seg = seg.split("-", 1)[1]
        cleaned.append(seg.replace("-", " "))
    return " ".join(cleaned)


def build_match_text(df: pd.DataFrame) -> pd.Series:
    name = df.get("name", pd.Series("", index=df.index)).fillna("").astype(str)
    brand = df.get("brand_raw", pd.Series("", index=df.index)).fillna("").astype(str)
    cat = (
        df.get("info_category_1", pd.Series("", index=df.index)).fillna("").astype(str)
        + " "
        + df.get("info_category_2", pd.Series("", index=df.index)).fillna("").astype(str)
        + " "
        + df.get("info_category_3", pd.Series("", index=df.index)).fillna("").astype(str)
    )
    url_slug = df.get("url", pd.Series("", index=df.index)).map(slug)
    combined = name + " " + brand + " " + url_slug + " " + cat
    return combined.map(normalize_text)


def top_k_indices_sparse(
    queries: csr_matrix, corpus: csr_matrix, k: int, batch: int = 256
) -> tuple[np.ndarray, np.ndarray]:
    """Return (idx, score) arrays of shape (queries, k) using batched sparse dot products."""
    n = queries.shape[0]
    idx_out = np.full((n, k), -1, dtype=np.int32)
    score_out = np.zeros((n, k), dtype=np.float32)
    corpus_t = corpus.T.tocsr()
    for start in range(0, n, batch):
        stop = min(start + batch, n)
        sims = (queries[start:stop] @ corpus_t).toarray()  # dense (batch, |corpus|)
        if sims.shape[1] <= k:
            order = np.argsort(-sims, axis=1)
        else:
            top = np.argpartition(-sims, k - 1, axis=1)[:, :k]
            # Sort each row's top-k by score desc
            row_scores = np.take_along_axis(sims, top, axis=1)
            order_within = np.argsort(-row_scores, axis=1)
            order = np.take_along_axis(top, order_within, axis=1)
        idx_out[start:stop, : order.shape[1]] = order[:, :k]
        score_out[start:stop, : order.shape[1]] = np.take_along_axis(sims, order[:, :k], axis=1)
    return idx_out, score_out


def main() -> None:
    ensure_output_dir()
    print("Loading datasets")
    a = add_matching_features("A", load_dataset("A")).reset_index(drop=True)
    b = add_matching_features("B", load_dataset("B")).reset_index(drop=True)

    print("Building match text")
    text_a = build_match_text(a)
    text_b = build_match_text(b)

    # Drop rows whose text is empty - cannot vectorize
    mask_a = text_a.str.strip() != ""
    mask_b = text_b.str.strip() != ""
    a_rows = a.loc[mask_a].reset_index(drop=True)
    b_rows = b.loc[mask_b].reset_index(drop=True)
    text_a = text_a.loc[mask_a].reset_index(drop=True)
    text_b = text_b.loc[mask_b].reset_index(drop=True)

    print(f"A vectorizable rows: {len(a_rows):,} / {len(a):,}")
    print(f"B vectorizable rows: {len(b_rows):,} / {len(b):,}")

    print("Fitting TF-IDF (word 1-2gram and char 3-5gram)")
    t0 = time.time()
    word_vec = TfidfVectorizer(
        analyzer="word", ngram_range=(1, 2), min_df=2, max_df=0.5, sublinear_tf=True
    )
    char_vec = TfidfVectorizer(
        analyzer="char_wb", ngram_range=(3, 5), min_df=2, max_df=0.5, sublinear_tf=True
    )
    word_vec.fit(pd.concat([text_a, text_b], ignore_index=True))
    char_vec.fit(pd.concat([text_a, text_b], ignore_index=True))
    fit_time = time.time() - t0

    print("Transforming")
    t0 = time.time()
    Xa_word = word_vec.transform(text_a)
    Xb_word = word_vec.transform(text_b)
    Xa_char = char_vec.transform(text_a)
    Xb_char = char_vec.transform(text_b)
    Xa = normalize(hstack([Xa_word, Xa_char]).tocsr())
    Xb = normalize(hstack([Xb_word, Xb_char]).tocsr())
    transform_time = time.time() - t0
    print(f"  feature dim: {Xa.shape[1]:,}")

    # Sample A rows for the timing/quality probe so this runs in minutes not hours
    sample_n = min(2000, len(a_rows))
    rng = np.random.default_rng(42)
    sample_idx = rng.choice(len(a_rows), size=sample_n, replace=False)
    sample_idx.sort()
    Xa_sample = Xa[sample_idx]

    print(f"Retrieving top-10 for {sample_n:,} A rows")
    t0 = time.time()
    top_idx, top_score = top_k_indices_sparse(Xa_sample, Xb, k=10, batch=256)
    retrieve_time = time.time() - t0

    # Quality signals:
    # 1) brand_match@k: any top-K B share the A row's brand_norm
    # 2) cat2_match@k: any top-K B share the A row's info_category_2_norm
    # 3) size_match@k: any top-K B share size_bucket
    # 4) any_strong@k: cosine > 0.3
    a_sample_df = a_rows.iloc[sample_idx].reset_index(drop=True)
    quality_rows = []
    for i in range(sample_n):
        a_row = a_sample_df.iloc[i]
        b_cands = b_rows.iloc[top_idx[i]]
        brand_match = int(
            (b_cands["brand_norm"] == a_row["brand_norm"]).any()
            and bool(a_row["brand_norm"])
        )
        cat_match = int(
            (b_cands["info_category_2_norm"] == a_row["info_category_2_norm"]).any()
            and bool(a_row.get("info_category_2_norm"))
        )
        size_match = int(
            (b_cands["size_bucket"] == a_row["size_bucket"]).any()
            and bool(a_row["size_bucket"])
        )
        strong = int((top_score[i] >= 0.3).any())
        quality_rows.append(
            {
                "brand_match_at_10": brand_match,
                "cat2_match_at_10": cat_match,
                "size_match_at_10": size_match,
                "strong_cosine_at_10": strong,
                "top1_score": float(top_score[i, 0]),
            }
        )
    q = pd.DataFrame(quality_rows)

    # Examples - first 50 with strong top-1
    example_rows = []
    for i in range(sample_n):
        if top_score[i, 0] < 0.4:
            continue
        a_row = a_sample_df.iloc[i]
        for rank, (bidx, score) in enumerate(zip(top_idx[i, :3], top_score[i, :3])):
            if bidx < 0:
                continue
            b_row = b_rows.iloc[bidx]
            example_rows.append(
                {
                    "item_id_A": a_row["item_id"],
                    "name_A": a_row["name"],
                    "brand_A": a_row.get("brand_raw"),
                    "cat_A": a_row.get("info_category_2"),
                    "rank": rank + 1,
                    "score": round(float(score), 4),
                    "item_id_B": b_row["item_id"],
                    "name_B": b_row["name"],
                    "brand_B": b_row.get("brand_raw"),
                    "cat_B": b_row.get("info_category_2"),
                }
            )
        if len(example_rows) >= 150:
            break
    pd.DataFrame(example_rows).to_csv(EXAMPLES_CSV, index=False)

    summary = {
        "vectorizer": {
            "word_ngram": [1, 2],
            "char_ngram": [3, 5],
            "fit_seconds": round(fit_time, 2),
            "transform_seconds": round(transform_time, 2),
            "feature_dim": int(Xa.shape[1]),
        },
        "retrieval": {
            "sample_a_rows": int(sample_n),
            "k": 10,
            "retrieve_seconds": round(retrieve_time, 2),
            "throughput_a_per_sec": round(sample_n / retrieve_time, 1),
            "estimated_full_run_minutes": round(
                (len(a_rows) / sample_n) * retrieve_time / 60.0, 1
            ),
        },
        "quality_at_10": {
            "brand_match_share": round(float(q["brand_match_at_10"].mean()), 4),
            "cat2_match_share": round(float(q["cat2_match_at_10"].mean()), 4),
            "size_match_share": round(float(q["size_match_at_10"].mean()), 4),
            "strong_cosine_share": round(float(q["strong_cosine_at_10"].mean()), 4),
            "top1_score_p50": round(float(q["top1_score"].quantile(0.5)), 4),
            "top1_score_p90": round(float(q["top1_score"].quantile(0.9)), 4),
        },
    }
    SUMMARY_JSON.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    md = ["# TF-IDF Blocking Probe", ""]
    md.append("## Vectorizer")
    md.append(f"- Feature dim (word + char): {summary['vectorizer']['feature_dim']:,}")
    md.append(f"- Fit time: {summary['vectorizer']['fit_seconds']}s")
    md.append(f"- Transform time: {summary['vectorizer']['transform_seconds']}s")
    md.extend(["", "## Top-10 retrieval"])
    md.append(f"- Sample A rows: {summary['retrieval']['sample_a_rows']:,}")
    md.append(f"- Retrieval time: {summary['retrieval']['retrieve_seconds']}s "
              f"({summary['retrieval']['throughput_a_per_sec']} rows/s)")
    md.append(f"- Estimated full A run: ~{summary['retrieval']['estimated_full_run_minutes']} min")
    md.extend(["", "## Quality of top-10 (on sample)"])
    md.append(f"- Brand match in top-10 (A has brand): {summary['quality_at_10']['brand_match_share']:.1%}")
    md.append(f"- Category-2 match in top-10: {summary['quality_at_10']['cat2_match_share']:.1%}")
    md.append(f"- Size-bucket match in top-10: {summary['quality_at_10']['size_match_share']:.1%}")
    md.append(f"- At least one cosine >= 0.3 in top-10: {summary['quality_at_10']['strong_cosine_share']:.1%}")
    md.append(f"- Top-1 cosine p50 / p90: "
              f"{summary['quality_at_10']['top1_score_p50']} / "
              f"{summary['quality_at_10']['top1_score_p90']}")
    SUMMARY_MD.write_text("\n".join(md), encoding="utf-8")
    print(f"Wrote {SUMMARY_JSON}")
    print(f"Wrote {SUMMARY_MD}")
    print(f"Wrote {EXAMPLES_CSV}")


if __name__ == "__main__":
    main()
