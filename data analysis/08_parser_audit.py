"""Audit the size parser and brand normalizer for fixable failure modes.

Findings drive parser improvements for the production matching pipeline.

Failure modes inspected:
1. Multipack-prefix patterns like "(3 pack) ..., 15.25 oz" - prior parser
   sets multiplier=1 because the regex requires digit-x-digit adjacency,
   so the bucket reflects per-unit size, not total pack size.
2. Leading-decimal sizes like ".5 L" - parser requires \d+ before the
   decimal, so these are silently dropped.
3. "1 ct" / "1 each" rows that produce a meaningless size_bucket=count:1.
4. Brand-norm near-misses: distinct normalized brand strings that are
   really the same brand (alias gap).

Outputs:
- outputs/parser_audit.md
- outputs/parser_audit.json
- outputs/parser_failure_examples.csv
- outputs/brand_alias_candidates.csv
"""
from __future__ import annotations

import json
import re
from collections import Counter
from typing import Any

import pandas as pd
from rapidfuzz import fuzz, process

from eda_utils import (
    OUTPUT_DIR,
    add_matching_features,
    ensure_output_dir,
    is_missing,
    load_dataset,
    normalize_text,
    parse_size,
)


SUMMARY_MD = OUTPUT_DIR / "parser_audit.md"
SUMMARY_JSON = OUTPUT_DIR / "parser_audit.json"
FAILURES_CSV = OUTPUT_DIR / "parser_failure_examples.csv"
BRAND_ALIASES_CSV = OUTPUT_DIR / "brand_alias_candidates.csv"

MULTIPACK_PREFIX_RE = re.compile(
    r"(?:\(?\s*(\d+)\s*(?:pack|pk|count|ct|x)\s*\)?)|"
    r"(?:(\d+)\s*[- ]?(?:pack|pk)\b)",
    re.I,
)
LEADING_DECIMAL_RE = re.compile(r"(?<!\d)\.\d+\s*(?:l|liter|litre|ml|kg|g|oz)\b", re.I)
TRIVIAL_COUNT_RE = re.compile(r"^\s*1\s*(?:ct|count|each|pc|piece|pk|pack)\s*$", re.I)


def detect_multipack(text: Any) -> int:
    if is_missing(text):
        return 0
    return int(bool(MULTIPACK_PREFIX_RE.search(str(text))))


def detect_leading_decimal(text: Any) -> int:
    if is_missing(text):
        return 0
    return int(bool(LEADING_DECIMAL_RE.search(str(text))))


def detect_trivial_count(size_user_friendly: Any) -> int:
    if is_missing(size_user_friendly):
        return 0
    return int(bool(TRIVIAL_COUNT_RE.match(str(size_user_friendly))))


def main() -> None:
    ensure_output_dir()
    print("Loading datasets")
    a = add_matching_features("A", load_dataset("A"))
    b = add_matching_features("B", load_dataset("B"))

    failure_examples: list[dict[str, Any]] = []
    summary: dict[str, Any] = {"per_store": {}}

    for label, df in [("A", a), ("B", b)]:
        n = len(df)
        # Source string the parser inspects (first non-empty among these)
        size_src = df.get("sizing_size_user_friendly", pd.Series("", index=df.index)).where(
            ~df.get("sizing_size_user_friendly", pd.Series("", index=df.index)).map(is_missing),
            df.get("size_raw", pd.Series("", index=df.index)),
        )
        size_src = size_src.where(~size_src.map(is_missing), df.get("name", pd.Series("", index=df.index)))

        mp_in_name = df.get("name", pd.Series("", index=df.index)).map(detect_multipack)
        mp_in_src = size_src.map(detect_multipack)
        ld_in_src = size_src.map(detect_leading_decimal)

        # Bucket of count:1 is suspicious for products like "Stapler 1 each"
        trivial_ct_bucket = df["size_bucket"].astype(str).str.lower() == "count:1"

        # Multipack without canonical accounting: name has multipack, bucket is plain weight/volume
        bucket_dim = df["size_bucket"].astype(str).str.split(":").str[0]
        bug_multipack = (mp_in_name == 1) & (bucket_dim.isin(["weight", "volume"]))

        bug_leading_decimal_no_parse = (ld_in_src == 1) & (df["size_bucket"] == "")

        summary["per_store"][label] = {
            "rows": int(n),
            "multipack_indicator_in_name": int(mp_in_name.sum()),
            "multipack_indicator_in_size_source": int(mp_in_src.sum()),
            "leading_decimal_in_size_source": int(ld_in_src.sum()),
            "trivial_count1_bucket": int(trivial_ct_bucket.sum()),
            "multipack_unaccounted_in_bucket": int(bug_multipack.sum()),
            "leading_decimal_unparsed_bucket_empty": int(bug_leading_decimal_no_parse.sum()),
            "bucket_empty_rows": int((df["size_bucket"] == "").sum()),
        }

        # Capture some real-world failure examples
        for kind, mask in [
            ("multipack_unaccounted", bug_multipack),
            ("leading_decimal_unparsed", bug_leading_decimal_no_parse),
            ("trivial_count1_bucket", trivial_ct_bucket),
        ]:
            for _, row in df.loc[mask].head(15).iterrows():
                failure_examples.append(
                    {
                        "store": label,
                        "issue": kind,
                        "item_id": row.get("item_id"),
                        "name": row.get("name"),
                        "size_user_friendly": row.get("sizing_size_user_friendly"),
                        "size_raw": row.get("size_raw"),
                        "current_bucket": row.get("size_bucket"),
                    }
                )

    pd.DataFrame(failure_examples).to_csv(FAILURES_CSV, index=False)

    # Brand alias near-matches: pairs of distinct A.brand_norm values that are RapidFuzz-similar
    a_brand_counts = Counter(a.loc[a["brand_norm"] != "", "brand_norm"])
    b_brand_counts = Counter(b.loc[b["brand_norm"] != "", "brand_norm"])
    a_brands = [b_ for b_, c in a_brand_counts.most_common(2000)]
    b_brands = [b_ for b_, c in b_brand_counts.most_common(2000)]

    # For each B brand not exact-shared, look up best A brand
    shared = set(a_brand_counts) & set(b_brand_counts)
    alias_rows: list[dict[str, Any]] = []
    for bb in b_brands:
        if bb in shared:
            continue
        match = process.extractOne(bb, a_brands, scorer=fuzz.token_set_ratio)
        if match is None:
            continue
        a_match, score, _ = match
        if score < 90 or a_match == bb:
            continue
        alias_rows.append(
            {
                "b_brand_norm": bb,
                "b_count": b_brand_counts[bb],
                "a_brand_norm": a_match,
                "a_count": a_brand_counts.get(a_match, 0),
                "score": float(score),
            }
        )
    alias_df = pd.DataFrame(alias_rows).sort_values(["score", "b_count"], ascending=False)
    alias_df.to_csv(BRAND_ALIASES_CSV, index=False)

    summary["brand_alias_candidates"] = {
        "rows": int(len(alias_df)),
        "rows_score_ge_95": int((alias_df["score"] >= 95).sum()) if not alias_df.empty else 0,
    }

    SUMMARY_JSON.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    md = ["# Size Parser & Brand Alias Audit", ""]
    for label, stats in summary["per_store"].items():
        md.append(f"## Store {label}")
        md.append(f"- Rows: {stats['rows']:,}")
        md.append(f"- Multipack indicator in `name`: {stats['multipack_indicator_in_name']:,}")
        md.append(f"- Multipack indicator in size source: {stats['multipack_indicator_in_size_source']:,}")
        md.append(f"- Leading-decimal size in size source: {stats['leading_decimal_in_size_source']:,}")
        md.append(f"- Trivial `count:1` buckets: {stats['trivial_count1_bucket']:,}")
        md.append(f"- BUG: multipack name with bucket missing pack multiplier: "
                  f"{stats['multipack_unaccounted_in_bucket']:,}")
        md.append(f"- BUG: leading-decimal size but bucket empty: "
                  f"{stats['leading_decimal_unparsed_bucket_empty']:,}")
        md.append(f"- Empty bucket rows overall: {stats['bucket_empty_rows']:,}")
        md.append("")
    md.append("## Brand alias near-matches")
    md.append(f"- Total candidate aliases (token-set-ratio >= 90): "
              f"{summary['brand_alias_candidates']['rows']:,}")
    md.append(f"- Score >= 95: {summary['brand_alias_candidates']['rows_score_ge_95']:,}")
    md.append("")
    md.append("### Top 25 alias candidates")
    for r in alias_df.head(25).itertuples():
        md.append(f"- B `{r.b_brand_norm}` (n={r.b_count}) ~ A `{r.a_brand_norm}` (n={r.a_count}) "
                  f"score={r.score:.0f}")

    SUMMARY_MD.write_text("\n".join(md), encoding="utf-8")
    print(f"Wrote {SUMMARY_JSON}")
    print(f"Wrote {SUMMARY_MD}")
    print(f"Wrote {FAILURES_CSV}")
    print(f"Wrote {BRAND_ALIASES_CSV}")


if __name__ == "__main__":
    main()
