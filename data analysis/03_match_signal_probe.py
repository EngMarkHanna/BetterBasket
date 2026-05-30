from __future__ import annotations

import itertools
import json
import re
from collections import Counter
from typing import Any

import pandas as pd
from rapidfuzz import fuzz, process

from eda_utils import (
    DATASETS,
    OUTPUT_DIR,
    add_matching_features,
    ensure_output_dir,
    is_missing,
    load_dataset,
    normalize_text,
)


SUMMARY_MD = OUTPUT_DIR / "match_signal_probe.md"
SUMMARY_JSON = OUTPUT_DIR / "match_signal_probe.json"
CANDIDATE_EXAMPLES = OUTPUT_DIR / "candidate_examples.csv"


def token_set(text: Any) -> set[str]:
    norm = normalize_text(text)
    return {tok for tok in norm.split() if len(tok) > 1}


def jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def has_upc_like_token(row: pd.Series) -> bool:
    text = " ".join(str(row.get(col, "")) for col in ["name", "description", "url", "item_info", "tags"])
    return bool(re.search(r"\b\d{12,14}\b", text))


def build_block_stats(a: pd.DataFrame, b: pd.DataFrame, columns: list[str]) -> dict[str, Any]:
    a_present = pd.Series(True, index=a.index)
    b_present = pd.Series(True, index=b.index)
    for col in columns:
        a_present &= ~a[col].map(is_missing)
        b_present &= ~b[col].map(is_missing)

    a_keys = a.loc[a_present, columns].fillna("").astype(str).agg("|".join, axis=1)
    b_keys = b.loc[b_present, columns].fillna("").astype(str).agg("|".join, axis=1)
    a_counts = a_keys.value_counts()
    b_counts = b_keys.value_counts()
    shared = sorted(set(a_counts.index) & set(b_counts.index))
    pair_estimate = int(sum(int(a_counts[key]) * int(b_counts[key]) for key in shared))
    a_covered = int(a_keys.isin(shared).sum())
    b_covered = int(b_keys.isin(shared).sum())
    return {
        "columns": columns,
        "shared_blocks": len(shared),
        "a_rows_covered": a_covered,
        "b_rows_covered": b_covered,
        "a_coverage": round(a_covered / len(a), 6),
        "b_coverage": round(b_covered / len(b), 6),
        "candidate_pair_estimate": pair_estimate,
        "largest_shared_blocks": [
            {
                "block": key,
                "a_count": int(a_counts[key]),
                "b_count": int(b_counts[key]),
                "pairs": int(a_counts[key] * b_counts[key]),
            }
            for key in sorted(shared, key=lambda k: a_counts[k] * b_counts[k], reverse=True)[:20]
        ],
    }


def overlap_counts(a: pd.DataFrame, b: pd.DataFrame, column: str) -> dict[str, Any]:
    a_values = set(a[column].dropna().astype(str)) - {""}
    b_values = set(b[column].dropna().astype(str)) - {""}
    shared = a_values & b_values
    return {
        "column": column,
        "a_unique": len(a_values),
        "b_unique": len(b_values),
        "shared_unique": len(shared),
        "shared_examples": sorted(shared)[:50],
    }


def make_probe_examples(a: pd.DataFrame, b: pd.DataFrame, limit: int = 200) -> pd.DataFrame:
    b_by_brand: dict[str, pd.DataFrame] = {
        brand: part.head(1000)
        for brand, part in b[b["brand_norm"] != ""].groupby("brand_norm", sort=False)
    }
    examples: list[dict[str, Any]] = []

    candidates = a[(a["brand_norm"] != "") & (a["size_bucket"] != "")].sample(
        min(limit, len(a)), random_state=7
    )
    for _, arow in candidates.iterrows():
        brand = arow["brand_norm"]
        bpool = b_by_brand.get(brand)
        if bpool is None or bpool.empty:
            continue
        if not is_missing(arow.get("size_bucket")):
            same_size = bpool[bpool["size_bucket"] == arow["size_bucket"]]
            if not same_size.empty:
                bpool = same_size
        choices = dict(zip(bpool["item_id"].astype(str), bpool["name_norm"].astype(str)))
        if not choices:
            continue
        match = process.extractOne(str(arow["name_norm"]), choices, scorer=fuzz.WRatio)
        if match is None:
            continue
        _, score, b_item_id = match
        brow = bpool[bpool["item_id"].astype(str) == str(b_item_id)].iloc[0]
        examples.append(
            {
                "item_id_A": arow["item_id"],
                "name_A": arow["name"],
                "brand_A": arow.get("brand_raw"),
                "size_A": arow.get("sizing_size_user_friendly"),
                "category_A": " > ".join(
                    str(arow.get(col))
                    for col in ["info_category_0", "info_category_1", "info_category_2"]
                    if not is_missing(arow.get(col))
                ),
                "item_id_B": brow["item_id"],
                "name_B": brow["name"],
                "brand_B": brow.get("brand_raw"),
                "size_B": brow.get("sizing_size_user_friendly"),
                "category_B": " > ".join(
                    str(brow.get(col))
                    for col in ["info_category_0", "info_category_1", "info_category_2"]
                    if not is_missing(brow.get(col))
                ),
                "size_bucket": arow["size_bucket"],
                "rapidfuzz_wratio": score,
                "token_jaccard": round(jaccard(token_set(arow["name_norm"]), token_set(brow["name_norm"])), 4),
            }
        )

    return pd.DataFrame(examples).sort_values(["rapidfuzz_wratio", "token_jaccard"], ascending=False)


def build_markdown(summary: dict[str, Any]) -> str:
    lines = ["# Match Signal Probe", ""]
    lines.append("## Core coverage")
    for label in ["A", "B"]:
        store = summary["stores"][label]
        lines.extend(
            [
                "",
                f"Store {label}:",
                f"- Rows: {store['rows']:,}",
                f"- UPC-like token rows: {store['upc_like_rows']:,} ({store['upc_like_rate']:.2%})",
                f"- Brand coverage: {store['brand_coverage']:.2%}",
                f"- Parsed size coverage: {store['size_coverage']:.2%}",
                f"- Inferred private label: {store['private_label_rows']:,} ({store['private_label_rate']:.2%})",
            ]
        )

    lines.extend(["", "## Cross-store overlap", ""])
    for row in summary["overlaps"]:
        lines.append(
            f"- `{row['column']}` shared unique values: {row['shared_unique']:,} "
            f"(A {row['a_unique']:,}, B {row['b_unique']:,})"
        )

    lines.extend(["", "## Blocking estimates", ""])
    for block in summary["blocks"]:
        cols = ", ".join(f"`{c}`" for c in block["columns"])
        lines.append(
            f"- {cols}: {block['shared_blocks']:,} shared blocks, "
            f"A coverage {block['a_coverage']:.1%}, B coverage {block['b_coverage']:.1%}, "
            f"estimated candidate pairs {block['candidate_pair_estimate']:,}"
        )

    lines.extend(["", "## Top shared brands", ""])
    for brand, counts in summary["top_shared_brands"][:25]:
        lines.append(f"- {brand}: A {counts['A']:,}, B {counts['B']:,}")

    return "\n".join(lines)


def main() -> None:
    ensure_output_dir()
    print("Loading and enriching datasets")
    a = add_matching_features("A", load_dataset("A"))
    b = add_matching_features("B", load_dataset("B"))

    summary: dict[str, Any] = {"stores": {}, "overlaps": [], "blocks": []}
    for label, df in [("A", a), ("B", b)]:
        upc_rows = int(df.apply(has_upc_like_token, axis=1).sum())
        brand_rows = int((df["brand_norm"] != "").sum())
        size_rows = int((df["size_bucket"] != "").sum())
        private_rows = int(df["is_private_label_inferred"].sum())
        summary["stores"][label] = {
            "rows": int(len(df)),
            "upc_like_rows": upc_rows,
            "upc_like_rate": round(upc_rows / len(df), 6),
            "brand_coverage": round(brand_rows / len(df), 6),
            "size_coverage": round(size_rows / len(df), 6),
            "private_label_rows": private_rows,
            "private_label_rate": round(private_rows / len(df), 6),
        }

    for col in [
        "brand_norm",
        "size_bucket",
        "info_category_0_norm",
        "info_category_1_norm",
        "info_category_2_norm",
        "info_category_3_norm",
    ]:
        if col in a.columns and col in b.columns:
            summary["overlaps"].append(overlap_counts(a, b, col))

    for columns in [
        ["brand_norm"],
        ["brand_norm", "size_bucket"],
        ["brand_norm", "size_bucket", "info_category_2_norm"],
        ["size_bucket", "info_category_2_norm"],
        ["is_private_label_inferred", "size_bucket", "info_category_2_norm"],
    ]:
        summary["blocks"].append(build_block_stats(a, b, columns))

    a_brand_counts = Counter(a.loc[a["brand_norm"] != "", "brand_norm"])
    b_brand_counts = Counter(b.loc[b["brand_norm"] != "", "brand_norm"])
    shared_brands = set(a_brand_counts) & set(b_brand_counts)
    summary["top_shared_brands"] = [
        (brand, {"A": int(a_brand_counts[brand]), "B": int(b_brand_counts[brand])})
        for brand in sorted(shared_brands, key=lambda x: a_brand_counts[x] + b_brand_counts[x], reverse=True)
    ][:100]

    print("Creating candidate examples")
    examples = make_probe_examples(a, b)
    examples.to_csv(CANDIDATE_EXAMPLES, index=False)

    SUMMARY_JSON.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    SUMMARY_MD.write_text(build_markdown(summary), encoding="utf-8")
    print(f"Wrote {SUMMARY_JSON}")
    print(f"Wrote {SUMMARY_MD}")
    print(f"Wrote {CANDIDATE_EXAMPLES}")


if __name__ == "__main__":
    main()
