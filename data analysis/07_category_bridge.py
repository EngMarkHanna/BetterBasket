"""Derive an A->B category bridge from shared-brand seed matches.

The prior agent noted that A and B taxonomies do not align directly:
- info_category_0: 0 shared values
- info_category_2: 47 shared values
- info_category_3: 62 shared values

But shared-brand + size + fuzzy-name matches give us a seed of plausible
pairs, and from those we can learn a *probabilistic* mapping
   A.info_category_2 -> distribution over B.info_category_2
that covers most of the A taxonomy without exact-string overlap.

Outputs:
- outputs/category_bridge.md
- outputs/category_bridge.json
- outputs/category_bridge_pairs.csv         (every (A cat, B cat) seed pair)
- outputs/category_bridge_a_to_b.csv        (per A cat, top-3 B cats with support)
"""
from __future__ import annotations

import json
from collections import Counter, defaultdict
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
)

SUMMARY_MD = OUTPUT_DIR / "category_bridge.md"
SUMMARY_JSON = OUTPUT_DIR / "category_bridge.json"
PAIRS_CSV = OUTPUT_DIR / "category_bridge_pairs.csv"
MAP_CSV = OUTPUT_DIR / "category_bridge_a_to_b.csv"


def make_key(row: pd.Series, cols: list[str]) -> str:
    vals = []
    for col in cols:
        v = row.get(col)
        if is_missing(v):
            return ""
        vals.append(str(v))
    return "|".join(vals)


def main() -> None:
    ensure_output_dir()
    print("Loading datasets")
    a = add_matching_features("A", load_dataset("A"))
    b = add_matching_features("B", load_dataset("B"))

    # Build seed matches: shared brand_norm + size_bucket, RapidFuzz best name match
    a = a[(a["brand_norm"] != "") & (a["size_bucket"] != "")]
    b = b[(b["brand_norm"] != "") & (b["size_bucket"] != "")]

    print(f"A keyed rows: {len(a):,}, B keyed rows: {len(b):,}")

    b_by_key: dict[str, pd.DataFrame] = {
        key: part for key, part in b.groupby(["brand_norm", "size_bucket"], sort=False)
    }

    seed_pairs: list[dict[str, Any]] = []
    a_keyed_count = 0
    for _, arow in a.iterrows():
        a_keyed_count += 1
        key = (arow["brand_norm"], arow["size_bucket"])
        part = b_by_key.get(key)
        if part is None or part.empty:
            continue
        choices = dict(zip(part["item_id"].astype(str), part["name_norm"].astype(str)))
        match = process.extractOne(str(arow["name_norm"]), choices, scorer=fuzz.WRatio)
        if match is None:
            continue
        _, score, b_id = match
        if score < 80:
            continue
        brow = part[part["item_id"].astype(str) == str(b_id)].iloc[0]
        seed_pairs.append(
            {
                "score": float(score),
                "a_cat0": arow.get("info_category_0", ""),
                "a_cat1": arow.get("info_category_1", ""),
                "a_cat2": arow.get("info_category_2", ""),
                "a_cat3": arow.get("info_category_3", ""),
                "b_cat0": brow.get("info_category_0", ""),
                "b_cat1": brow.get("info_category_1", ""),
                "b_cat2": brow.get("info_category_2", ""),
                "b_cat3": brow.get("info_category_3", ""),
                "name_A": arow.get("name", ""),
                "name_B": brow.get("name", ""),
            }
        )

    pairs_df = pd.DataFrame(seed_pairs)
    pairs_df.to_csv(PAIRS_CSV, index=False)
    print(f"Seed pairs (score>=80): {len(pairs_df):,}")

    # Build A.cat2 -> top B.cat2 distribution
    def build_map(level_a: str, level_b: str) -> pd.DataFrame:
        sub = pairs_df[(pairs_df[level_a] != "") & (pairs_df[level_b] != "")]
        counts: dict[str, Counter] = defaultdict(Counter)
        for _, row in sub.iterrows():
            counts[row[level_a]][row[level_b]] += 1
        rows = []
        for a_cat, ctr in counts.items():
            total = sum(ctr.values())
            top3 = ctr.most_common(3)
            row = {
                "a_level": level_a,
                "b_level": level_b,
                "a_category": a_cat,
                "support": total,
            }
            for i, (b_cat, c) in enumerate(top3, start=1):
                row[f"b_top{i}"] = b_cat
                row[f"b_top{i}_share"] = round(c / total, 4)
            rows.append(row)
        return pd.DataFrame(rows).sort_values("support", ascending=False)

    map_cat0 = build_map("a_cat0", "b_cat0")
    map_cat1 = build_map("a_cat1", "b_cat1")
    map_cat2 = build_map("a_cat2", "b_cat2")
    map_cat3 = build_map("a_cat3", "b_cat3")

    map_all = pd.concat([map_cat0, map_cat1, map_cat2, map_cat3], ignore_index=True)
    map_all.to_csv(MAP_CSV, index=False)

    # Coverage stats
    def coverage(level: str, mapping: pd.DataFrame) -> dict[str, Any]:
        n_unique_a_cats = int(a[level].dropna().astype(str).replace("", pd.NA).dropna().nunique())
        n_mapped_cats = int(mapping.shape[0])
        n_with_high_confidence = int((mapping.get("b_top1_share", pd.Series([0])) >= 0.6).sum())
        n_with_support_5 = int((mapping["support"] >= 5).sum())
        return {
            "level": level,
            "unique_a_categories_in_full_dataset": n_unique_a_cats,
            "categories_with_at_least_one_seed": n_mapped_cats,
            "categories_with_support_>=5": n_with_support_5,
            "categories_with_top_b_share_>=0.6": n_with_high_confidence,
        }

    coverage_summary = [
        coverage("info_category_0", map_cat0),
        coverage("info_category_1", map_cat1),
        coverage("info_category_2", map_cat2),
        coverage("info_category_3", map_cat3),
    ]

    summary = {
        "seed_pair_count": int(len(pairs_df)),
        "score_threshold": 80,
        "level_coverage": coverage_summary,
        "examples_cat2_top": map_cat2.head(20).to_dict("records"),
    }
    SUMMARY_JSON.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    md = ["# Category Bridge", "",
          f"Seed pairs from shared brand+size with RapidFuzz score >= 80: {len(pairs_df):,}",
          "",
          "## Per-level coverage", ""]
    for c in coverage_summary:
        md.extend([
            f"### `{c['level']}`",
            f"- Unique A categories in dataset: {c['unique_a_categories_in_full_dataset']:,}",
            f"- A categories with at least one seed pair: {c['categories_with_at_least_one_seed']:,}",
            f"- A categories with support >=5: {c['categories_with_support_>=5']:,}",
            f"- A categories where top-B share >=60%: {c['categories_with_top_b_share_>=0.6']:,}",
            "",
        ])
    md.append("## Top mapped A.info_category_2 categories")
    md.append("")
    for r in map_cat2.head(25).itertuples():
        top1 = getattr(r, "b_top1", "")
        share1 = getattr(r, "b_top1_share", 0)
        md.append(f"- A `{r.a_category}` (support {r.support}) -> B `{top1}` ({share1:.0%})")
    SUMMARY_MD.write_text("\n".join(md), encoding="utf-8")
    print(f"Wrote {SUMMARY_JSON}")
    print(f"Wrote {SUMMARY_MD}")
    print(f"Wrote {PAIRS_CSV}")
    print(f"Wrote {MAP_CSV}")


if __name__ == "__main__":
    main()
