"""Canonical loader: read the two CSVs, parse nested fields, attach all
features the rest of the pipeline depends on.

The result of `load_store(label)` is a DataFrame with one row per
product and a stable set of columns documented in the canonical schema.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from .parse import (
    best_size_text,
    canonical_brand,
    category_tokens,
    extract_url_slug,
    infer_is_food_like,
    infer_is_fresh_like,
    infer_is_organic,
    infer_private_label,
    is_missing,
    normalize_text,
    parse_mapping,
    parse_size,
    parse_tags,
)


ROOT = Path(__file__).resolve().parents[2]
DATASET_DIR = ROOT / "dataset"
CSVS = {
    "A": DATASET_DIR / "grocery_store_a_items_final.csv",
    "B": DATASET_DIR / "grocery_store_b_items_final.csv",
}


INFO_FIELDS = [
    "category_0",
    "category_1",
    "category_2",
    "category_3",
    "storage_type",
    "packaging_description",
    "ingredients",
    "planogram",
]

SIZING_FIELDS = [
    "unit_price",
    "uom_unit_price",
    "billed_by_weight",
    "ordered_by_weight",
    "avg_size_per_piece",
    "size_user_friendly",
    "size_from_unit_price",
    "uom_billed_by_weight",
    "uom_ordered_by_weight",
    "num_servings_nutrition",
    "serving_size_nutrition",
    "serving_size_uom_nutrition",
]


def _expand_mapping(df: pd.DataFrame, source: str, fields: list[str], prefix: str) -> None:
    if source not in df.columns:
        for f in fields:
            df[f"{prefix}_{f}"] = None
        return
    parsed = df[source].map(parse_mapping)
    for f in fields:
        df[f"{prefix}_{f}"] = parsed.map(lambda obj, k=f: obj.get(k))


def load_store(label: str, nrows: int | None = None) -> pd.DataFrame:
    """Read a single store's CSV and produce the canonical table."""
    path = CSVS[label]
    df = pd.read_csv(path, dtype=str, low_memory=False, nrows=nrows)
    # Drop malformed rows (CSVs have a handful of rows where the name/desc
    # column overflowed and corrupted the item_id field). Keep first.
    df = df.dropna(subset=["item_id"])
    df = df[~df["item_id"].astype(str).str.contains("|", regex=False).fillna(False) | df["item_id"].astype(str).str.match(r"^\d+$")]
    df = df.drop_duplicates(subset=["item_id"], keep="first").reset_index(drop=True)
    df["store_id"] = label

    # Expand nested fields.
    _expand_mapping(df, "item_info", INFO_FIELDS, "info")
    _expand_mapping(df, "sizing_comp", SIZING_FIELDS, "sizing")
    if "tags" in df.columns:
        df["tags_list"] = df["tags"].map(parse_tags)
        df["tags_norm"] = df["tags_list"].map(
            lambda tags: " ".join(sorted(normalize_text(t) for t in tags))
        )
    else:
        df["tags_list"] = [[] for _ in range(len(df))]
        df["tags_norm"] = ""

    # Names.
    if "name_clean" in df.columns:
        name_src = df["name_clean"].where(~df["name_clean"].map(is_missing), df.get("name"))
    else:
        name_src = df.get("name", pd.Series("", index=df.index))
    df["name_clean_str"] = name_src.fillna("")
    df["name_norm"] = name_src.map(normalize_text)

    # Brand normalization + canonicalization.
    df["brand_norm"] = df.get("brand_raw", pd.Series("", index=df.index)).map(normalize_text)
    df["brand_canonical"] = df["brand_norm"].map(canonical_brand)

    # URL slug.
    df["url_slug_norm"] = df.get("url", pd.Series("", index=df.index)).map(extract_url_slug)

    # Category path normalizations.
    for col in ("info_category_0", "info_category_1", "info_category_2", "info_category_3"):
        df[f"{col}_norm"] = df[col].map(normalize_text) if col in df.columns else ""
    df["category_path_norm"] = df.apply(
        lambda row: category_tokens(
            row.get("info_category_0", ""),
            row.get("info_category_1", ""),
            row.get("info_category_2", ""),
            row.get("info_category_3", ""),
        ),
        axis=1,
    )

    # Description normalization.
    df["description_norm"] = df.get("description", pd.Series("", index=df.index)).map(normalize_text)

    # Ingredients.
    df["ingredients_norm"] = df.get("info_ingredients", pd.Series("", index=df.index)).map(normalize_text)

    # Size parsing (unit + total).
    size_text = df.apply(
        lambda row: best_size_text(
            {
                "sizing_size_user_friendly": row.get("sizing_size_user_friendly"),
                "size_raw": row.get("size_raw"),
                "name": row.get("name"),
            }
        ),
        axis=1,
    )
    df["size_text"] = size_text
    size_parsed = size_text.map(parse_size)
    for key in (
        "unit_value",
        "unit_unit",
        "unit_dim",
        "unit_canonical",
        "unit_bucket",
        "total_value",
        "total_unit",
        "total_dim",
        "total_canonical",
        "total_bucket",
        "pack_count",
        "count_bucket",
    ):
        df[key] = size_parsed.map(lambda d, k=key: d.get(k))

    # Inferred flags.
    df["is_private_label_inferred"] = df.apply(
        lambda row: infer_private_label(
            label, row["brand_norm"], row["tags_list"], row.get("is_private_label")
        ),
        axis=1,
    )
    df["is_food_like"] = df["category_path_norm"].map(infer_is_food_like)
    df["is_fresh_like"] = df["category_path_norm"].map(infer_is_fresh_like)
    df["is_organic_inferred"] = df.apply(
        lambda row: infer_is_organic(row["name_norm"], row["tags_norm"], row.get("is_organic")),
        axis=1,
    )

    # Final retrieval-ready text blob.
    df["retrieval_text"] = (
        df["name_norm"].fillna("") + " "
        + df["brand_canonical"].fillna("") + " "
        + df["url_slug_norm"].fillna("") + " "
        + df["category_path_norm"].fillna("") + " "
        + df["description_norm"].fillna("").str.slice(0, 200)
    ).str.strip()

    return df


def keep_columns() -> list[str]:
    """Stable column subset that downstream code may rely on."""
    return [
        "store_id",
        "item_id",
        "name",
        "name_clean_str",
        "name_norm",
        "brand_raw",
        "brand_norm",
        "brand_canonical",
        "url",
        "url_slug_norm",
        "description_norm",
        "ingredients_norm",
        "info_category_0",
        "info_category_1",
        "info_category_2",
        "info_category_3",
        "category_path_norm",
        "tags_norm",
        "size_text",
        "unit_value",
        "unit_unit",
        "unit_dim",
        "unit_canonical",
        "unit_bucket",
        "total_value",
        "total_unit",
        "total_dim",
        "total_canonical",
        "total_bucket",
        "pack_count",
        "count_bucket",
        "is_private_label_inferred",
        "is_food_like",
        "is_fresh_like",
        "is_organic_inferred",
        "retrieval_text",
    ]
