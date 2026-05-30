from __future__ import annotations

import ast
import json
import math
import re
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "dataset"
OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"

DATASETS = {
    "A": DATA_DIR / "grocery_store_a_items_final.csv",
    "B": DATA_DIR / "grocery_store_b_items_final.csv",
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

PRIVATE_LABEL_A_BRANDS = {
    "bettergoods",
    "equate",
    "freshness guaranteed",
    "great value",
    "marketside",
    "members mark",
    "mainstays",
    "ozark trail",
    "parent's choice",
    "sam's choice",
    "time and tru",
    "walmart",
}

PRIVATE_LABEL_B_BRANDS = {
    "wegmans",
    "wegmans organic",
    "wegmans food you feel good about",
    "wegmans ez meals",
}


def ensure_output_dir() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_dataset(label: str, nrows: int | None = None) -> pd.DataFrame:
    path = DATASETS[label]
    return pd.read_csv(path, dtype=str, low_memory=False, nrows=nrows)


def is_missing(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, (list, tuple, set, dict)):
        return len(value) == 0
    if isinstance(value, float) and math.isnan(value):
        return True
    if pd.isna(value):
        return True
    text = str(value).strip()
    return text == "" or text.lower() in {"nan", "none", "null"}


def parse_object(value: Any) -> Any:
    if is_missing(value):
        return {}
    text = str(value).strip()
    if text in {"{}", "[]"}:
        return {} if text == "{}" else []
    try:
        return json.loads(text)
    except Exception:
        pass
    try:
        return ast.literal_eval(text)
    except Exception:
        return {}


def parse_mapping(value: Any) -> dict[str, Any]:
    parsed = parse_object(value)
    return parsed if isinstance(parsed, dict) else {}


def parse_tags(value: Any) -> list[str]:
    parsed = parse_object(value)
    if isinstance(parsed, list):
        return [str(x) for x in parsed]
    if isinstance(parsed, dict):
        return [str(k) for k, v in parsed.items() if bool(v)]
    return []


def normalize_text(value: Any) -> str:
    if is_missing(value):
        return ""
    text = str(value).lower()
    text = text.replace("&", " and ")
    text = re.sub(r"[^a-z0-9]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def extract_fields(df: pd.DataFrame) -> pd.DataFrame:
    enriched = df.copy()

    if "item_info" in enriched.columns:
        info = enriched["item_info"].map(parse_mapping)
        for field in INFO_FIELDS:
            enriched[f"info_{field}"] = info.map(lambda obj, f=field: obj.get(f))

    if "sizing_comp" in enriched.columns:
        sizing = enriched["sizing_comp"].map(parse_mapping)
        for field in SIZING_FIELDS:
            enriched[f"sizing_{field}"] = sizing.map(lambda obj, f=field: obj.get(f))

    if "tags" in enriched.columns:
        enriched["tags_list"] = enriched["tags"].map(parse_tags)
        enriched["tags_norm"] = enriched["tags_list"].map(lambda tags: " ".join(sorted(normalize_text(t) for t in tags)))

    enriched["brand_norm"] = enriched.get("brand_raw", pd.Series(index=enriched.index, dtype=str)).map(normalize_text)

    name_source = pd.Series("", index=enriched.index, dtype=object)
    if "name" in enriched.columns:
        name_source = enriched["name"]
    if "name_clean" in enriched.columns:
        name_source = enriched["name_clean"].where(~enriched["name_clean"].map(is_missing), name_source)
    enriched["name_norm"] = name_source.map(normalize_text)

    return enriched


SIZE_PATTERNS = [
    (re.compile(r"(\d+(?:\.\d+)?)\s*(?:fl\.?\s*oz|fluid\s*ounces?|floz)\b", re.I), "fl_oz", "volume", 1.0),
    (re.compile(r"(\d+(?:\.\d+)?)\s*(?:l|liter|liters|litre|litres)\b", re.I), "fl_oz", "volume", 33.814),
    (re.compile(r"(\d+(?:\.\d+)?)\s*(?:ml|milliliter|milliliters|millilitre|millilitres)\b", re.I), "fl_oz", "volume", 0.033814),
    (re.compile(r"(\d+(?:\.\d+)?)\s*(?:oz|ounce|ounces)\b", re.I), "oz", "weight", 1.0),
    (re.compile(r"(\d+(?:\.\d+)?)\s*(?:lb|lbs|pound|pounds)\b", re.I), "oz", "weight", 16.0),
    (re.compile(r"(\d+(?:\.\d+)?)\s*(?:kg|kilogram|kilograms)\b", re.I), "oz", "weight", 35.274),
    (re.compile(r"(\d+(?:\.\d+)?)\s*(?:g|gram|grams)\b", re.I), "oz", "weight", 0.035274),
    (re.compile(r"(\d+(?:\.\d+)?)\s*(?:ct|count|counts|piece|pieces|pk|pack|packs)\b", re.I), "count", "count", 1.0),
]


def parse_size(value: Any) -> dict[str, Any]:
    if is_missing(value):
        return {"size_value": None, "size_unit": None, "size_dimension": None, "size_canonical": None}

    text = str(value)
    multiplier = 1.0
    pack_match = re.search(r"(\d+)\s*(?:x|pk|pack)\s*(\d+(?:\.\d+)?)", text, re.I)
    if pack_match:
        multiplier = float(pack_match.group(1))

    for pattern, unit, dimension, factor in SIZE_PATTERNS:
        match = pattern.search(text)
        if match:
            raw_value = float(match.group(1))
            canonical = raw_value * factor * multiplier
            return {
                "size_value": raw_value,
                "size_unit": unit,
                "size_dimension": dimension,
                "size_canonical": round(canonical, 4),
            }

    return {"size_value": None, "size_unit": None, "size_dimension": None, "size_canonical": None}


def add_size_features(df: pd.DataFrame) -> pd.DataFrame:
    enriched = df.copy()
    size_source = pd.Series("", index=enriched.index, dtype=object)
    for col in ["sizing_size_user_friendly", "size_raw", "name"]:
        if col in enriched.columns:
            size_source = size_source.where(~size_source.map(is_missing), enriched[col])
    parsed = size_source.map(parse_size)
    for key in ["size_value", "size_unit", "size_dimension", "size_canonical"]:
        enriched[key] = parsed.map(lambda obj, k=key: obj[k])
    enriched["size_bucket"] = enriched.apply(
        lambda row: make_size_bucket(row["size_dimension"], row["size_canonical"]), axis=1
    )
    return enriched


def make_size_bucket(dimension: Any, canonical: Any) -> str:
    if is_missing(dimension) or is_missing(canonical):
        return ""
    try:
        value = float(canonical)
    except Exception:
        return ""
    if str(dimension) == "count":
        rounded = round(value)
    elif value < 1:
        rounded = round(value, 2)
    elif value < 10:
        rounded = round(value, 1)
    else:
        rounded = round(value)
    return f"{dimension}:{rounded:g}"


def infer_private_label(label: str, row: pd.Series) -> bool:
    brand = normalize_text(row.get("brand_raw"))
    if label == "A":
        explicit = normalize_text(row.get("is_private_label"))
        if explicit in {"true", "1", "yes"}:
            return True
        return brand in PRIVATE_LABEL_A_BRANDS

    tags = row.get("tags_list")
    if isinstance(tags, list) and any("wegmans_brand" == str(tag).lower() for tag in tags):
        return True
    return brand in PRIVATE_LABEL_B_BRANDS or brand.startswith("wegmans")


def add_matching_features(label: str, df: pd.DataFrame) -> pd.DataFrame:
    enriched = extract_fields(df)
    enriched = add_size_features(enriched)
    for col in ["info_category_0", "info_category_1", "info_category_2", "info_category_3"]:
        if col in enriched.columns:
            enriched[f"{col}_norm"] = enriched[col].map(normalize_text)
    enriched["is_private_label_inferred"] = enriched.apply(lambda row: infer_private_label(label, row), axis=1)
    return enriched


def nonmissing_rate(series: pd.Series) -> float:
    return float((~series.map(is_missing)).mean())
