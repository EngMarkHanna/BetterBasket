"""Parsing primitives: sizes (unit + total), brand canonicalization, slug,
private-label inference, food/fresh hints.

The size parser is intentionally more careful than the EDA version:
- It extracts a per-unit size AND a total package size when a multipack
  pattern is present, so we don't collapse a 12-pack into one number.
- It produces buckets for both, so retrieval can block on either.

Brand aliasing is the curated 20-entry map locked in Phase 0
(`13_phase0_feasibility.py`).
"""
from __future__ import annotations

import ast
import json
import math
import re
from typing import Any


# 20-entry curated alias map. Promoted from
# `outputs/brand_alias_candidates.csv` (score >= 95, not on the risky
# generic list).
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


PRIVATE_LABEL_A_BRANDS: set[str] = {
    "bettergoods",
    "equate",
    "freshness guaranteed",
    "great value",
    "marketside",
    "members mark",
    "mainstays",
    "ozark trail",
    "parent s choice",
    "parents choice",
    "sam s choice",
    "time and tru",
    "walmart",
}

PRIVATE_LABEL_B_BRANDS: set[str] = {
    "wegmans",
    "wegmans organic",
    "wegmans food you feel good about",
    "wegmans ez meals",
}


FOOD_HINTS = {
    "food",
    "grocery",
    "dairy",
    "frozen",
    "produce",
    "beverages",
    "snacks",
    "pantry",
    "bakery",
    "meat",
    "seafood",
    "deli",
    "candy",
    "breakfast",
    "household essentials",
}

FRESH_HINTS = {
    "produce",
    "meat",
    "seafood",
    "deli",
    "bakery",
    "prepared foods",
    "floral",
    "fresh",
}

ORGANIC_HINTS = {"organic"}


def is_missing(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    text = str(value).strip()
    return text == "" or text.lower() in {"nan", "none", "null"}


def parse_object(value: Any) -> Any:
    if is_missing(value):
        return {}
    text = str(value).strip()
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


_WORD_RE = re.compile(r"[^a-z0-9]+")
_SPACE_RE = re.compile(r"\s+")


def normalize_text(value: Any) -> str:
    if is_missing(value):
        return ""
    text = str(value).lower().replace("&", " and ")
    text = _WORD_RE.sub(" ", text)
    return _SPACE_RE.sub(" ", text).strip()


def canonical_brand(brand_norm: str) -> str:
    if not brand_norm:
        return ""
    return BRAND_ALIASES.get(brand_norm, brand_norm)


_URL_SLUG_RE = re.compile(r"/([a-z0-9][a-z0-9\-_]+)/?$", re.I)


def extract_url_slug(url: str) -> str:
    if is_missing(url):
        return ""
    match = _URL_SLUG_RE.search(str(url).rstrip("/"))
    return normalize_text(match.group(1)) if match else ""


# Size patterns ordered by specificity; the multipack handler tries to
# extract pack count + per-unit size from "12 x 12 oz", "12pk 12 fl oz",
# "12 ct 0.5 oz" etc.
_PACK_RE = re.compile(
    r"(?P<count>\d+)\s*(?:x|pk|pack|ct|count)\s*(?P<size>\d+(?:\.\d+)?)\s*"
    r"(?P<unit>fl\.?\s*oz|fluid\s*ounces?|floz|ml|millilit(?:re|er)s?|l|liters?|litres?"
    r"|oz|ounces?|lb|lbs|pound|pounds?|kg|kilograms?|g|grams?)",
    re.I,
)

_SIZE_PATTERNS: list[tuple[re.Pattern, str, str, float]] = [
    (re.compile(r"(\d+(?:\.\d+)?)\s*(?:fl\.?\s*oz|fluid\s*ounces?|floz)\b", re.I), "fl_oz", "volume", 1.0),
    (re.compile(r"(\d+(?:\.\d+)?)\s*(?:l|liter|liters|litre|litres)\b", re.I), "fl_oz", "volume", 33.814),
    (re.compile(r"(\d+(?:\.\d+)?)\s*(?:ml|milliliter|milliliters|millilitre|millilitres)\b", re.I), "fl_oz", "volume", 0.033814),
    (re.compile(r"(\d+(?:\.\d+)?)\s*(?:oz|ounce|ounces)\b", re.I), "oz", "weight", 1.0),
    (re.compile(r"(\d+(?:\.\d+)?)\s*(?:lb|lbs|pound|pounds)\b", re.I), "oz", "weight", 16.0),
    (re.compile(r"(\d+(?:\.\d+)?)\s*(?:kg|kilogram|kilograms)\b", re.I), "oz", "weight", 35.274),
    (re.compile(r"(\d+(?:\.\d+)?)\s*(?:g|gram|grams)\b", re.I), "oz", "weight", 0.035274),
    (re.compile(r"(\d+(?:\.\d+)?)\s*(?:ct|count|counts|piece|pieces|pk|pack|packs)\b", re.I), "count", "count", 1.0),
]

_UNIT_FACTORS = {
    "fl_oz": [
        (re.compile(r"^fl\.?\s*oz|fluid\s*ounces?|floz$", re.I), 1.0),
        (re.compile(r"^l|liters?|litres?$", re.I), 33.814),
        (re.compile(r"^ml|millilit(?:re|er)s?$", re.I), 0.033814),
    ],
    "oz": [
        (re.compile(r"^oz|ounces?$", re.I), 1.0),
        (re.compile(r"^lb|lbs|pound|pounds?$", re.I), 16.0),
        (re.compile(r"^kg|kilograms?$", re.I), 35.274),
        (re.compile(r"^g|grams?$", re.I), 0.035274),
    ],
}


def _classify_unit(unit_text: str) -> tuple[str, str, float] | None:
    """Return (canonical_unit, dimension, factor_to_canonical)."""
    norm = unit_text.strip().lower()
    for canonical, options in _UNIT_FACTORS.items():
        for pattern, factor in options:
            if pattern.match(norm):
                dim = "volume" if canonical == "fl_oz" else "weight"
                return canonical, dim, factor
    return None


def _bucket(dimension: str, canonical_value: float) -> str:
    if not dimension or canonical_value is None:
        return ""
    v = float(canonical_value)
    if dimension == "count":
        rounded = round(v)
    elif v < 1:
        rounded = round(v, 2)
    elif v < 10:
        rounded = round(v, 1)
    else:
        rounded = round(v)
    return f"{dimension}:{rounded:g}"


def parse_size(text: str) -> dict[str, Any]:
    """Parse a size string into unit + total features.

    Returns:
      unit_value / unit_unit / unit_dim / unit_canonical / unit_bucket
      total_value / total_unit / total_dim / total_canonical / total_bucket
      pack_count
      count_bucket   (only when pack_count > 1)

    For non-multipack inputs, unit == total.
    """
    out: dict[str, Any] = {
        "unit_value": None,
        "unit_unit": None,
        "unit_dim": None,
        "unit_canonical": None,
        "unit_bucket": "",
        "total_value": None,
        "total_unit": None,
        "total_dim": None,
        "total_canonical": None,
        "total_bucket": "",
        "pack_count": None,
        "count_bucket": "",
    }
    if is_missing(text):
        return out

    s = str(text)

    pack_match = _PACK_RE.search(s)
    if pack_match:
        count = int(pack_match.group("count"))
        size = float(pack_match.group("size"))
        unit_classified = _classify_unit(pack_match.group("unit"))
        if unit_classified:
            canonical_unit, dim, factor = unit_classified
            unit_canon = round(size * factor, 4)
            total_canon = round(unit_canon * count, 4)
            out.update(
                {
                    "unit_value": size,
                    "unit_unit": canonical_unit,
                    "unit_dim": dim,
                    "unit_canonical": unit_canon,
                    "unit_bucket": _bucket(dim, unit_canon),
                    "total_value": round(size * count, 4),
                    "total_unit": canonical_unit,
                    "total_dim": dim,
                    "total_canonical": total_canon,
                    "total_bucket": _bucket(dim, total_canon),
                    "pack_count": count,
                    "count_bucket": f"count:{count}" if count > 1 else "",
                }
            )
            return out

    for pattern, unit, dim, factor in _SIZE_PATTERNS:
        match = pattern.search(s)
        if match:
            value = float(match.group(1))
            canonical = round(value * factor, 4)
            bucket = _bucket(dim, canonical)
            out.update(
                {
                    "unit_value": value,
                    "unit_unit": unit,
                    "unit_dim": dim,
                    "unit_canonical": canonical,
                    "unit_bucket": bucket,
                    "total_value": value,
                    "total_unit": unit,
                    "total_dim": dim,
                    "total_canonical": canonical,
                    "total_bucket": bucket,
                    "pack_count": 1 if dim != "count" else int(round(canonical)),
                    "count_bucket": "",
                }
            )
            return out

    return out


def best_size_text(row_fields: dict[str, Any]) -> str:
    """Pick the most informative size string from a row's available fields."""
    for key in ("sizing_size_user_friendly", "size_raw", "name"):
        v = row_fields.get(key)
        if not is_missing(v):
            return str(v)
    return ""


def infer_private_label(store: str, brand_norm: str, tags_list: list[str], is_private_label_raw: Any) -> bool:
    if store == "A":
        if normalize_text(is_private_label_raw) in {"true", "1", "yes"}:
            return True
        return brand_norm in PRIVATE_LABEL_A_BRANDS
    if isinstance(tags_list, list) and any("wegmans_brand" == str(t).lower() for t in tags_list):
        return True
    return brand_norm in PRIVATE_LABEL_B_BRANDS or brand_norm.startswith("wegmans")


def category_tokens(*categories: str) -> str:
    parts = [normalize_text(c) for c in categories if not is_missing(c)]
    return " ".join(p for p in parts if p)


def has_any(text: str, words: set[str]) -> bool:
    if not text:
        return False
    tokens = set(text.split())
    return bool(tokens & words)


def infer_is_food_like(category_norm: str) -> bool:
    return has_any(category_norm, FOOD_HINTS)


def infer_is_fresh_like(category_norm: str) -> bool:
    return has_any(category_norm, FRESH_HINTS)


def infer_is_organic(name_norm: str, tags_norm: str, is_organic_raw: Any) -> bool:
    if normalize_text(is_organic_raw) in {"true", "1", "yes"}:
        return True
    return "organic" in name_norm.split() or "organic" in tags_norm.split()
