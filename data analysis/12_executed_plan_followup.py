from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pandas as pd

from eda_utils import OUTPUT_DIR, add_matching_features, is_missing, load_dataset, normalize_text


SUMMARY_MD = OUTPUT_DIR / "executed_plan_followup.md"
SUMMARY_JSON = OUTPUT_DIR / "executed_plan_followup.json"
FAILURES_CSV = OUTPUT_DIR / "eval_failure_review.csv"


SAFE_ALIAS_PAIRS = {
    ("e l f cosmetics", "e l f"),
    ("amy s kitchen", "amy s"),
    ("l oreal paris", "l oreal"),
    ("nature s path organic", "nature s path"),
    ("bigelow tea", "bigelow"),
    ("so delicious dairy free", "so delicious"),
    ("bush s best", "bush s"),
    ("stonyfield organic", "stonyfield"),
    ("nestl toll house", "toll house"),
    ("lindt lindor", "lindt"),
    ("u by kotex", "kotex"),
    ("rachael ray nutrish", "nutrish"),
    ("clif bar", "clif"),
    ("voortman bakery", "voortman"),
    ("mt olive pickle", "mt olive"),
    ("suave essentials", "suave"),
    ("death wish coffee co", "death wish coffee"),
    ("monster energy ultra", "monster energy"),
    ("c4 energy", "c4"),
    ("bobs red mill", "bob s red mill"),
}


UNIT_PATTERNS = [
    (re.compile(r"(?<![\d.])(\d+(?:\.\d+)?|\.\d+)\s*(?:fl\.?\s*oz|fluid\s*ounces?|floz)\b", re.I), "fl_oz", "volume", 1.0),
    (re.compile(r"(?<![\d.])(\d+(?:\.\d+)?|\.\d+)\s*(?:l|liter|liters|litre|litres)\b", re.I), "fl_oz", "volume", 33.814),
    (re.compile(r"(?<![\d.])(\d+(?:\.\d+)?|\.\d+)\s*(?:ml|milliliter|milliliters|millilitre|millilitres)\b", re.I), "fl_oz", "volume", 0.033814),
    (re.compile(r"(?<![\d.])(\d+(?:\.\d+)?|\.\d+)\s*(?:oz|ounce|ounces)\b", re.I), "oz", "weight", 1.0),
    (re.compile(r"(?<![\d.])(\d+(?:\.\d+)?|\.\d+)\s*(?:lb|lbs|pound|pounds)\b", re.I), "oz", "weight", 16.0),
    (re.compile(r"(?<![\d.])(\d+(?:\.\d+)?|\.\d+)\s*(?:kg|kilogram|kilograms)\b", re.I), "oz", "weight", 35.274),
    (re.compile(r"(?<![\d.])(\d+(?:\.\d+)?|\.\d+)\s*(?:g|gram|grams)\b", re.I), "oz", "weight", 0.035274),
]

COUNT_PATTERN = re.compile(r"(?<![\d.])(\d+(?:\.\d+)?)\s*(?:ct|count|counts|each|piece|pieces)\b", re.I)
PACK_PATTERNS = [
    re.compile(r"^\s*\(?\s*(\d{1,3})\s*[- ]?(?:pack|pk)\s*\)?\b", re.I),
    re.compile(r"\bpack\s+of\s+(\d{1,3})\b", re.I),
    re.compile(r"\b(\d{1,3})\s*[- ]?(?:pack|pk)\b", re.I),
    re.compile(r"\b(\d{1,3})\s*(?:ct|count)\s+(?:cans|bottles|bars|packets|pouches|bags|cups)\b", re.I),
]


def canonical_brand(value: Any) -> str:
    brand = normalize_text(value)
    for left, right in SAFE_ALIAS_PAIRS:
        if brand == left:
            return right
        if brand == right:
            return right
    return brand


def parse_number(value: str) -> float:
    if value.startswith("."):
        value = "0" + value
    return float(value)


def bucket(dimension: str | None, value: float | None) -> str:
    if not dimension or value is None:
        return ""
    if dimension == "count":
        rounded: float | int = round(value)
        if rounded <= 1:
            return ""
    elif value < 1:
        rounded = round(value, 2)
    elif value < 10:
        rounded = round(value, 1)
    else:
        rounded = round(value)
    return f"{dimension}:{rounded:g}"


def extract_pack_count(text: str) -> int:
    for pattern in PACK_PATTERNS:
        match = pattern.search(text)
        if match:
            count = int(match.group(1))
            if 1 < count <= 100:
                return count
    return 1


def improved_parse_size(value: Any) -> dict[str, Any]:
    if is_missing(value):
        return {
            "unit_dimension": "",
            "unit_value": None,
            "unit_bucket": "",
            "total_bucket": "",
            "pack_count": 1,
            "count_bucket": "",
        }

    text = str(value)
    pack_count = extract_pack_count(text)
    for pattern, _, dimension, factor in UNIT_PATTERNS:
        match = pattern.search(text)
        if match:
            unit_value = parse_number(match.group(1)) * factor
            return {
                "unit_dimension": dimension,
                "unit_value": round(unit_value, 4),
                "unit_bucket": bucket(dimension, unit_value),
                "total_bucket": bucket(dimension, unit_value * pack_count),
                "pack_count": pack_count,
                "count_bucket": "",
            }

    match = COUNT_PATTERN.search(text)
    if match:
        count = parse_number(match.group(1))
        return {
            "unit_dimension": "count",
            "unit_value": count,
            "unit_bucket": bucket("count", count),
            "total_bucket": bucket("count", count),
            "pack_count": pack_count,
            "count_bucket": bucket("count", count),
        }

    return {
        "unit_dimension": "",
        "unit_value": None,
        "unit_bucket": "",
        "total_bucket": "",
        "pack_count": pack_count,
        "count_bucket": "",
    }


def size_source(df: pd.DataFrame) -> pd.Series:
    source = pd.Series("", index=df.index, dtype=object)
    if "name" in df.columns:
        source = df["name"]
    if "size_raw" in df.columns:
        source = df["size_raw"].where(~df["size_raw"].map(is_missing), source)
    if "sizing_size_user_friendly" in df.columns:
        source = df["sizing_size_user_friendly"].where(~df["sizing_size_user_friendly"].map(is_missing), source)
    return source


def add_followup_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    source = size_source(out)
    parsed = source.map(improved_parse_size)
    for key in ["unit_dimension", "unit_value", "unit_bucket", "total_bucket", "pack_count", "count_bucket"]:
        out[f"new_{key}"] = parsed.map(lambda obj, k=key: obj[k])
    out["brand_canonical_probe"] = out["brand_norm"].map(canonical_brand)
    return out


def block_coverage(a: pd.DataFrame, b: pd.DataFrame, brand_col: str, size_cols: list[str]) -> dict[str, Any]:
    b_keys: set[str] = set()
    for _, row in b.iterrows():
        brand = row.get(brand_col, "")
        if is_missing(brand):
            continue
        for col in size_cols:
            value = row.get(col, "")
            if not is_missing(value):
                b_keys.add(f"{brand}|{value}")

    complete_a = 0
    covered_a = 0
    for _, row in a.iterrows():
        brand = row.get(brand_col, "")
        if is_missing(brand):
            continue
        row_keys = []
        for col in size_cols:
            value = row.get(col, "")
            if not is_missing(value):
                row_keys.append(f"{brand}|{value}")
        if row_keys:
            complete_a += 1
            if any(key in b_keys for key in row_keys):
                covered_a += 1

    return {
        "brand_col": brand_col,
        "size_cols": size_cols,
        "a_rows_with_complete_key": complete_a,
        "a_rows_with_shared_b_block": covered_a,
        "a_shared_block_coverage": round(covered_a / len(a), 6),
    }


def parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if is_missing(value):
        return False
    return str(value).strip().lower() in {"true", "1", "yes"}


def metrics(df: pd.DataFrame, pred_col: str) -> dict[str, Any]:
    labels = df["label_is_match"].map(parse_bool)
    preds = df[pred_col].map(parse_bool)
    tp = int((labels & preds).sum())
    fp = int((~labels & preds).sum())
    fn = int((labels & ~preds).sum())
    tn = int((~labels & ~preds).sum())
    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / (tp + fn) if tp + fn else None
    f1 = 2 * precision * recall / (precision + recall) if precision and recall else None
    return {
        "n": int(len(df)),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": None if precision is None else round(precision, 4),
        "recall": None if recall is None else round(recall, 4),
        "f1": None if f1 is None else round(f1, 4),
    }


def analyze_eval() -> tuple[dict[str, Any], pd.DataFrame]:
    df = pd.read_csv(OUTPUT_DIR / "eval_results.csv")
    df["label_bool"] = df["label_is_match"].map(parse_bool)
    df["pred_bool"] = df["predicted_is_match"].map(parse_bool)
    df["pred_conf"] = pd.to_numeric(df["predicted_confidence"], errors="coerce").fillna(0.0)

    df["policy_llm_any_positive"] = df["pred_bool"]
    df["policy_llm_conf_gte_085"] = df["pred_bool"] & (df["pred_conf"] >= 0.85)
    df["policy_llm_conf_gte_070"] = df["pred_bool"] & (df["pred_conf"] >= 0.70)
    df["policy_auto_strong_plus_llm085"] = (
        df["stratum"].eq("A_strong") | df["policy_llm_conf_gte_085"]
    )
    df["policy_auto_strong_tfidf_plus_llm085"] = (
        df["stratum"].isin(["A_strong", "T_strong_tfidf"]) | df["policy_llm_conf_gte_085"]
    )

    policy_metrics = {
        col.replace("policy_", ""): metrics(df, col)
        for col in [
            "policy_llm_any_positive",
            "policy_llm_conf_gte_085",
            "policy_llm_conf_gte_070",
            "policy_auto_strong_plus_llm085",
            "policy_auto_strong_tfidf_plus_llm085",
        ]
    }

    strata = []
    for stratum, part in df.groupby("stratum"):
        strata.append(
            {
                "stratum": stratum,
                "n": int(len(part)),
                "label_pos_rate": round(float(part["label_bool"].mean()), 4),
                "model_pos_rate": round(float(part["pred_bool"].mean()), 4),
                "model_conf_gte_085_pos": int((part["pred_bool"] & (part["pred_conf"] >= 0.85)).sum()),
            }
        )

    failures = df[
        (df["label_bool"] != df["pred_bool"])
        | (df["pred_bool"] & (df["pred_conf"] < 0.85))
        | (~df["json_valid"].map(parse_bool))
    ].copy()
    failures.to_csv(FAILURES_CSV, index=False)

    return {"policies": policy_metrics, "strata": strata}, failures


def analyze_parser_and_aliases() -> dict[str, Any]:
    print("Loading/enriching datasets for parser and alias probe")
    a = add_followup_features(add_matching_features("A", load_dataset("A")))
    b = add_followup_features(add_matching_features("B", load_dataset("B")))

    summary: dict[str, Any] = {"stores": {}, "block_coverage": []}
    for label, df in [("A", a), ("B", b)]:
        old_nonempty = int((df["size_bucket"] != "").sum())
        unit_nonempty = int((df["new_unit_bucket"] != "").sum())
        total_nonempty = int((df["new_total_bucket"] != "").sum())
        recovered = int(((df["size_bucket"] == "") & (df["new_unit_bucket"] != "")).sum())
        trivial_removed = int(((df["size_bucket"] == "count:1") & (df["new_unit_bucket"] == "")).sum())
        multipack_with_unit = int(((df["new_pack_count"] > 1) & (df["new_unit_bucket"] != "")).sum())
        changed = int(((df["size_bucket"] != df["new_unit_bucket"]) & (df["new_unit_bucket"] != "")).sum())
        summary["stores"][label] = {
            "rows": int(len(df)),
            "old_size_bucket_nonempty": old_nonempty,
            "new_unit_bucket_nonempty": unit_nonempty,
            "new_total_bucket_nonempty": total_nonempty,
            "old_empty_recovered_by_new_unit": recovered,
            "old_count1_removed": trivial_removed,
            "multipack_rows_with_unit_size": multipack_with_unit,
            "rows_where_old_differs_from_new_unit": changed,
            "shared_brand_rows_old": int(df["brand_norm"].isin(set(a["brand_norm"]) & set(b["brand_norm"])).sum()),
            "shared_brand_rows_alias_probe": int(
                df["brand_canonical_probe"].isin(set(a["brand_canonical_probe"]) & set(b["brand_canonical_probe"])).sum()
            ),
        }

    summary["block_coverage"].append(block_coverage(a, b, "brand_norm", ["size_bucket"]))
    summary["block_coverage"].append(block_coverage(a, b, "brand_norm", ["new_unit_bucket"]))
    summary["block_coverage"].append(block_coverage(a, b, "brand_norm", ["new_total_bucket"]))
    summary["block_coverage"].append(block_coverage(a, b, "brand_norm", ["new_unit_bucket", "new_total_bucket"]))
    summary["block_coverage"].append(block_coverage(a, b, "brand_canonical_probe", ["new_unit_bucket", "new_total_bucket"]))
    return summary


def pct(part: int, whole: int) -> str:
    return f"{part / whole:.1%}" if whole else "0.0%"


def build_markdown(summary: dict[str, Any]) -> str:
    lines = ["# Executed Plan Follow-up EDA", ""]

    lines.append("## Parser and brand-alias probe")
    for label, store in summary["parser_alias"]["stores"].items():
        lines.extend(
            [
                "",
                f"Store {label}:",
                f"- Rows: {store['rows']:,}",
                f"- Old parsed size buckets: {store['old_size_bucket_nonempty']:,} ({pct(store['old_size_bucket_nonempty'], store['rows'])})",
                f"- New unit-size buckets: {store['new_unit_bucket_nonempty']:,} ({pct(store['new_unit_bucket_nonempty'], store['rows'])})",
                f"- Old empty rows recovered by new parser: {store['old_empty_recovered_by_new_unit']:,}",
                f"- Old `count:1` rows removed: {store['old_count1_removed']:,}",
                f"- Multipack rows with parsed unit size: {store['multipack_rows_with_unit_size']:,}",
                f"- Rows where old bucket differs from new unit bucket: {store['rows_where_old_differs_from_new_unit']:,}",
                f"- Rows with shared raw normalized brand: {store['shared_brand_rows_old']:,}",
                f"- Rows with shared alias-canonical brand probe: {store['shared_brand_rows_alias_probe']:,}",
            ]
        )

    lines.extend(["", "Brand + size block coverage variants:", ""])
    for row in summary["parser_alias"]["block_coverage"]:
        lines.append(
            f"- `{row['brand_col']}` + `{'+'.join(row['size_cols'])}`: "
            f"A complete {row['a_rows_with_complete_key']:,}, "
            f"A shared-B block {row['a_rows_with_shared_b_block']:,} "
            f"({row['a_shared_block_coverage']:.1%} of A)"
        )

    lines.extend(["", "## Eval-set routing policy simulation", ""])
    for name, m in summary["eval"]["policies"].items():
        lines.append(
            f"- `{name}`: TP {m['tp']}, FP {m['fp']}, FN {m['fn']}, TN {m['tn']}, "
            f"P={m['precision']}, R={m['recall']}, F1={m['f1']}"
        )

    lines.extend(["", "Stratum recap:", ""])
    for row in summary["eval"]["strata"]:
        lines.append(
            f"- `{row['stratum']}`: n={row['n']}, label positive rate={row['label_pos_rate']}, "
            f"model positive rate={row['model_pos_rate']}, high-conf model positives={row['model_conf_gte_085_pos']}"
        )

    lines.extend(
        [
            "",
            "## Planning implications",
            "",
            "- Store the new parser as unit-size plus total-size features. Do not collapse multipacks into a single bucket only; the eval labels accept same per-unit SKU even when A is a multipack and B is a single.",
            "- Keep LLM auto-accept at confidence >= 0.85. On this eval set it has perfect precision but lower recall, so deterministic auto-accept rules are still needed for obvious strong strata.",
            "- Do not auto-accept weak TF-IDF or private-label mid-score candidates. They need stronger deterministic filters or LLM review.",
            "- Batch API should be removed from the near-term plan for this deployment. Use K-candidate prompts, `reasoning_effort='minimal'`, concurrency, and backoff.",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    summary = {
        "parser_alias": analyze_parser_and_aliases(),
        "eval": analyze_eval()[0],
    }
    SUMMARY_JSON.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    SUMMARY_MD.write_text(build_markdown(summary), encoding="utf-8")
    print(f"Wrote {SUMMARY_JSON}")
    print(f"Wrote {SUMMARY_MD}")
    print(f"Wrote {FAILURES_CSV}")


if __name__ == "__main__":
    main()

