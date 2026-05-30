from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from eda_utils import (
    DATASETS,
    OUTPUT_DIR,
    add_matching_features,
    ensure_output_dir,
    is_missing,
    load_dataset,
    nonmissing_rate,
    normalize_text,
)


PROFILE_JSON = OUTPUT_DIR / "dataset_profile.json"
PROFILE_MD = OUTPUT_DIR / "dataset_profile.md"
COLUMN_SUMMARY = OUTPUT_DIR / "column_summary.csv"


def safe_value(value: Any) -> Any:
    if isinstance(value, (list, tuple, set, dict)):
        return str(value)
    if pd.isna(value):
        return None
    if isinstance(value, (int, float, str, bool)) or value is None:
        return value
    return str(value)


def value_counts_records(df: pd.DataFrame, label: str, columns: list[str], top_n: int = 25) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    total = len(df)
    for col in columns:
        if col not in df.columns:
            continue
        series = df[col].where(~df[col].map(is_missing))
        counts = series.dropna().astype(str).value_counts().head(top_n)
        for value, count in counts.items():
            records.append(
                {
                    "store": label,
                    "field": col,
                    "value": value,
                    "count": int(count),
                    "share": round(float(count) / total, 6) if total else 0.0,
                }
            )
    return records


def column_summary_records(df: pd.DataFrame, label: str) -> list[dict[str, Any]]:
    records = []
    for col in df.columns:
        series = df[col]
        missing = int(series.map(is_missing).sum())
        nonmissing = len(series) - missing
        records.append(
            {
                "store": label,
                "column": col,
                "nonmissing": nonmissing,
                "missing": missing,
                "missing_pct": round(missing / len(series), 6) if len(series) else 0.0,
                "unique_nonmissing": int(series.dropna().astype(str).nunique()),
                "example": next((safe_value(x) for x in series if not is_missing(x)), None),
            }
        )
    return records


def text_length_summary(df: pd.DataFrame, columns: list[str]) -> dict[str, dict[str, float]]:
    output: dict[str, dict[str, float]] = {}
    for col in columns:
        if col not in df.columns:
            continue
        lengths = df[col].where(~df[col].map(is_missing)).dropna().astype(str).str.len()
        if lengths.empty:
            continue
        output[col] = {
            "mean": round(float(lengths.mean()), 2),
            "p50": round(float(lengths.quantile(0.5)), 2),
            "p90": round(float(lengths.quantile(0.9)), 2),
            "p99": round(float(lengths.quantile(0.99)), 2),
        }
    return output


def build_markdown(profile: dict[str, Any]) -> str:
    lines = ["# Dataset Profile", ""]
    for label, details in profile["stores"].items():
        lines.extend(
            [
                f"## Store {label}",
                "",
                f"- Source: `{DATASETS[label].name}`",
                f"- Rows: {details['rows']:,}",
                f"- Columns: {details['columns']}",
                f"- Unique item_id: {details['unique_item_id']:,}",
                f"- Duplicate item_id rows: {details['duplicate_item_id_rows']:,}",
                f"- Nonmissing brand_raw: {details['nonmissing_rates'].get('brand_raw', 0):.1%}",
                f"- Nonmissing parsed size: {details['nonmissing_rates'].get('size_bucket', 0):.1%}",
                f"- Inferred private label rows: {details['inferred_private_label_rows']:,}",
                "",
                "Top parsed departments/categories:",
                "",
            ]
        )
        top_cats = [
            row
            for row in profile["top_values"]
            if row["store"] == label and row["field"] in {"info_category_0", "info_category_1"}
        ][:20]
        for row in top_cats:
            lines.append(f"- `{row['field']}` = {row['value']}: {row['count']:,} ({row['share']:.1%})")
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    ensure_output_dir()
    profile: dict[str, Any] = {"stores": {}, "top_values": []}
    column_records: list[dict[str, Any]] = []

    for label in DATASETS:
        print(f"Loading store {label}")
        raw = load_dataset(label)
        df = add_matching_features(label, raw)

        sample_cols = [
            "item_id",
            "name",
            "brand_raw",
            "name_clean",
            "info_category_0",
            "info_category_1",
            "info_category_2",
            "info_category_3",
            "sizing_size_user_friendly",
            "size_bucket",
            "is_private_label_inferred",
            "url",
        ]
        existing_sample_cols = [col for col in sample_cols if col in df.columns]
        df[existing_sample_cols].sample(min(50, len(df)), random_state=42).to_csv(
            OUTPUT_DIR / f"sample_records_{label}.csv", index=False
        )

        top_columns = [
            "brand_raw",
            "brand_norm",
            "info_category_0",
            "info_category_1",
            "info_category_2",
            "info_category_3",
            "sizing_size_user_friendly",
            "size_bucket",
            "is_private_label",
            "is_private_label_inferred",
            "item_type",
            "is_organic",
        ]
        profile["top_values"].extend(value_counts_records(df, label, top_columns))
        column_records.extend(column_summary_records(df, label))

        nonmissing_cols = [
            "brand_raw",
            "name_clean",
            "description",
            "info_category_0",
            "info_category_1",
            "info_category_2",
            "info_category_3",
            "sizing_size_user_friendly",
            "size_bucket",
            "url",
        ]
        nonmissing_rates = {
            col: nonmissing_rate(df[col]) for col in nonmissing_cols if col in df.columns
        }

        normalized_brand_count = int((df["brand_norm"] != "").sum())
        profile["stores"][label] = {
            "rows": int(len(df)),
            "columns": int(raw.shape[1]),
            "column_names": list(raw.columns),
            "unique_item_id": int(df["item_id"].nunique()) if "item_id" in df.columns else 0,
            "duplicate_item_id_rows": int(df.duplicated("item_id").sum()) if "item_id" in df.columns else 0,
            "unique_name": int(df["name"].dropna().nunique()) if "name" in df.columns else 0,
            "unique_name_clean": int(df["name_clean"].dropna().nunique()) if "name_clean" in df.columns else 0,
            "normalized_brand_count": normalized_brand_count,
            "inferred_private_label_rows": int(df["is_private_label_inferred"].sum()),
            "nonmissing_rates": {k: round(float(v), 6) for k, v in nonmissing_rates.items()},
            "text_lengths": text_length_summary(df, ["name", "name_clean", "description"]),
        }

        df[["item_id", "name", "brand_norm", "name_norm", "size_bucket"]].head(0)

    pd.DataFrame(column_records).to_csv(COLUMN_SUMMARY, index=False)
    pd.DataFrame(profile["top_values"]).to_csv(OUTPUT_DIR / "top_values.csv", index=False)
    PROFILE_JSON.write_text(json.dumps(profile, indent=2), encoding="utf-8")
    PROFILE_MD.write_text(build_markdown(profile), encoding="utf-8")

    print(f"Wrote {PROFILE_JSON}")
    print(f"Wrote {PROFILE_MD}")
    print(f"Wrote {COLUMN_SUMMARY}")


if __name__ == "__main__":
    main()
