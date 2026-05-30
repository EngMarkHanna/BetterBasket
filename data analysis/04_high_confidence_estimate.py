from __future__ import annotations

import json
from collections import defaultdict
from typing import Any

import pandas as pd
from rapidfuzz import fuzz, process

from eda_utils import OUTPUT_DIR, add_matching_features, ensure_output_dir, is_missing, load_dataset, normalize_text


SUMMARY_MD = OUTPUT_DIR / "high_confidence_estimate.md"
SUMMARY_JSON = OUTPUT_DIR / "high_confidence_estimate.json"
EXAMPLES_CSV = OUTPUT_DIR / "high_confidence_examples.csv"

PRIVATE_LABEL_TERMS = [
    "great value",
    "marketside",
    "bettergoods",
    "equate",
    "sam s choice",
    "walmart",
    "wegmans",
    "wegmans organic",
    "wegmans brand",
]


def strip_private_label_terms(text: Any) -> str:
    normalized = normalize_text(text)
    for term in PRIVATE_LABEL_TERMS:
        normalized = normalized.replace(term, " ")
    return " ".join(normalized.split())


def token_set(text: Any) -> set[str]:
    return {token for token in normalize_text(text).split() if len(token) > 1}


def jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def make_key(row: pd.Series, columns: list[str]) -> str:
    values = []
    for col in columns:
        value = row.get(col)
        if is_missing(value):
            return ""
        values.append(str(value))
    return "|".join(values)


def threshold_counts(scored_pairs: list[tuple[float, float]]) -> dict[str, int]:
    return {
        "score_gte_95": sum(score >= 95 for score, _ in scored_pairs),
        "score_gte_90": sum(score >= 90 for score, _ in scored_pairs),
        "score_gte_85": sum(score >= 85 for score, _ in scored_pairs),
        "score_gte_80": sum(score >= 80 for score, _ in scored_pairs),
        "score_gte_90_jaccard_gte_0_50": sum(score >= 90 and jac >= 0.50 for score, jac in scored_pairs),
        "score_gte_85_jaccard_gte_0_50": sum(score >= 85 and jac >= 0.50 for score, jac in scored_pairs),
        "score_gte_85_jaccard_gte_0_35": sum(score >= 85 and jac >= 0.35 for score, jac in scored_pairs),
    }


def best_matches_by_block(
    a: pd.DataFrame,
    b: pd.DataFrame,
    block_cols: list[str],
    query_col: str,
    choice_col: str,
    max_examples: int = 300,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    b_groups: dict[str, dict[str, str]] = defaultdict(dict)
    for _, row in b.iterrows():
        key = make_key(row, block_cols)
        if key:
            b_groups[key][str(row["item_id"])] = str(row[choice_col])

    examples: list[dict[str, Any]] = []
    scored_pairs: list[tuple[float, float]] = []
    a_considered = 0
    shared_block_a_rows = 0
    for _, row in a.iterrows():
        key = make_key(row, block_cols)
        if not key:
            continue
        a_considered += 1
        choices = b_groups.get(key)
        if not choices:
            continue
        shared_block_a_rows += 1
        match = process.extractOne(str(row[query_col]), choices, scorer=fuzz.WRatio)
        if match is None:
            continue
        choice_text, score, b_item_id = match
        jac = jaccard(token_set(row[query_col]), token_set(choice_text))
        scored_pairs.append((score, jac))
        if score >= 80:
            brow = b[b["item_id"].astype(str) == str(b_item_id)].iloc[0]
            examples.append(
                {
                    "block": key,
                    "item_id_A": row["item_id"],
                    "name_A": row["name"],
                    "brand_A": row.get("brand_raw"),
                    "size_A": row.get("sizing_size_user_friendly"),
                    "item_id_B": brow["item_id"],
                    "name_B": brow["name"],
                    "brand_B": brow.get("brand_raw"),
                    "size_B": brow.get("sizing_size_user_friendly"),
                    "score": round(float(score), 4),
                    "token_jaccard": round(float(jac), 4),
                    "block_cols": "+".join(block_cols),
                }
            )

    examples = sorted(examples, key=lambda row: (row["score"], row["token_jaccard"]), reverse=True)[:max_examples]
    stats = {
        "block_cols": block_cols,
        "a_rows_with_complete_key": a_considered,
        "a_rows_with_shared_block": shared_block_a_rows,
        "matches_scored": len(scored_pairs),
        **threshold_counts(scored_pairs),
    }
    return examples, stats


def build_markdown(summary: dict[str, Any]) -> str:
    lines = ["# High Confidence Match Estimate", ""]
    for section in ["national_brand_size", "private_label_size"]:
        stats = summary[section]
        title = "National/shared brand + size" if section == "national_brand_size" else "Private label + size"
        lines.append(f"## {title}")
        lines.extend(
            [
                f"- A rows with complete key: {stats['a_rows_with_complete_key']:,}",
                f"- A rows with shared B block: {stats['a_rows_with_shared_block']:,}",
                f"- Scored candidate best matches: {stats['matches_scored']:,}",
                f"- Score >= 95: {stats['score_gte_95']:,}",
                f"- Score >= 90: {stats['score_gte_90']:,}",
                f"- Score >= 85: {stats['score_gte_85']:,}",
                f"- Score >= 80: {stats['score_gte_80']:,}",
                f"- Score >= 90 and token Jaccard >= 0.50: {stats['score_gte_90_jaccard_gte_0_50']:,}",
                f"- Score >= 85 and token Jaccard >= 0.50: {stats['score_gte_85_jaccard_gte_0_50']:,}",
                f"- Score >= 85 and token Jaccard >= 0.35: {stats['score_gte_85_jaccard_gte_0_35']:,}",
                "",
            ]
        )
    return "\n".join(lines)


def main() -> None:
    ensure_output_dir()
    print("Loading and enriching datasets")
    a = add_matching_features("A", load_dataset("A"))
    b = add_matching_features("B", load_dataset("B"))

    print("Estimating national/shared brand matches")
    national_examples, national_stats = best_matches_by_block(
        a[a["brand_norm"] != ""],
        b[b["brand_norm"] != ""],
        ["brand_norm", "size_bucket"],
        "name_norm",
        "name_norm",
    )

    print("Estimating private-label size-block matches")
    a_private = a[a["is_private_label_inferred"]].copy()
    b_private = b[b["is_private_label_inferred"]].copy()
    a_private["private_query"] = a_private["name"].map(strip_private_label_terms)
    b_private["private_choice"] = b_private["name"].map(strip_private_label_terms)
    private_examples, private_stats = best_matches_by_block(
        a_private,
        b_private,
        ["size_bucket"],
        "private_query",
        "private_choice",
    )

    examples = pd.DataFrame(national_examples + private_examples)
    examples.to_csv(EXAMPLES_CSV, index=False)

    summary = {
        "national_brand_size": national_stats,
        "private_label_size": private_stats,
    }
    SUMMARY_JSON.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    SUMMARY_MD.write_text(build_markdown(summary), encoding="utf-8")
    print(f"Wrote {SUMMARY_JSON}")
    print(f"Wrote {SUMMARY_MD}")
    print(f"Wrote {EXAMPLES_CSV}")


if __name__ == "__main__":
    main()
