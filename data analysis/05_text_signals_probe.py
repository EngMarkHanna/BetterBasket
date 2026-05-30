"""Probe textual signals the prior agent did not mine: descriptions,
URL slugs, direct UPC token overlap, and item_info storage/packaging hints.

Outputs:
- outputs/text_signals_probe.md
- outputs/text_signals_probe.json
- outputs/upc_overlap_examples.csv
- outputs/url_slug_examples.csv
"""
from __future__ import annotations

import json
import re
from collections import Counter
from typing import Any
from urllib.parse import unquote, urlparse

import pandas as pd

from eda_utils import (
    OUTPUT_DIR,
    add_matching_features,
    ensure_output_dir,
    is_missing,
    load_dataset,
    normalize_text,
    parse_mapping,
)


SUMMARY_MD = OUTPUT_DIR / "text_signals_probe.md"
SUMMARY_JSON = OUTPUT_DIR / "text_signals_probe.json"
UPC_EXAMPLES = OUTPUT_DIR / "upc_overlap_examples.csv"
URL_SLUG_EXAMPLES = OUTPUT_DIR / "url_slug_examples.csv"

UPC_RE = re.compile(r"\b(\d{12,14})\b")
SLUG_DROP = {"ip", "shop", "product"}


def extract_upcs(text: Any) -> list[str]:
    if is_missing(text):
        return []
    return UPC_RE.findall(str(text))


def normalize_upc(code: str) -> str:
    """Drop leading zeros so 12, 13, 14-digit variants of the same code collapse."""
    digits = code.lstrip("0") or "0"
    return digits


def slug_from_url(url: Any) -> str:
    if is_missing(url):
        return ""
    try:
        parts = [seg for seg in urlparse(str(url)).path.split("/") if seg]
    except Exception:
        return ""
    if not parts:
        return ""
    # Walmart: /ip/<slug>/<numeric_id>
    # Wegmans: /shop/product/<numeric_id-slug>
    cleaned: list[str] = []
    for seg in parts:
        if seg in SLUG_DROP:
            continue
        # Drop leading numeric id-prefix like "940814-Strawberry-..."
        seg = re.sub(r"^\d+[-_]", "", seg)
        # Drop pure numeric segments
        if seg.isdigit():
            continue
        cleaned.append(seg)
    if not cleaned:
        return ""
    return unquote("-".join(cleaned))


def slug_tokens(slug: str) -> set[str]:
    return {tok for tok in normalize_text(slug.replace("-", " ")).split() if len(tok) > 1}


def name_tokens(text: Any) -> set[str]:
    return {tok for tok in normalize_text(text).split() if len(tok) > 1}


def slug_minus_name_tokens(a: pd.DataFrame) -> dict[str, float]:
    """How often does the URL slug add tokens beyond the name field?"""
    if "url" not in a.columns or "name" not in a.columns:
        return {}
    sample = a[~a["url"].map(is_missing) & ~a["name"].map(is_missing)].sample(
        min(5000, len(a)), random_state=11
    )
    extras: list[int] = []
    for _, row in sample.iterrows():
        n = name_tokens(row["name"])
        s = slug_tokens(slug_from_url(row["url"]))
        extras.append(len(s - n))
    return {
        "sample_rows": int(len(sample)),
        "mean_extra_tokens": round(float(pd.Series(extras).mean()), 3),
        "p50_extra_tokens": round(float(pd.Series(extras).quantile(0.5)), 3),
        "p90_extra_tokens": round(float(pd.Series(extras).quantile(0.9)), 3),
        "rows_with_extra_token": int((pd.Series(extras) > 0).sum()),
    }


def item_info_field_coverage(df: pd.DataFrame, fields: list[str]) -> dict[str, float]:
    if "item_info" not in df.columns:
        return {}
    parsed = df["item_info"].map(parse_mapping)
    out: dict[str, float] = {}
    n = len(df)
    for f in fields:
        cov = parsed.map(lambda obj, k=f: not is_missing(obj.get(k))).mean()
        out[f] = round(float(cov), 4)
    return out


def description_signal_stats(df: pd.DataFrame, label: str) -> dict[str, Any]:
    if "description" not in df.columns:
        return {}
    s = df["description"].dropna().astype(str)
    n = len(df)
    has_upc = s.str.contains(UPC_RE, regex=True).sum()
    has_size = s.str.contains(
        r"\b\d+(?:\.\d+)?\s*(?:oz|fl\.?\s*oz|ml|l|g|kg|lb|ct|count|pack)\b",
        regex=True,
        case=False,
    ).sum()
    has_brand_kw = s.str.contains(r"\b(?:brand|by [A-Z])", regex=True).sum()
    lengths = s.str.len()
    return {
        "store": label,
        "rows_total": int(n),
        "rows_nonmissing_description": int(len(s)),
        "coverage": round(len(s) / n, 4),
        "rows_with_upc_token": int(has_upc),
        "rows_with_size_token": int(has_size),
        "rows_with_brand_keyword": int(has_brand_kw),
        "len_p50": int(lengths.quantile(0.5)),
        "len_p90": int(lengths.quantile(0.9)),
        "len_p99": int(lengths.quantile(0.99)),
    }


def upc_overlap(a: pd.DataFrame, b: pd.DataFrame) -> dict[str, Any]:
    """Direct UPC token overlap across name + description + item_info."""
    def row_upcs(row: pd.Series) -> set[str]:
        codes: set[str] = set()
        for col in ["name", "description", "item_info", "tags", "url"]:
            for c in extract_upcs(row.get(col)):
                codes.add(normalize_upc(c))
        return codes

    a_codes_per_row = a.apply(row_upcs, axis=1)
    b_codes_per_row = b.apply(row_upcs, axis=1)

    a_index: dict[str, list[Any]] = {}
    for idx, codes in a_codes_per_row.items():
        for c in codes:
            a_index.setdefault(c, []).append(idx)

    matches: list[dict[str, Any]] = []
    for idx, codes in b_codes_per_row.items():
        for c in codes:
            if c in a_index:
                for aidx in a_index[c]:
                    matches.append(
                        {
                            "upc": c,
                            "item_id_A": a.at[aidx, "item_id"],
                            "name_A": a.at[aidx, "name"],
                            "brand_A": a.at[aidx, "brand_raw"],
                            "item_id_B": b.at[idx, "item_id"],
                            "name_B": b.at[idx, "name"],
                            "brand_B": b.at[idx, "brand_raw"],
                        }
                    )

    df_matches = pd.DataFrame(matches)
    return {
        "a_rows_with_any_upc": int((a_codes_per_row.map(len) > 0).sum()),
        "b_rows_with_any_upc": int((b_codes_per_row.map(len) > 0).sum()),
        "unique_a_upcs": len(a_index),
        "unique_shared_upcs": int(df_matches["upc"].nunique()) if not df_matches.empty else 0,
        "match_pair_rows": int(len(df_matches)),
        "match_unique_item_a": int(df_matches["item_id_A"].nunique()) if not df_matches.empty else 0,
        "match_unique_item_b": int(df_matches["item_id_B"].nunique()) if not df_matches.empty else 0,
        "examples_csv": str(UPC_EXAMPLES),
        "_df": df_matches,
    }


def url_slug_overlap_sample(a: pd.DataFrame, b: pd.DataFrame, k: int = 200) -> pd.DataFrame:
    """For k random A rows, find B rows whose slug tokens have highest overlap."""
    b_slugs = b[~b["url"].map(is_missing)].copy()
    b_slugs["slug"] = b_slugs["url"].map(slug_from_url)
    b_slugs = b_slugs[b_slugs["slug"].astype(bool)]
    b_slugs["slug_tokens"] = b_slugs["slug"].map(slug_tokens)

    sample = a[~a["url"].map(is_missing) & (a["brand_norm"] != "")].sample(
        min(k, len(a)), random_state=17
    )
    examples = []
    # restrict B candidates per A row by shared brand to keep this tractable
    b_by_brand = {brand: part for brand, part in b_slugs.groupby("brand_norm")}
    for _, arow in sample.iterrows():
        a_slug = slug_from_url(arow["url"])
        a_tok = slug_tokens(a_slug)
        if not a_tok:
            continue
        pool = b_by_brand.get(arow["brand_norm"])
        if pool is None or pool.empty:
            continue
        best_idx = -1
        best_score = 0.0
        for idx, brow in pool.iterrows():
            inter = len(a_tok & brow["slug_tokens"])
            union = len(a_tok | brow["slug_tokens"])
            score = inter / union if union else 0.0
            if score > best_score:
                best_score = score
                best_idx = idx
        if best_idx >= 0 and best_score > 0:
            brow = pool.loc[best_idx]
            examples.append(
                {
                    "item_id_A": arow["item_id"],
                    "name_A": arow["name"],
                    "slug_A": a_slug,
                    "item_id_B": brow["item_id"],
                    "name_B": brow["name"],
                    "slug_B": brow["slug"],
                    "slug_jaccard": round(best_score, 4),
                }
            )
    return pd.DataFrame(examples).sort_values("slug_jaccard", ascending=False)


def build_markdown(summary: dict[str, Any]) -> str:
    lines = ["# Text Signals Probe", ""]
    lines.append("## Description coverage and content")
    for stats in summary["descriptions"]:
        if not stats:
            continue
        lines.extend(
            [
                "",
                f"Store {stats['store']}:",
                f"- Description coverage: {stats['rows_nonmissing_description']:,} / "
                f"{stats['rows_total']:,} ({stats['coverage']:.1%})",
                f"- Rows with UPC-like token in description: {stats['rows_with_upc_token']:,}",
                f"- Rows with size-like token in description: {stats['rows_with_size_token']:,}",
                f"- Description length p50/p90/p99: "
                f"{stats['len_p50']} / {stats['len_p90']} / {stats['len_p99']} chars",
            ]
        )

    lines.extend(["", "## UPC token cross-store overlap"])
    upc = summary["upc_overlap"]
    lines.extend(
        [
            f"- A rows with any 12-14 digit token across name+desc+item_info+tags+url: {upc['a_rows_with_any_upc']:,}",
            f"- B rows with any 12-14 digit token: {upc['b_rows_with_any_upc']:,}",
            f"- Unique normalized UPC codes in A: {upc['unique_a_upcs']:,}",
            f"- Unique normalized UPC codes shared with B: {upc['unique_shared_upcs']:,}",
            f"- Cross-store UPC-linked pairs: {upc['match_pair_rows']:,}",
            f"- Unique A items linked via UPC: {upc['match_unique_item_a']:,}",
            f"- Unique B items linked via UPC: {upc['match_unique_item_b']:,}",
        ]
    )

    lines.extend(["", "## URL slug informativeness (Store A sample)"])
    s = summary["slug_extra_tokens_A"]
    if s:
        lines.extend(
            [
                f"- Sample rows: {s['sample_rows']:,}",
                f"- Mean extra tokens in slug beyond `name`: {s['mean_extra_tokens']}",
                f"- p50 / p90 extra: {s['p50_extra_tokens']} / {s['p90_extra_tokens']}",
                f"- Rows where slug adds at least one token: {s['rows_with_extra_token']:,}",
            ]
        )

    lines.extend(["", "## item_info fields beyond categories"])
    for label, cov in summary["item_info_extra_fields"].items():
        lines.append(f"\nStore {label}:")
        for k, v in cov.items():
            lines.append(f"- `{k}` coverage: {v:.1%}")

    return "\n".join(lines)


def main() -> None:
    ensure_output_dir()
    print("Loading datasets")
    a = add_matching_features("A", load_dataset("A"))
    b = add_matching_features("B", load_dataset("B"))

    print("Computing description signal stats")
    desc_stats = [description_signal_stats(a, "A"), description_signal_stats(b, "B")]

    print("Computing direct UPC overlap (this scans full datasets)")
    upc = upc_overlap(a, b)
    df_upc = upc.pop("_df")
    df_upc.to_csv(UPC_EXAMPLES, index=False)

    print("Computing slug informativeness for A")
    slug_extras = slug_minus_name_tokens(a)

    print("Sampling URL-slug overlap pairs")
    df_slug = url_slug_overlap_sample(a, b, k=300)
    df_slug.to_csv(URL_SLUG_EXAMPLES, index=False)

    print("Inspecting extra item_info fields")
    extra_fields = ["storage_type", "packaging_description", "ingredients"]
    item_info_extra = {
        "A": item_info_field_coverage(a, extra_fields),
        "B": item_info_field_coverage(b, extra_fields),
    }

    summary = {
        "descriptions": desc_stats,
        "upc_overlap": upc,
        "slug_extra_tokens_A": slug_extras,
        "item_info_extra_fields": item_info_extra,
        "url_slug_examples_csv": str(URL_SLUG_EXAMPLES),
        "upc_examples_csv": str(UPC_EXAMPLES),
    }
    SUMMARY_JSON.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    SUMMARY_MD.write_text(build_markdown(summary), encoding="utf-8")
    print(f"Wrote {SUMMARY_JSON}")
    print(f"Wrote {SUMMARY_MD}")
    print(f"Wrote {UPC_EXAMPLES}")
    print(f"Wrote {URL_SLUG_EXAMPLES}")


if __name__ == "__main__":
    main()
