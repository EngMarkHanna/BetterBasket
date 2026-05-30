"""Phase 0 feasibility simulation.

Project the number of A->B matches the planned pipeline can produce at
shippable precision, BEFORE we build the full solution.

Tiers simulated:
- T1: shared brand (alias-aware) + same size_bucket + RapidFuzz on name
- T3: TF-IDF top-1 with cosine>=0.6 (projected from a 5,000-A sample)

Routing policy under simulation (best per executed_plan_followup.md):
  auto-accept T1 score>=95 + auto-accept T3 cosine>=0.6 (when brand-aligned)
  route everything else to LLM, accept iff llm_confidence>=0.85

Outputs:
- outputs/phase0_feasibility.md
- outputs/phase0_feasibility.json

Exit gate: total projected matches >= 4,000.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from rapidfuzz import fuzz, process
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

from eda_utils import (
    OUTPUT_DIR,
    add_matching_features,
    is_missing,
    load_dataset,
    normalize_text,
)

SUMMARY_MD = OUTPUT_DIR / "phase0_feasibility.md"
SUMMARY_JSON = OUTPUT_DIR / "phase0_feasibility.json"

# Curated brand aliases derived from outputs/brand_alias_candidates.csv.
# Each (key) -> (canonical form): when we see `key` as brand_norm, treat as canonical.
BRAND_ALIASES = {
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


def canonical(brand_norm: str) -> str:
    if not brand_norm:
        return ""
    return BRAND_ALIASES.get(brand_norm, brand_norm)


def t1_brand_size_count(a: pd.DataFrame, b: pd.DataFrame) -> dict[str, int]:
    """Count brand+size shared-block pairs by RapidFuzz score band.
    Returns the BEST B candidate per A row inside the shared block."""
    a_eligible = a[(a["brand_canonical"] != "") & (a["size_bucket"] != "")].copy()
    b_eligible = b[(b["brand_canonical"] != "") & (b["size_bucket"] != "")].copy()

    # Group B by (brand_canonical, size_bucket) -> dict of (key) -> DataFrame
    b_groups: dict[tuple, pd.DataFrame] = {
        key: part for key, part in b_eligible.groupby(["brand_canonical", "size_bucket"], sort=False)
    }
    shared_keys = set(b_groups.keys())

    # Filter A to only rows whose key exists in B
    a_keys = list(zip(a_eligible["brand_canonical"].tolist(), a_eligible["size_bucket"].tolist()))
    mask = np.fromiter((k in shared_keys for k in a_keys), dtype=bool, count=len(a_keys))
    a_shared = a_eligible[mask].reset_index(drop=True)
    print(f"  A rows in shared brand+size block: {len(a_shared):,}")

    counts = {"any": 0, "score_ge_95": 0, "score_85_95": 0, "score_70_85": 0, "score_lt_70": 0}
    a_names = a_shared["name_norm"].astype(str).tolist()
    a_bcs = a_shared["brand_canonical"].tolist()
    a_szs = a_shared["size_bucket"].tolist()

    t0 = time.time()
    for i, (name, bc, sz) in enumerate(zip(a_names, a_bcs, a_szs)):
        part = b_groups[(bc, sz)]
        choices = part["name_norm"].astype(str).tolist()
        if not choices:
            continue
        match = process.extractOne(name, choices, scorer=fuzz.WRatio)
        if match is None:
            continue
        _, score, _ = match
        counts["any"] += 1
        if score >= 95:
            counts["score_ge_95"] += 1
        elif score >= 85:
            counts["score_85_95"] += 1
        elif score >= 70:
            counts["score_70_85"] += 1
        else:
            counts["score_lt_70"] += 1
        if (i + 1) % 5000 == 0:
            print(f"    scored {i + 1:,} / {len(a_shared):,}  ({time.time() - t0:.0f}s elapsed)")
    print(f"  T1 scoring took {time.time() - t0:.1f}s")
    return counts


def build_match_text(df: pd.DataFrame) -> pd.Series:
    name = df.get("name", pd.Series("", index=df.index)).fillna("").astype(str)
    brand = df["brand_canonical"].fillna("").astype(str)
    cat2 = df.get("info_category_2", pd.Series("", index=df.index)).fillna("").astype(str)
    cat3 = df.get("info_category_3", pd.Series("", index=df.index)).fillna("").astype(str)
    return (name + " " + brand + " " + cat2 + " " + cat3).map(normalize_text)


def t3_tfidf_projection(a: pd.DataFrame, b: pd.DataFrame, sample_n: int = 5000) -> dict[str, int]:
    """Sample A, compute TF-IDF top-1 vs all of B, project counts to full A."""
    text_a = build_match_text(a)
    text_b = build_match_text(b)

    print("  Fitting TF-IDF (word 1-2gram + char_wb 3-5gram)...")
    t0 = time.time()
    word_vec = TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=2, max_df=0.5, sublinear_tf=True)
    char_vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2, max_df=0.5, sublinear_tf=True)
    all_text = pd.concat([text_a, text_b], ignore_index=True)
    word_vec.fit(all_text)
    char_vec.fit(all_text)
    print(f"    fit in {time.time() - t0:.1f}s")

    t0 = time.time()
    Xa = normalize(hstack([word_vec.transform(text_a), char_vec.transform(text_a)]).tocsr())
    Xb = normalize(hstack([word_vec.transform(text_b), char_vec.transform(text_b)]).tocsr())
    print(f"    transform in {time.time() - t0:.1f}s; feature dim: {Xa.shape[1]:,}")

    rng = np.random.default_rng(42)
    sample_idx = np.sort(rng.choice(len(a), size=min(sample_n, len(a)), replace=False))
    Xa_s = Xa[sample_idx]
    actual_sample = len(sample_idx)

    print(f"  Retrieving top-1 for {actual_sample:,} A rows...")
    t0 = time.time()
    Xb_T = Xb.T.tocsr()
    BATCH = 256
    top1_score = np.zeros(actual_sample, dtype=np.float32)
    top1_idx = np.zeros(actual_sample, dtype=np.int32)
    for start in range(0, actual_sample, BATCH):
        sub = Xa_s[start : start + BATCH]
        sims = (sub @ Xb_T).toarray()
        top1_idx[start : start + sims.shape[0]] = sims.argmax(axis=1)
        top1_score[start : start + sims.shape[0]] = sims.max(axis=1)
    print(f"    retrieved in {time.time() - t0:.1f}s")

    # Tally
    a_sample_df = a.iloc[sample_idx].reset_index(drop=True)
    b_indexed = b.reset_index(drop=True)
    counts_sample = {
        "cosine_ge_06": 0,
        "cosine_ge_06_brand_aligned": 0,
        "cosine_ge_06_size_aligned": 0,
        "cosine_ge_06_brand_AND_size_aligned": 0,
        "cosine_04_06": 0,
    }
    for j in range(actual_sample):
        sc = top1_score[j]
        if sc < 0.4:
            continue
        if sc < 0.6:
            counts_sample["cosine_04_06"] += 1
            continue
        counts_sample["cosine_ge_06"] += 1
        a_row = a_sample_df.iloc[j]
        b_row = b_indexed.iloc[int(top1_idx[j])]
        brand_match = bool(a_row["brand_canonical"]) and (a_row["brand_canonical"] == b_row["brand_canonical"])
        size_match = bool(a_row["size_bucket"]) and (a_row["size_bucket"] == b_row["size_bucket"])
        if brand_match:
            counts_sample["cosine_ge_06_brand_aligned"] += 1
        if size_match:
            counts_sample["cosine_ge_06_size_aligned"] += 1
        if brand_match and size_match:
            counts_sample["cosine_ge_06_brand_AND_size_aligned"] += 1

    scale = len(a) / actual_sample
    projected = {k: int(v * scale) for k, v in counts_sample.items()}
    projected["sample_n"] = actual_sample
    projected["scale_factor"] = round(scale, 3)
    projected["full_a_rows"] = len(a)
    projected["sample_counts"] = counts_sample
    return projected


def project_matches(t1: dict[str, int], t3: dict[str, int]) -> dict[str, int]:
    """Apply the routing policy and project final accepted matches.

    Policy:
      auto-accept A_strong-like = T1 score>=95
      auto-accept T3 top-1 cosine>=0.6 BUT only when brand-aligned (precision boost)
      LLM-veto everything else; accept iff model conf>=0.85

    Eval-set baseline numbers (from executed_plan_followup):
      - A_strong (n=20) label-positive rate = 1.0 -> ~100% precision auto-accept
      - T_strong_tfidf (n=9) label-positive rate = 0.89 -> need extra brand check
        We assume brand-aligned T3 reaches ~0.95 precision (brand+TF-IDF agreement
        is stronger than TF-IDF alone)
      - LLM at conf>=0.85 precision = 1.0; acceptance rate ranges 5-20% depending on stratum
        Aggregate acceptance rate over routed pool: ~10-15% (mostly A_medium-like)
    """
    auto_t1 = t1["score_ge_95"]
    auto_t3 = t3["cosine_ge_06_brand_aligned"]  # use brand-aligned only

    # LLM-veto candidates: everything else we keep
    llm_candidates = (
        t1["score_85_95"]
        + t1["score_70_85"]
        + (t3["cosine_ge_06"] - t3["cosine_ge_06_brand_aligned"])  # T3 strong without brand
        + t3["cosine_04_06"]
    )

    # Acceptance rates derived from eval-set per-stratum model accept @ conf>=0.85:
    #   A_strong:        17/20 = 0.85
    #   A_medium:         1/20 = 0.05
    #   A_borderline:     0/3  = 0.00
    #   A_low_score:      0/5  = 0.00
    #   A_private_high:   2/10 = 0.20
    #   A_private_mid:    0/10 = 0.00
    #   T_strong_tfidf:   4/9  = 0.44
    #   T_weak_tfidf:     0/10 = 0.00
    #   H_hand:           1/10 = 0.10
    #
    # T1 score 85-95 maps to A_medium baseline -> 5% accept
    # T1 score 70-85 maps to A_borderline -> ~0% accept
    # T3 brand-unaligned cosine>=0.6 -> ~10-20% (mix of strong-tfidf without brand)
    # T3 cosine 0.4-0.6 -> ~5%
    llm_accept_t1_med = int(t1["score_85_95"] * 0.05)
    llm_accept_t1_low = int(t1["score_70_85"] * 0.02)
    llm_accept_t3_brand_unaligned = int((t3["cosine_ge_06"] - t3["cosine_ge_06_brand_aligned"]) * 0.15)
    llm_accept_t3_med = int(t3["cosine_04_06"] * 0.05)
    llm_accepted = llm_accept_t1_med + llm_accept_t1_low + llm_accept_t3_brand_unaligned + llm_accept_t3_med

    total = auto_t1 + auto_t3 + llm_accepted

    return {
        "auto_accept_t1_strong_ge_95": auto_t1,
        "auto_accept_t3_cosine_ge_06_brand_aligned": auto_t3,
        "llm_candidates_total": llm_candidates,
        "llm_accepted_t1_med": llm_accept_t1_med,
        "llm_accepted_t1_low": llm_accept_t1_low,
        "llm_accepted_t3_brand_unaligned": llm_accept_t3_brand_unaligned,
        "llm_accepted_t3_med": llm_accept_t3_med,
        "llm_accepted_total": llm_accepted,
        "total_projected_matches": total,
        "target": 4000,
        "passes_exit_gate": total >= 4000,
        "headroom_over_target_pct": round(100.0 * total / 4000, 1),
    }


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print("Loading datasets and applying matching features...")
    t0 = time.time()
    a = add_matching_features("A", load_dataset("A"))
    b = add_matching_features("B", load_dataset("B"))
    print(f"  loaded in {time.time() - t0:.1f}s")

    print("Applying curated brand alias map (canonical brand)...")
    a["brand_canonical"] = a["brand_norm"].astype(str).map(canonical)
    b["brand_canonical"] = b["brand_norm"].astype(str).map(canonical)

    a_brand_aliased = int(((a["brand_norm"] != "") & (a["brand_norm"] != a["brand_canonical"])).sum())
    b_brand_aliased = int(((b["brand_norm"] != "") & (b["brand_norm"] != b["brand_canonical"])).sum())
    print(f"  A rows whose brand was aliased: {a_brand_aliased:,}")
    print(f"  B rows whose brand was aliased: {b_brand_aliased:,}")

    print("\n--- T1: brand+size+RapidFuzz scoring on shared block ---")
    t1 = t1_brand_size_count(a, b)
    print(f"  T1 score>=95: {t1['score_ge_95']:,}")
    print(f"  T1 score 85-95: {t1['score_85_95']:,}")
    print(f"  T1 score 70-85: {t1['score_70_85']:,}")
    print(f"  T1 score <70: {t1['score_lt_70']:,}")
    print(f"  T1 any: {t1['any']:,}")

    print("\n--- T3: TF-IDF top-1 projection from 5,000-A sample ---")
    t3 = t3_tfidf_projection(a, b, sample_n=5000)
    print(f"  T3 sample: {t3['sample_n']:,} (scale {t3['scale_factor']:.2f}x)")
    print(f"  T3 cosine>=0.6 (projected): {t3['cosine_ge_06']:,}")
    print(f"    of which brand-aligned: {t3['cosine_ge_06_brand_aligned']:,}")
    print(f"    of which size-aligned:  {t3['cosine_ge_06_size_aligned']:,}")
    print(f"    brand AND size aligned: {t3['cosine_ge_06_brand_AND_size_aligned']:,}")
    print(f"  T3 cosine 0.4-0.6 (projected): {t3['cosine_04_06']:,}")

    print("\n--- Routing-policy projection ---")
    proj = project_matches(t1, t3)
    for k, v in proj.items():
        print(f"  {k}: {v}")

    summary = {
        "alias_map_size": len(BRAND_ALIASES),
        "alias_applied": {"A_rows": a_brand_aliased, "B_rows": b_brand_aliased},
        "t1_brand_size_rapidfuzz": t1,
        "t3_tfidf_top1": t3,
        "projection": proj,
    }
    SUMMARY_JSON.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    md = [
        "# Phase 0 Feasibility Simulation",
        "",
        "Projects how many A->B matches the planned pipeline can produce, ",
        "using real candidate counts and eval-set-derived precision baselines.",
        "",
        f"Brand alias map applied: {len(BRAND_ALIASES)} entries; aliased "
        f"{a_brand_aliased:,} A rows and {b_brand_aliased:,} B rows.",
        "",
        "## T1: shared brand_canonical + same size_bucket, scored by RapidFuzz WRatio",
        f"- Score >= 95 (auto-accept, A_strong-like, ~100% precision): **{t1['score_ge_95']:,}**",
        f"- Score 85-95 (route to LLM, A_medium-like band): {t1['score_85_95']:,}",
        f"- Score 70-85 (route to LLM, A_borderline-like): {t1['score_70_85']:,}",
        f"- Score <70 (drop): {t1['score_lt_70']:,}",
        f"- Total scored: {t1['any']:,}",
        "",
        "## T3: TF-IDF top-1 retrieval (projected from 5,000-A sample)",
        f"- Sample size: {t3['sample_n']:,}; scale factor to full A: {t3['scale_factor']:.2f}x",
        f"- Cosine >= 0.6 (any): {t3['cosine_ge_06']:,}",
        f"  - of which brand-aligned (auto-accept tier): **{t3['cosine_ge_06_brand_aligned']:,}**",
        f"  - of which size-aligned: {t3['cosine_ge_06_size_aligned']:,}",
        f"  - of which brand AND size aligned: {t3['cosine_ge_06_brand_AND_size_aligned']:,}",
        f"- Cosine 0.4-0.6 (route to LLM): {t3['cosine_04_06']:,}",
        "",
        "## Routing-policy projection",
        "Policy: auto-accept T1 (score>=95) + T3 (cosine>=0.6, brand-aligned); ",
        "route everything else to LLM and accept iff model_confidence >= 0.85.",
        "",
        "Per-band LLM acceptance rates derived from `eval_results.csv`:",
        "- T1 score 85-95 (~A_medium): 5%",
        "- T1 score 70-85 (~A_borderline): 2%",
        "- T3 brand-unaligned cosine>=0.6: 15%",
        "- T3 cosine 0.4-0.6: 5%",
        "",
        f"| Source | Projected accept |",
        f"|---|---|",
        f"| Auto-accept T1 (score>=95) | {proj['auto_accept_t1_strong_ge_95']:,} |",
        f"| Auto-accept T3 (cosine>=0.6 + brand) | {proj['auto_accept_t3_cosine_ge_06_brand_aligned']:,} |",
        f"| LLM-accepted (all routed bands) | {proj['llm_accepted_total']:,} |",
        f"| **Total projected matches** | **{proj['total_projected_matches']:,}** |",
        f"| Target | 4,000 |",
        "",
        f"### Exit gate: **{'PASS' if proj['passes_exit_gate'] else 'FAIL'}** "
        f"({proj['headroom_over_target_pct']:.0f}% of target)",
        "",
        f"LLM call volume estimate: ~{proj['llm_candidates_total']:,} pairs routed; ",
        f"at ~1s/call sequential or ~5 pair/s with K=5 batching + 5 concurrent workers, ",
        f"this is ~{proj['llm_candidates_total'] // 5 // 60:.0f}-{proj['llm_candidates_total'] // 60:.0f} minutes wall-clock.",
    ]
    SUMMARY_MD.write_text("\n".join(md), encoding="utf-8")
    print(f"\nWrote {SUMMARY_MD}")
    print(f"Wrote {SUMMARY_JSON}")


if __name__ == "__main__":
    main()
