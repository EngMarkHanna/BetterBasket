"""Build a stratified candidate pool for the gold/silver eval set.

Pulls from three existing example CSVs and emits a single
`eval_candidates.csv` that I (the labeling agent) will read and label by
adding a `label_match_type` column. Stratification gives us calibration
data per confidence band.

Strata:
- A_strong       : high_confidence_examples national block, score >= 95
- A_medium       : high_confidence_examples national block, score 85-95
- A_borderline   : high_confidence_examples national block, score 80-85
- A_private      : high_confidence_examples private block, score >= 80
- T_strong_tfidf : tfidf_retrieval_examples rank=1, score >= 0.6
- T_weak_tfidf   : tfidf_retrieval_examples rank=1, score 0.4-0.6
- H_hand         : hand-constructed edge cases (cross-domain, flavor, size variants)

Outputs:
- outputs/eval_candidates.csv (waiting to be labeled)
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"
HIGH_CONF_CSV = OUTPUT_DIR / "high_confidence_examples.csv"
CANDIDATE_CSV = OUTPUT_DIR / "candidate_examples.csv"
TFIDF_CSV = OUTPUT_DIR / "tfidf_retrieval_examples.csv"
EVAL_CSV = OUTPUT_DIR / "eval_candidates.csv"

RANDOM_SEED = 19


def sample_high_conf() -> list[pd.DataFrame]:
    hc = pd.read_csv(HIGH_CONF_CSV)
    natl = hc[hc["block_cols"] == "brand_norm+size_bucket"]
    priv = hc[hc["block_cols"] == "size_bucket"]

    rng = np.random.default_rng(RANDOM_SEED)

    def take(df: pd.DataFrame, n: int, stratum: str) -> pd.DataFrame:
        if df.empty:
            return df
        n = min(n, len(df))
        idx = rng.choice(df.index, size=n, replace=False)
        out = df.loc[idx].copy()
        out["stratum"] = stratum
        out["source_score"] = out["score"]
        out["source_signal"] = "rapidfuzz_wratio"
        return out

    # high_confidence_examples was top-300 per block, so all rows are
    # already at the top of the score distribution; the medium/borderline
    # bands come from candidate_examples.csv instead.
    strong = take(natl[natl["score"] >= 95], 20, "A_strong")
    private_high = take(priv[priv["score"] >= 90], 10, "A_private_high")
    private_mid = take(priv[(priv["score"] >= 85) & (priv["score"] < 90)], 10, "A_private_mid")
    return [strong, private_high, private_mid]


def sample_candidate_examples() -> list[pd.DataFrame]:
    ce = pd.read_csv(CANDIDATE_CSV)
    rng = np.random.default_rng(RANDOM_SEED + 2)

    def take(df: pd.DataFrame, n: int, stratum: str) -> pd.DataFrame:
        if df.empty:
            return df
        n = min(n, len(df))
        idx = rng.choice(df.index, size=n, replace=False)
        out = df.loc[idx].copy()
        out["stratum"] = stratum
        out["source_score"] = out["rapidfuzz_wratio"]
        out["source_signal"] = "rapidfuzz_wratio_brand_block"
        return out

    medium = take(ce[(ce["rapidfuzz_wratio"] >= 85) & (ce["rapidfuzz_wratio"] < 95)], 20, "A_medium")
    borderline = take(ce[(ce["rapidfuzz_wratio"] >= 75) & (ce["rapidfuzz_wratio"] < 85)], 3, "A_borderline")
    low = take(ce[ce["rapidfuzz_wratio"] < 75], 5, "A_low_score")
    return [medium, borderline, low]


def sample_tfidf() -> list[pd.DataFrame]:
    tf = pd.read_csv(TFIDF_CSV)
    rng = np.random.default_rng(RANDOM_SEED + 1)
    rank1 = tf[tf["rank"] == 1].copy()

    def take(df: pd.DataFrame, n: int, stratum: str) -> pd.DataFrame:
        if df.empty:
            return df
        n = min(n, len(df))
        idx = rng.choice(df.index, size=n, replace=False)
        out = df.loc[idx].copy()
        out["stratum"] = stratum
        out["source_score"] = out["score"]
        out["source_signal"] = "tfidf_cosine"
        return out

    # The TF-IDF csv only emitted rows with top-1 score >= 0.4
    strong_tf = take(rank1[rank1["score"] >= 0.6], 15, "T_strong_tfidf")
    weak_tf = take(rank1[(rank1["score"] >= 0.4) & (rank1["score"] < 0.6)], 10, "T_weak_tfidf")
    return [strong_tf, weak_tf]


def hand_constructed() -> pd.DataFrame:
    """Pairs constructed to stress specific failure modes the model needs to handle."""
    rows = [
        # Cross-domain obvious negatives
        dict(
            item_id_A="hand_cd1",
            name_A="HART 140-Piece 1/4 and 3/8-inch Drive Mechanics Tool Set, Chrome Finish",
            brand_A="HART", size_A="140 pc", cat_A="Home Improvement > Tools > Hand Tools",
            item_id_B="hand_cd1b",
            name_B="Wegmans Honey Ham, Thin Shaved Chipped",
            brand_B="Wegmans", size_B="", cat_B="More Departments > Deli > Ham",
            stratum="H_hand", source_score=0.0, source_signal="manual_cross_domain",
        ),
        dict(
            item_id_A="hand_cd2",
            name_A="Funko POP Movies: MMPR Movie- Red Ranger",
            brand_A="Funko", size_A="", cat_A="Toys > Action Figures",
            item_id_B="hand_cd2b",
            name_B="Wegmans Organic Whole Milk",
            brand_B="Wegmans", size_B="64 fl oz", cat_B="Dairy > Milk",
            stratum="H_hand", source_score=0.0, source_signal="manual_cross_domain",
        ),
        # Flavor variants (should be no_match for pricing)
        dict(
            item_id_A="hand_fv1",
            name_A="M&M's Peanut Butter Chocolate Christmas Candy - 10 oz Bag",
            brand_A="M&M'S", size_A="10 Oz", cat_A="Food > Candy > Chocolate",
            item_id_B="hand_fv1b",
            name_B="M&M'S Milk Chocolate Christmas Candy Bag",
            brand_B="M&M'S", size_B="10 ounce", cat_B="Grocery > Candy > Seasonal & Holiday Candy",
            stratum="H_hand", source_score=88.0, source_signal="manual_flavor_variant",
        ),
        dict(
            item_id_A="hand_fv2",
            name_A="Ocean Spray Cranberry Apple Juice 64 fl oz",
            brand_A="Ocean Spray", size_A="64 fl oz", cat_A="Food > Beverages > Juices",
            item_id_B="hand_fv2b",
            name_B="Ocean Spray Cranberry Pomegranate Juice 64 fl oz",
            brand_B="Ocean Spray", size_B="64 fl. oz.", cat_B="Grocery > Beverages > Bottled Juice",
            stratum="H_hand", source_score=85.0, source_signal="manual_flavor_variant",
        ),
        # Size mismatch (should be no_match - different SKU/price point)
        dict(
            item_id_A="hand_sm1",
            name_A="Q-Tips Cotton Swabs Travel Size Pack 30 Count",
            brand_A="Q-tips", size_A="30 ct", cat_A="Beauty > Cotton Swabs",
            item_id_B="hand_sm1b",
            name_B="Q-Tips Cotton Swabs 500 ct",
            brand_B="Q-Tips", size_B="500 ct.", cat_B="Personal Care > Cotton Swabs",
            stratum="H_hand", source_score=80.0, source_signal="manual_size_variant",
        ),
        # Multipack vs single (interesting case - debatable)
        dict(
            item_id_A="hand_mp1",
            name_A="(3 pack) Betty Crocker Super Moist Chocolate Fudge Cake Mix, 15.25 oz.",
            brand_A="Betty Crocker", size_A="3 x 15.25 oz", cat_A="Food > Baking > Easy to Make",
            item_id_B="hand_mp1b",
            name_B="Betty Crocker Super Moist Chocolate Fudge Cake Mix",
            brand_B="Betty Crocker", size_B="13.25 ounce", cat_B="Grocery > Baking & Baking Ingredients > Baking Mixes",
            stratum="H_hand", source_score=95.0, source_signal="manual_multipack",
        ),
        # Private-label equivalent (should be true)
        dict(
            item_id_A="hand_pl1",
            name_A="Great Value Organic Tomato Sauce 8 oz",
            brand_A="Great Value", size_A="8 oz", cat_A="Food > Pantry > Canned goods",
            item_id_B="hand_pl1b",
            name_B="Wegmans Organic Tomato Sauce 8 oz",
            brand_B="Wegmans", size_B="8 ounce", cat_B="Grocery > Pantry > Canned Tomatoes",
            stratum="H_hand", source_score=70.0, source_signal="manual_private_label",
        ),
        # Private-label vs national (NOT a match - different value tier)
        dict(
            item_id_A="hand_pln1",
            name_A="Great Value Whole Milk 1 Gallon",
            brand_A="Great Value", size_A="1 gal", cat_A="Food > Dairy > Milk",
            item_id_B="hand_pln1b",
            name_B="Organic Valley Whole Milk 1 Gallon",
            brand_B="Organic Valley", size_B="128 fl oz", cat_B="Dairy > Milk",
            stratum="H_hand", source_score=60.0, source_signal="manual_private_vs_national",
        ),
        # Same brand, same product, organic vs not (NOT a match - different SKU)
        dict(
            item_id_A="hand_org1",
            name_A="Heinz Tomato Ketchup 14 oz",
            brand_A="Heinz", size_A="14 oz", cat_A="Food > Pantry > Condiments",
            item_id_B="hand_org1b",
            name_B="Heinz Organic Tomato Ketchup 14 oz",
            brand_B="Heinz", size_B="14 ounce", cat_B="Grocery > Condiments > Ketchup",
            stratum="H_hand", source_score=92.0, source_signal="manual_organic_variant",
        ),
        # Exact match with brand alias
        dict(
            item_id_A="hand_al1",
            name_A="L'Oreal Paris True Match Foundation Natural Beige",
            brand_A="L'Oreal Paris", size_A="1 fl oz", cat_A="Beauty > Makeup > Foundation",
            item_id_B="hand_al1b",
            name_B="L'Oreal True Match Foundation Natural Beige",
            brand_B="L'Oreal", size_B="1 fl. oz.", cat_B="Personal Care > Makeup > Foundation",
            stratum="H_hand", source_score=92.0, source_signal="manual_brand_alias",
        ),
    ]
    return pd.DataFrame(rows)


def main() -> None:
    parts: list[pd.DataFrame] = []
    parts.extend(sample_high_conf())
    parts.extend(sample_candidate_examples())
    parts.extend(sample_tfidf())
    parts.append(hand_constructed())

    # Normalize column names. Some sources call it "cat_A", others "category_A".
    normalized = []
    for df in parts:
        if df.empty:
            continue
        df = df.copy()
        if "cat_A" in df.columns and "category_A" not in df.columns:
            df["category_A"] = df["cat_A"]
        if "cat_B" in df.columns and "category_B" not in df.columns:
            df["category_B"] = df["cat_B"]
        for col in [
            "item_id_A", "name_A", "brand_A", "size_A", "category_A",
            "item_id_B", "name_B", "brand_B", "size_B", "category_B",
            "stratum", "source_score", "source_signal",
        ]:
            if col not in df.columns:
                df[col] = ""
        normalized.append(
            df[
                [
                    "item_id_A", "name_A", "brand_A", "size_A", "category_A",
                    "item_id_B", "name_B", "brand_B", "size_B", "category_B",
                    "stratum", "source_score", "source_signal",
                ]
            ]
        )

    combined = pd.concat(normalized, ignore_index=True)
    combined.insert(0, "pair_id", [f"P{i:04d}" for i in range(len(combined))])

    print("Stratum counts:")
    print(combined["stratum"].value_counts())
    combined.to_csv(EVAL_CSV, index=False)
    print(f"\nWrote {EVAL_CSV}  ({len(combined)} pairs)")


if __name__ == "__main__":
    main()
