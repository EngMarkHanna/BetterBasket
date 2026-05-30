"""Deterministic feature scoring + routing decision per candidate pair.

The scorer attaches a feature dict and a final score in [0, 1] plus
routing tag in {'auto_accept', 'route_to_llm', 'drop'}.

Hard veto flags drop the pair regardless of score.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

import pandas as pd
from rapidfuzz import fuzz

from .parse import normalize_text


CONFLICT_FLAVOR_HINTS = {
    "vanilla",
    "chocolate",
    "strawberry",
    "lemon",
    "lime",
    "orange",
    "apple",
    "grape",
    "cherry",
    "mango",
    "blueberry",
    "raspberry",
    "peach",
    "mint",
    "cinnamon",
    "caramel",
    "honey",
    "maple",
    "coconut",
    "pumpkin",
    "berry",
    "watermelon",
    "tropical",
}


FORM_HINTS = {
    "ground",
    "whole bean",
    "instant",
    "k cup",
    "k cups",
    "pods",
    "liquid",
    "powder",
    "spray",
    "stick",
    "wipes",
    "lotion",
    "cream",
    "shampoo",
    "conditioner",
    "bar",
    "frozen",
    "fresh",
    "dried",
    "chips",
    "crackers",
    "cookies",
}


@dataclass
class ScoredCandidate:
    item_id_a: str
    item_id_b: str
    candidate_source: str
    final_score: float
    route: str  # auto_accept | route_to_llm | drop
    features: dict = field(default_factory=dict)


def _tokens(s: str) -> set[str]:
    return set((s or "").split())


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 0.0
    inter = a & b
    union = a | b
    return len(inter) / len(union) if union else 0.0


def _brand_relation(a_brand: str, b_brand: str, a_canon: str, b_canon: str, a_pl: bool, b_pl: bool) -> str:
    if not a_brand and not b_brand:
        return "unknown"
    if not a_brand or not b_brand:
        return "unknown"
    if a_brand == b_brand:
        return "exact"
    if a_canon and b_canon and a_canon == b_canon:
        return "alias"
    if a_pl and b_pl:
        return "private_label_compatible"
    return "conflict"


def _size_relation(a_canon, b_canon, a_dim, b_dim) -> str:
    if a_canon is None or b_canon is None or pd.isna(a_canon) or pd.isna(b_canon):
        return "unknown"
    if a_dim != b_dim:
        return "conflict_dim"
    a = float(a_canon)
    b = float(b_canon)
    if a == 0 or b == 0:
        return "unknown"
    ratio = a / b
    if 0.92 <= ratio <= 1.08:
        return "same"
    if 0.85 <= ratio <= 1.15:
        return "near"
    # Multipack check
    for r in (ratio, 1 / ratio):
        rr = round(r)
        if 2 <= rr <= 24 and abs(r - rr) < 0.10:
            return "multipack"
    return "off"


def _pack_relation(a_pack, b_pack) -> str:
    if a_pack is None or b_pack is None or pd.isna(a_pack) or pd.isna(b_pack):
        return "unknown"
    a = int(a_pack)
    b = int(b_pack)
    if a == b:
        return "same"
    # Either side could be 1 and the other could be a multipack.
    if (a == 1 and b > 1) or (b == 1 and a > 1):
        return "multipack"
    return "off"


def _has_flavor_conflict(a_name: str, b_name: str) -> bool:
    """Slot-aware flavor veto (audit fix #8).

    Old behaviour: veto when symmetric_difference was non-empty. That
    fired on 'Caramel Chocolate' vs 'Caramel' (one extra descriptor)
    even though both share 'Caramel' - a clear non-conflict.

    New behaviour: only veto when the two sides have NO overlapping
    flavor tokens, i.e. they're talking about different things entirely.
    """
    a_flav = _tokens(a_name) & CONFLICT_FLAVOR_HINTS
    b_flav = _tokens(b_name) & CONFLICT_FLAVOR_HINTS
    if not a_flav or not b_flav:
        return False
    return not (a_flav & b_flav)


def _has_form_conflict(a_name: str, b_name: str) -> bool:
    """Slot-aware form veto (audit fix #8). Same logic as flavor."""
    a_form = _tokens(a_name) & FORM_HINTS
    b_form = _tokens(b_name) & FORM_HINTS
    if not a_form or not b_form:
        return False
    return not (a_form & b_form)


def _organic_relation(a_org: bool, b_org: bool, is_food: bool) -> str:
    if a_org == b_org:
        return "match"
    return "conflict" if is_food else "soft_conflict"


def score_candidates(
    candidates_df: pd.DataFrame,
    a_index: dict[str, dict],
    b_index: dict[str, dict],
) -> pd.DataFrame:
    """Attach features + final_score + route to each candidate row.

    a_index / b_index are dict-of-rows keyed by item_id for O(1) lookup.
    """
    rows = []
    for cand in candidates_df.itertuples(index=False):
        a = a_index.get(cand.item_id_a)
        b = b_index.get(cand.item_id_b)
        if a is None or b is None:
            continue

        # Names + fuzzy + tokens.
        rapid = cand.rapidfuzz_wratio_name if hasattr(cand, "rapidfuzz_wratio_name") and not pd.isna(cand.rapidfuzz_wratio_name) else float(
            fuzz.WRatio(a["name_norm"], b["name_norm"])
        )
        token_jacc = _jaccard(_tokens(a["name_norm"]), _tokens(b["name_norm"]))
        cosine = cand.tfidf_cosine if hasattr(cand, "tfidf_cosine") and not pd.isna(cand.tfidf_cosine) else 0.0

        brand_rel = _brand_relation(
            a["brand_norm"], b["brand_norm"], a["brand_canonical"], b["brand_canonical"],
            bool(a["is_private_label_inferred"]), bool(b["is_private_label_inferred"]),
        )
        unit_rel = _size_relation(
            a.get("unit_canonical"), b.get("unit_canonical"), a.get("unit_dim"), b.get("unit_dim")
        )
        total_rel = _size_relation(
            a.get("total_canonical"), b.get("total_canonical"), a.get("total_dim"), b.get("total_dim")
        )
        pack_rel = _pack_relation(a.get("pack_count"), b.get("pack_count"))

        cat_tok_jacc = _jaccard(_tokens(a.get("category_path_norm", "")), _tokens(b.get("category_path_norm", "")))

        flavor_conflict = _has_flavor_conflict(a["name_norm"], b["name_norm"])
        form_conflict = _has_form_conflict(a["name_norm"], b["name_norm"])
        organic_rel = _organic_relation(
            bool(a["is_organic_inferred"]), bool(b["is_organic_inferred"]), bool(a["is_food_like"] or b["is_food_like"])
        )

        # Hard vetoes:
        veto = False
        veto_reasons = []
        if brand_rel == "conflict" and not (a["is_private_label_inferred"] and b["is_private_label_inferred"]):
            veto = True
            veto_reasons.append("brand_conflict")
        if unit_rel == "conflict_dim" or total_rel == "conflict_dim":
            veto = True
            veto_reasons.append("size_dim_conflict")
        if flavor_conflict:
            veto = True
            veto_reasons.append("flavor_conflict")
        if form_conflict:
            veto = True
            veto_reasons.append("form_conflict")
        if organic_rel == "conflict":
            veto = True
            veto_reasons.append("organic_conflict")

        # Composite score (weighted).
        # Brand contribution
        brand_w = {
            "exact": 0.35,
            "alias": 0.30,
            "private_label_compatible": 0.18,
            "unknown": 0.05,
            "conflict": 0.0,
        }[brand_rel]
        # Size contribution
        size_w = {"same": 0.30, "multipack": 0.22, "near": 0.18, "unknown": 0.06, "off": 0.0, "conflict_dim": 0.0}[unit_rel]
        if size_w < 0.30 and total_rel == "same":
            size_w = max(size_w, 0.22)
        # Name contribution
        name_w = 0.18 * (rapid / 100.0)
        # Cosine + cat
        cos_w = 0.12 * max(0.0, min(1.0, cosine))
        cat_w = 0.05 * cat_tok_jacc

        final_score = brand_w + size_w + name_w + cos_w + cat_w
        if veto:
            final_score = 0.0

        # Routing
        # Auto-accept guardrails (these tiers measured ~95-100% precision on eval):
        #   (a) T1 strong: brand-aligned + size-aligned + name fuzz >= 92
        #   (b) T3 strong: cosine >= 0.6 + brand-aligned + size aligned
        # Audit fix #9: require positive size evidence (same/near/multipack);
        # 'unknown' must route to LLM, not auto-accept.
        auto = False
        if not veto:
            if brand_rel in {"exact", "alias"} and unit_rel in {"same", "multipack"} and rapid >= 92:
                auto = True
            elif (
                cosine >= 0.6
                and brand_rel in {"exact", "alias"}
                and unit_rel in {"same", "near", "multipack"}
            ):
                auto = True

        # Route-to-LLM: anything not vetoed and not already auto-accepted gets
        # a look, provided it has *some* signal beyond noise. We drop only:
        #   - vetoed candidates
        #   - low-signal T3-only candidates (cosine < 0.45, no brand, no size)
        is_pure_noise = (
            cand.candidate_source == "T3"
            and cosine < 0.45
            and brand_rel not in {"exact", "alias", "private_label_compatible"}
            and unit_rel not in {"same", "multipack", "near"}
        )

        if veto:
            route = "drop"
        elif auto:
            route = "auto_accept"
        elif is_pure_noise:
            route = "drop"
        else:
            route = "route_to_llm"

        rows.append(
            {
                "item_id_a": cand.item_id_a,
                "item_id_b": cand.item_id_b,
                "candidate_source": cand.candidate_source,
                "tfidf_cosine": cosine,
                "rapidfuzz_wratio_name": rapid,
                "token_jaccard_name": token_jacc,
                "brand_relation": brand_rel,
                "unit_size_relation": unit_rel,
                "total_size_relation": total_rel,
                "pack_count_relation": pack_rel,
                "category_token_jaccard": cat_tok_jacc,
                "organic_relation": organic_rel,
                "flavor_conflict": flavor_conflict,
                "form_conflict": form_conflict,
                "veto_reasons": ",".join(veto_reasons),
                "final_score": round(final_score, 4),
                "route": route,
            }
        )

    return pd.DataFrame(rows)


def build_lookup_index(df: pd.DataFrame) -> dict[str, dict]:
    """item_id -> row-as-dict for fast lookup by retrieve/judge."""
    cols = [
        "name_norm",
        "brand_norm",
        "brand_canonical",
        "category_path_norm",
        "is_private_label_inferred",
        "is_food_like",
        "is_fresh_like",
        "is_organic_inferred",
        "unit_canonical",
        "unit_dim",
        "unit_bucket",
        "total_canonical",
        "total_dim",
        "total_bucket",
        "pack_count",
        "size_text",
        "name",
        "brand_raw",
        "info_category_0",
        "info_category_1",
        "info_category_2",
        "url_slug_norm",
        "description_norm",
        "ingredients_norm",
    ]
    have = [c for c in cols if c in df.columns]
    sub = df[["item_id"] + have].copy()
    sub = sub.set_index("item_id")
    return sub.to_dict("index")
