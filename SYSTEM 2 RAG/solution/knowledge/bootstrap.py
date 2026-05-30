"""Bootstrap the knowledge base from existing EDA artifacts.

Each EDA output becomes one JSONL file under `knowledge/`:

  knowledge/rules.jsonl              hand-written rubric clauses
  knowledge/brand_aliases.jsonl      from outputs/brand_alias_candidates.csv
  knowledge/category_bridges.jsonl   from outputs/category_bridge_a_to_b.csv
  knowledge/accepted_examples.jsonl  from outputs/eval_results.csv where label_is_match
  knowledge/rejected_examples.jsonl  from outputs/eval_results.csv where not label_is_match
  knowledge/edge_cases.jsonl         hand-curated patterns

Re-running bootstrap is idempotent: same inputs produce the same JSONL.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from .entry import EntryType, KnowledgeEntry, write_jsonl


# ----------------------------------------------------------------------------
# Hand-written rules (the rubric, decomposed into discrete entries)
# ----------------------------------------------------------------------------

RULES: list[KnowledgeEntry] = [
    KnowledgeEntry(
        id="rule_multipack_equivalence",
        type=EntryType.RULE,
        title="Multipack vs single SKU is an acceptable match",
        content=(
            "When the per-unit SKU is the same (same brand, same product, same per-unit "
            "size), a multipack on one side can be an exact_national_brand match against "
            "a single on the other side. Different pack counts are NOT a match-breaker. "
            "Example: A '12-pack 12 oz Coke' matches B '12 oz Coke' if per-unit fields align."
        ),
        match_type="exact_national_brand",
        requires_brand_relation="exact",
        tags=["multipack", "per_unit"],
        source="rubric_v2",
    ),
    KnowledgeEntry(
        id="rule_word_order_drift",
        type=EntryType.RULE,
        title="Word order and punctuation drift do not break a match",
        content=(
            "Reordered words or differences in punctuation, capitalization, or marketing "
            "phrasing are NOT match-breakers. 'Folgers Black Silk Ground Coffee Dark Roast' "
            "and 'Folgers Coffee Ground Dark Black Silk' are the same SKU."
        ),
        match_type="exact_national_brand",
        tags=["naming_drift"],
        source="rubric_v2",
    ),
    KnowledgeEntry(
        id="rule_marketing_copy_drift",
        type=EntryType.RULE,
        title="Marketing copy drift on same SKU",
        content=(
            "Different marketing taglines on the same SKU ('Daily Coconut Hydrate' vs "
            "'for Dry Skin') do not break a match when brand, size, form, and product "
            "family all agree."
        ),
        match_type="exact_national_brand",
        tags=["marketing_drift"],
        source="rubric_v2",
    ),
    KnowledgeEntry(
        id="rule_organic_food_conflict",
        type=EntryType.RULE,
        title="Organic vs conventional is a hard conflict for food",
        content=(
            "For food, dairy, produce, baby food, and ingredients, organic vs "
            "conventional IS a meaningful mismatch. Same brand, same name, same size, "
            "but one organic and the other conventional should be no_match."
        ),
        match_type="no_match",
        product_domain="food",
        tags=["organic", "domain_rule"],
        source="rubric_v2",
    ),
    KnowledgeEntry(
        id="rule_flavor_conflict",
        type=EntryType.RULE,
        title="Different flavors are no_match",
        content=(
            "Different flavors, scents, shades, or formulations are no_match unless the "
            "two product lines are explicitly equivalent. Strawberry yogurt is not the "
            "same SKU as vanilla yogurt even from the same brand at the same size."
        ),
        match_type="no_match",
        tags=["flavor", "variant"],
        source="rubric_v2",
    ),
    KnowledgeEntry(
        id="rule_private_label_equivalence",
        type=EntryType.RULE,
        title="Private-label equivalence rules",
        content=(
            "Two private-label products (Great Value, Marketside, Wegmans, Equate, etc.) "
            "of the same specific product, same size, same form qualify as "
            "private_label_equivalent. Different flavors or different products do NOT "
            "qualify. Ingredient overlap is strong evidence of equivalence when available."
        ),
        match_type="private_label_equivalent",
        requires_brand_relation="private_label_compatible",
        tags=["private_label"],
        source="rubric_v2",
    ),
    KnowledgeEntry(
        id="rule_size_tolerance",
        type=EntryType.RULE,
        title="Manufacturer size revision tolerance",
        content=(
            "Per-unit size drift up to ~15% can be acceptable for the same SKU when all "
            "other signals (brand, name, form, category) align. Example: A 12.6 oz can "
            "and a 13.0 oz can of the same product family are likely the same SKU."
        ),
        match_type="exact_national_brand",
        requires_size_relation="near",
        tags=["size_tolerance"],
        source="rubric_v2",
    ),
    KnowledgeEntry(
        id="rule_size_dim_conflict",
        type=EntryType.RULE,
        title="Different size dimensions are not comparable",
        content=(
            "If the size dimensions differ (one is weight, the other is volume), they "
            "are not directly comparable and should be no_match unless the product "
            "family explicitly comes in both forms."
        ),
        match_type="no_match",
        tags=["size_dim"],
        source="rubric_v2",
    ),
    KnowledgeEntry(
        id="rule_form_conflict",
        type=EntryType.RULE,
        title="Product form mismatches",
        content=(
            "Liquid vs powder, ground vs whole bean, K-cups vs ground coffee, lotion vs "
            "spray are no_match. Different forms of the same brand are different SKUs."
        ),
        match_type="no_match",
        tags=["form"],
        source="rubric_v2",
    ),
    KnowledgeEntry(
        id="rule_strict_default",
        type=EntryType.RULE,
        title="Strict default when uncertain",
        content=(
            "When uncertain, prefer no_match. But do not reject matches over trivial "
            "naming, formatting, or marketing differences if the underlying SKU is the "
            "same."
        ),
        tags=["default"],
        source="rubric_v2",
    ),
]


# Risky generic brand stems we do NOT want as canonical aliases - they
# overgeneralize and would create huge spurious blocks.
ALIAS_GENERIC_DENYLIST = {
    "good", "apple", "bell", "diamond", "york", "hero", "bell and evans",
}


# ----------------------------------------------------------------------------
# Loaders for the per-source CSVs
# ----------------------------------------------------------------------------


def _aliases_from_csv(path: Path, score_min: float = 95.0) -> list[KnowledgeEntry]:
    if not path.exists():
        return []
    df = pd.read_csv(path)
    out: list[KnowledgeEntry] = []
    for r in df.itertuples(index=False):
        a = str(getattr(r, "a_brand_norm", "") or "").strip()
        b = str(getattr(r, "b_brand_norm", "") or "").strip()
        score = float(getattr(r, "score", 0.0) or 0.0)
        if not a or not b or score < score_min:
            continue
        if a in ALIAS_GENERIC_DENYLIST or b in ALIAS_GENERIC_DENYLIST:
            continue
        # Canonical: shorter form (heuristic that matches our curated map).
        canonical = a if len(a) <= len(b) else b
        variant = b if canonical == a else a
        out.append(
            KnowledgeEntry(
                id=f"alias_{canonical.replace(' ', '_')}_{variant.replace(' ', '_')}",
                type=EntryType.ALIAS,
                title=f"Brand alias: {variant!r} -> {canonical!r}",
                content=(
                    f"The brand string '{variant}' is the same manufacturer as "
                    f"'{canonical}'. Treat as identical brands when judging matches."
                ),
                requires_brand_relation="alias",
                tags=["alias"],
                source=f"brand_alias_candidates.csv (score={score:.0f})",
                confidence=score / 100.0,
            )
        )
    return out


def _bridges_from_csv(
    path: Path,
    support_min: int = 10,
    share_min: float = 0.4,
    max_b_per_a: int = 2,
) -> list[KnowledgeEntry]:
    if not path.exists():
        return []
    df = pd.read_csv(path)
    df = df[df["support"] >= support_min]
    out: list[KnowledgeEntry] = []
    for r in df.itertuples(index=False):
        a_cat = str(getattr(r, "a_category", "")).strip()
        if not a_cat:
            continue
        # Take up to max_b_per_a top targets per A category.
        targets: list[tuple[str, float]] = []
        for col_cat, col_share in (
            ("b_top1", "b_top1_share"),
            ("b_top2", "b_top2_share"),
            ("b_top3", "b_top3_share"),
        ):
            b_cat = getattr(r, col_cat, None)
            share = getattr(r, col_share, None)
            if pd.notna(b_cat) and pd.notna(share) and float(share) >= share_min:
                targets.append((str(b_cat).strip(), float(share)))
                if len(targets) >= max_b_per_a:
                    break
        if not targets:
            continue
        target_text = "; ".join(f"{b} (share={s:.2f})" for b, s in targets)
        out.append(
            KnowledgeEntry(
                id=f"bridge_{a_cat.replace(' ', '_').lower()}",
                type=EntryType.BRIDGE,
                title=f"Category bridge: {a_cat!r}",
                content=(
                    f"Store A category {a_cat!r} most commonly maps to Store B "
                    f"categories: {target_text}. Use when judging whether two products "
                    f"are in the same product domain."
                ),
                tags=["category_bridge"],
                source=f"category_bridge_a_to_b.csv (support={int(r.support)})",
                confidence=float(targets[0][1]),
            )
        )
    return out


def _examples_from_eval(
    candidates_csv: Path, labels_csv: Path
) -> tuple[list[KnowledgeEntry], list[KnowledgeEntry]]:
    if not candidates_csv.exists() or not labels_csv.exists():
        return [], []
    cand = pd.read_csv(candidates_csv, dtype=str)
    labels = pd.read_csv(labels_csv)
    joined = cand.merge(labels, on="pair_id", how="inner")
    joined["label_is_match"] = (
        joined["label_is_match"].astype(str).str.lower().isin({"true", "1"})
    )
    accepted: list[KnowledgeEntry] = []
    rejected: list[KnowledgeEntry] = []
    for r in joined.itertuples(index=False):
        body = (
            f"A: name={getattr(r, 'name_A', '')!r}, brand={getattr(r, 'brand_A', '')!r}, "
            f"size={getattr(r, 'size_A', '')!r}, category={getattr(r, 'category_A', '')!r}. "
            f"B: name={getattr(r, 'name_B', '')!r}, brand={getattr(r, 'brand_B', '')!r}, "
            f"size={getattr(r, 'size_B', '')!r}, category={getattr(r, 'category_B', '')!r}. "
            f"Reason: {getattr(r, 'label_notes', '') or ''}"
        )
        match_type = getattr(r, "label_match_type", None)
        confidence = float(getattr(r, "label_confidence", 1.0) or 1.0)
        if bool(r.label_is_match):
            accepted.append(
                KnowledgeEntry(
                    id=f"acc_{getattr(r, 'pair_id', '')}",
                    type=EntryType.ACCEPTED_EXAMPLE,
                    title=f"Accepted example ({match_type})",
                    content=body,
                    match_type=str(match_type) if match_type else None,
                    tags=["example", "positive", getattr(r, "stratum", "")],
                    source=f"eval_labels.csv pair={getattr(r, 'pair_id', '')}",
                    confidence=confidence,
                )
            )
        else:
            rejected.append(
                KnowledgeEntry(
                    id=f"rej_{getattr(r, 'pair_id', '')}",
                    type=EntryType.REJECTED_EXAMPLE,
                    title="Rejected example",
                    content=body,
                    tags=["example", "negative", getattr(r, "stratum", "")],
                    source=f"eval_labels.csv pair={getattr(r, 'pair_id', '')}",
                    confidence=confidence,
                )
            )
    return accepted, rejected


def _edge_cases_default() -> list[KnowledgeEntry]:
    """Hand-curated tricky patterns we encountered during EDA."""
    return [
        KnowledgeEntry(
            id="edge_multipack_pattern_parsing",
            type=EntryType.EDGE_CASE,
            title="Multipack patterns that bypass parsers",
            content=(
                "Catalog names like '(3 pack) Betty Crocker Muffin Mix 15.25 oz' carry "
                "the pack count detached from the per-unit size. When judging, treat the "
                "per-unit size as the relevant comparison axis."
            ),
            tags=["multipack", "parser"],
            source="parser_failure_examples.csv",
        ),
        KnowledgeEntry(
            id="edge_unit_size_in_different_units",
            type=EntryType.EDGE_CASE,
            title="Per-unit size expressed in different units",
            content=(
                "A may say '12 fl oz' and B may say '355 ml' for the same SKU. Both are "
                "the same canonical volume; treat as compatible_size."
            ),
            tags=["size", "unit_conversion"],
            source="hand_curated",
        ),
        KnowledgeEntry(
            id="edge_organic_private_label_mismatch",
            type=EntryType.EDGE_CASE,
            title="Organic vs unmarked on private-label food",
            content=(
                "Even between private-label brands, an explicit 'Organic' product is a "
                "different SKU from an unmarked private-label product of the same name "
                "and size if the domain is food. Default to no_match."
            ),
            match_type="no_match",
            product_domain="food",
            tags=["private_label", "organic"],
            source="hand_curated",
        ),
    ]


# ----------------------------------------------------------------------------
# Bootstrap driver
# ----------------------------------------------------------------------------


@dataclass
class BootstrapStats:
    rules: int = 0
    aliases: int = 0
    bridges: int = 0
    accepted_examples: int = 0
    rejected_examples: int = 0
    edge_cases: int = 0

    @property
    def total(self) -> int:
        return sum(getattr(self, f) for f in (
            "rules", "aliases", "bridges",
            "accepted_examples", "rejected_examples", "edge_cases",
        ))


def bootstrap_all(
    eda_outputs_dir: Path,
    knowledge_dir: Path,
) -> BootstrapStats:
    """Read EDA artifacts, emit JSONL knowledge files. Returns counts."""
    knowledge_dir.mkdir(parents=True, exist_ok=True)
    stats = BootstrapStats()

    stats.rules = write_jsonl(RULES, knowledge_dir / "rules.jsonl")

    aliases = _aliases_from_csv(eda_outputs_dir / "brand_alias_candidates.csv")
    stats.aliases = write_jsonl(aliases, knowledge_dir / "brand_aliases.jsonl")

    bridges = _bridges_from_csv(eda_outputs_dir / "category_bridge_a_to_b.csv")
    stats.bridges = write_jsonl(bridges, knowledge_dir / "category_bridges.jsonl")

    accepted, rejected = _examples_from_eval(
        eda_outputs_dir / "eval_candidates.csv",
        eda_outputs_dir / "eval_labels.csv",
    )
    stats.accepted_examples = write_jsonl(accepted, knowledge_dir / "accepted_examples.jsonl")
    stats.rejected_examples = write_jsonl(rejected, knowledge_dir / "rejected_examples.jsonl")

    stats.edge_cases = write_jsonl(_edge_cases_default(), knowledge_dir / "edge_cases.jsonl")
    return stats
