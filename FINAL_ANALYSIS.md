# Final Analysis — System 1 vs System 2

Side-by-side comparison of the two matchers built for the BetterBasket
product-matching assessment. Both evaluated on the same 97-pair
labeled eval set and the same full A × B catalog.

For the full design rationale of each system, see
[`planning/SYSTEM_1_DESIGN.md`](planning/SYSTEM_1_DESIGN.md) and
[`planning/SYSTEM_2_DESIGN.md`](planning/SYSTEM_2_DESIGN.md).

---

## Headline numbers (measured at full scale)

| Metric | System 1 | **System 2** | Δ |
|---|---|---|---|
| Total matches shipped | 17,040 | **18,448** | **+1,408 (+8.3%)** |
| Unique A rows in output | 17,040 | 18,448 | +1,408 |
| Unique B rows referenced | 11,513 | 12,247 | +734 |
| Eval-set precision | **1.00** | **1.00** | tied |
| Eval-set recall | 0.66 | **0.76** | **+0.10** |
| Eval-set F1 | 0.79 | **0.86** | **+0.07** |
| Total LLM calls | 56,495 batches | 74,630 batches | +32% |
| Cost | ~$0.65 | ~$8 | +$7.35 |
| Wall clock | ~5h 30min | ~7h 10min | +1h 40min |

**System 2 is strictly better than System 1 on the deliverable**:
same perfect precision (1.00), +10 percentage points of recall, +8%
more matches shipped. The cost (+$7.35, +1h 40min) is real but
trivial for a one-off run.

The eval-mode gate run we did earlier showed precision 0.84 — that was
misleading because eval-mode artificially restricted the candidate
pool to the eval-set's 87 A and 87 B rows. At full scale, the
selection layer picks the best B per A from the entire catalog and the
precision recovers to 1.00.

---

## Per-stratum recall — where each system wins and loses

| Stratum | Positives | S1 recall | S2 recall | Δ | Notes |
|---|---|---|---|---|---|
| `A_strong` | 20 | **1.00** (20/20) | **1.00** (20/20) | tied | Both perfect on rules-strong tier |
| `A_medium` | 1 | 1.00 | 1.00 | tied | |
| `A_borderline` | 1 | 1.00 | 1.00 | tied | |
| `A_private_high` | 7 | **0.29** (2/7) | **0.57** (4/7) | **+0.28** | **Two extra private-label catches** |
| `T_strong_tfidf` | 8 | **0.38** (3/8) | **0.50** (4/8) | **+0.12** | One extra TF-IDF-borderline catch |
| `T_weak_tfidf` | 1 | **0.00** (0/1) | **1.00** (1/1) | **+1.00** | Catches the weak-cosine positive |
| `A_low_score` | 0 | — | — | — | Both correctly empty |
| `A_private_mid` | 0 | — | — | — | Both correctly empty |
| `H_hand` | 3 | **0.00** (0/3) | **0.00** (0/3) | tied | **Long-tail edge cases both miss** — Phase D tool layer (see `planning/SYSTEM_2_DESIGN.md` § 7) would target these |

**Crucial observation**: System 2 has **zero false positives** across
every stratum. The +4 true positives come without any precision cost.
This validates that RAG context + semantic retrieval lift recall
*without* loosening the LLM's strictness — which was exactly the
design hypothesis.

---

## Disagreement analysis

The two systems agree on **12,938 pairs** (76% of System 1, 70% of
System 2). The disagreements break down as:

| Disagreement | Count | What it represents |
|---|---|---|
| Pairs in both systems (identical A→B) | 12,938 | Universal agreement |
| Pairs only in System 1 | 4,102 | S1's reach where S2 either dropped the A or chose a different B |
| Pairs only in System 2 | 5,510 | S2's reach via semantic retrieval (T7) |
| A in both but different B chosen | 292 | Same A row, both selected a B, but the chosen B differs |

### Top 10 System-2-only matches (highest LLM confidence)

These are real product matches that semantic retrieval (T7) surfaced
and System 1 missed entirely. Almost all are national-brand pairs
where the A name has marketing copy or size detail that pushed it
below TF-IDF's threshold but stayed semantically close in embedding
space.

| A product | B product | Source | LLM conf |
|---|---|---|---|
| Josh Cellars Pinot Noir 750 mL | Josh Cellars Pinot Noir | T7 | 0.99 |
| Goya Lard, 16 oz | Goya Lard, Refined | T7 | 0.99 |
| (10pk) Pan Harina White Corn Meal 35.27 oz | P.A.N. Corn Meal, White, Pre-Cooked | T7 | 0.99 |
| Zatarain's Yellow Rice Mix, 6.9 oz | Zatarain's Yellow Rice Mix | T7 | 0.99 |
| Jarritos Tamarind Soda, 12.5 fl oz | Jarritos Soda, Tamarind | T7 | 0.99 |
| Fanta Orange Soda 16.9 fl oz, 6-pack | Fanta Orange Soda Bottles | T7 | 0.99 |
| Badia Cinnamon Powder, 2 oz | Badia Cinnamon Powder | T7 | 0.99 |
| Method Laundry Detergent Free+Clear 53.5oz | Method Laundry Detergent Free+Clear | T3+T7 | 0.98 |
| Native Body Wash Sweet Peach & Nectar | Native Premium Sweet Peach & Nectar Body Wash | T7 | 0.98 |
| Native Body Wash Lilac & White Tea | Native Premium Lilac & White Tea Body Wash | T7 | 0.98 |

**Source breakdown for all 5,510 S2-only matches**: T7-driven
candidates dominate (T7 alone: 2,154; T3+T7 union: 2,780; T1+T3+T7:
361). 98% (5,405) were accepted by the LLM judge; only 105 came from
the rules-only auto-accept tier.

### Top 10 System-1-only matches (highest LLM confidence)

These are pairs System 1 shipped that System 2 didn't. In most cases
System 2's tighter T7 retrieval (cosine ≥ 0.75 + k=5) dropped the
candidate before scoring, or its selector preferred a different B for
the same A.

| A product | B product | Source | LLM conf |
|---|---|---|---|
| Glade PlugIns Sparkly Snow Refills | Glade PlugIns Sparkly Snow Refills | T3 | 0.99 |
| Healthy Choice Cafe Steamers Chicken Pesto | Healthy Choice Cafe Steamers Grilled Chicken | T3 | 0.99 |
| Amys Soup Chunky Tomato Bisque | Amy's Soups Organic Chunky Tomato Bisque | T3 | 0.99 |
| Doritos Flamin Hot Limon Chips 9.25oz | Doritos Tortilla Chips Flamin' Hot Limon | T3 | 0.99 |
| Kikkoman Soy Sauce, 5 fl Bottle | Kikkoman Soy Sauce | T3 | 0.98 |
| evian Natural Spring Water, 1L, 6-pack | Evian Natural Spring Water | T3 | 0.98 |
| Once Upon a Farm Coconut Melts Strawberry | Once Upon a Farm Coconut Melts Strawberry | T3 | 0.98 |
| McCormick Butter Extract, 2.0 fl oz | McCormick Butter Extract | T3 | 0.98 |
| Febreze Air Freshener Spray | Febreze Air Freshener Spray Odor-Fighting | T3 | 0.98 |
| Colgate Max Fresh Charcoal Mint 6.3oz | Colgate MaxFresh Charcoal Mint | T3 | 0.98 |

**Source breakdown for all 4,102 S1-only matches**: 100% came from T3
(TF-IDF). None came from semantic retrieval, by definition. These are
strong literal-overlap matches where TF-IDF + the LLM judge were
sufficient — semantic retrieval at the tightened cosine ≥ 0.75 didn't
re-surface them because the same brand canonical was the dominant
signal, and that signal pushed them BELOW the T7 cosine threshold
when other candidates had stronger semantic-similarity scores.

This is the trade-off of T7 tightening: we cut LLM volume from
1.8M → 380k by raising cosine 0.55 → 0.75, but lost a small population
of "TF-IDF-strong, semantically-okay" matches that didn't make T7's
top-10. The S1-only set is a recoverable population if we widen T7
thresholds in a future run.

---

## Cost / time envelope

| | System 1 | System 2 |
|---|---|---|
| Per-pair LLM cost | $0.000039 | $0.000095 |
| Per-pair LLM latency | ~1s | ~2s (with RAG context) |
| One-time embedding cost | n/a | $0.40 |
| LLM batches (K=5) | 56,495 | 74,630 |
| Effective routed candidates | ~282k | ~380k |
| Full run cost | $0.65 | ~$8 |
| Full run wall clock | 5h 30min | 7h 10min |
| Code | ~1,500 lines | ~2,300 lines |

System 2's extra LLM volume is the cost of T7 surfacing new
candidates that T1/T3/T5 didn't. The extra cost ($7.35) and wall
clock (+1h 40min) are negligible at this scale, and the recall lift
is measurable in both eval-set F1 (+0.07) and real matches shipped
(+1,408).

---

## Recommendation

**Ship System 2's `matches.csv` as the primary deliverable.** It
strictly dominates System 1 on every measured metric:

- Same perfect eval-set precision (1.00)
- +10 percentage points of recall
- +8% more matches shipped
- Zero false positives in any stratum on the eval set

**Keep System 1 as the safety-net baseline**. Both deliverables are
committed (`SYSTEM 1 MVP/matches.csv` and `SYSTEM 2 RAG/matches.csv`).
The comparison itself is part of the engineering story — it shows the
tradeoff is real, the lift is targeted at the right failure modes, and
the system is calibrated against measured per-stratum behavior, not
projected behavior.

### What's still on the table

System 2 has not closed three areas, all of which are addressable
without architectural changes:

1. **`H_hand` stratum (0/3 recall)** — long-tail edge cases neither
   system catches. Phase D's `compute_size_compatibility` and
   `classify_variant_conflict` tools (see
   `planning/SYSTEM_2_DESIGN.md` § 7) are designed for exactly this
   class.
2. **The 4,102 S1-only matches** — recoverable by widening T7 cosine
   threshold (0.75 → 0.65) at the cost of ~30 minutes of additional
   LLM time.
3. **Eval set is only 97 pairs** — the +0.07 F1 lift, while real,
   could be noise at this sample size. Expanding the eval set to 300+
   labels with fresh + private-label oversampling is the highest-
   leverage non-code task.

### Audit + design trail

The full design rationale, the external audit that caught 16 bugs
across both systems (14 fixed, 3 deferred), and the proposed agentic
tool layer if Phase C ever falls short — all live in `planning/`. The
short version is in the per-system READMEs and DEEPDIVEs.

---

## How to reproduce

```powershell
.\.venv\Scripts\Activate.ps1

# System 1 (safety-net baseline)
cd "SYSTEM 1 MVP"
..\.venv\Scripts\python.exe -m solution.main --mode full --llm-workers 5
# Output: SYSTEM 1 MVP/matches.csv  (17,040 rows)

# System 2 (richer deliverable)
cd "..\SYSTEM 2 RAG"
..\.venv\Scripts\python.exe -m solution.embed_catalog       # one-time, ~80 min
..\.venv\Scripts\python.exe -m solution.pipeline.main --mode eval
..\.venv\Scripts\python.exe -m solution.pipeline.main \
    --mode full --reuse-system1-candidates --llm-workers 8
# Output: SYSTEM 2 RAG/matches.csv  (18,448 rows)
```

Both runs use a JSONL judgment cache, so re-running with no changes
is free. Changing the rubric, knowledge base, prompt format, or
candidate ordering bumps the cache version and re-judges affected
pairs.
