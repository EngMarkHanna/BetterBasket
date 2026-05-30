# EDA Analysis — Findings

The exploratory data analysis that shaped both systems. Captures what
the data looks like, what doesn't work, what works, and the
quantitative basis for every design decision in `SYSTEM_1_DESIGN.md`
and `SYSTEM_2_DESIGN.md`.

For the scripts themselves and how to re-run, see
`data analysis/README.md`. This document is the *findings*.

---

## 1. The deliverable, in context

Match each of 233,195 Walmart (Store A) products to the single
closest match in 55,516 Wegmans (Store B) products. Output:
`item_id_A,item_id_B` CSV with at least 4,000 high-quality matches.

Two valid match types:
- **Exact / national brand**: same product sold by both stores. UPC
  would be ideal but isn't reliably available.
- **Private-label or fresh equivalent**: customer-equivalent products
  where brands differ (A's "Great Value" ≈ B's "Wegmans").

The full match set is expected to exceed 10,000.

---

## 2. Catalog profile

| | Store A (Walmart) | Store B (Wegmans) |
|---|---|---|
| Rows | 233,195 | 55,516 |
| Unique `item_id` | 233,193 (2 malformed) | 55,516 |
| `brand_raw` coverage | 54.1% | 90.0% |
| Old parsed-size coverage | 58.1% | 85.2% |
| `description` coverage | 8.1% (sparse) | 93.6% (rich) |
| `ingredients` (in `item_info`) | 3.4% | **52.7%** |
| Inferred private-label rows | 7,193 (3.1%) | 8,064 (14.5%) |
| Useful `item_info` fields | `category_0..3` | `category_0..3` + `ingredients` |

**Key asymmetry**: A is larger and messier; B is smaller and better-
structured. A's data quality is the bottleneck for most decisions.

Top `info_category_0` values:
- **Store A**: Food 31.5%, Health & Medicine 8.8%, Personal Care 6.7%,
  Household Essentials 6.3%, Toys 6.1%, Pets 5.6%, Baby 5.5%.
- **Store B**: More Departments 35.3%, Grocery 33.2%, Wine/Beer/Spirits
  9.1%, Frozen 6.6%, Dairy 5.3%.

---

## 3. What doesn't work as a join key

### No ID overlap

`item_id`, `datapoint_id`, `raw_data_id` share **zero values** across
stores. The data is naturally disjoint — there's no easy ID-based join.

### UPC matching is dead

Across `name + description + item_info + tags + url` I extracted every
12–14 digit token, normalized (drop leading zeros so 12/13/14-digit
variants collapse), and intersected:

- Store A: 1,122 rows carry a UPC-like token (1,150 unique normalized
  codes).
- Store B: 366 rows carry one.
- **Unique codes shared with B: 0. Cross-store UPC-linked pairs: 0.**

The 12–14 digit tokens are almost entirely internal Walmart item IDs
(`walmart.com/ip/.../15783863493`), ISBNs, model numbers, or Wegmans
internal product IDs. UPC is unusable as a matching signal in either
direction. The pipeline should not spend cycles on it.

### Top-level categories are incompatible

`info_category_0` has **0 shared values** between stores. Shared
values increase deeper in the hierarchy but remain limited:

| Level | Shared values |
|---|---|
| `info_category_0` | 0 |
| `info_category_1` | 4 |
| `info_category_2` | 47 |
| `info_category_3` | 62 |

Direct category equality is meaningless. We need a learned bridge
(see section 5).

### `name` alone is too noisy

Marketing variants crowd lexical matching: word reordering, taglines,
brand qualifiers. Pure name matching produces high false-positive
rates without supporting structure (brand, size, category).

---

## 4. What does work

### Strict brand + size blocking — high precision, narrow recall

| Block | A coverage | B coverage | Pairs |
|---|---|---|---|
| `brand_norm` | 14.8% | 51.7% | ~1,070,709 |
| `brand_norm + size_bucket` | 4.2% | 23.5% | ~57,938 |
| `brand_norm + size_bucket + cat2` | 0.3% | 1.7% | ~3,611 |
| `size_bucket + cat2` | 2.3% | 7.9% | ~93,834 |

Only 4.2% of A rows have a shared `brand + size` block with B. That's
too narrow to deliver 4,000–10,000 matches alone, but at that block
the precision is excellent (the labeled `A_strong` stratum was 100%
positive). **It's a precision tier, not a recall mechanism.**

### TF-IDF retrieval — universal candidate generator

Built joint TF-IDF over `name + brand + URL slug + category` using
**word 1–2grams + char_wb 3–5grams**. Numbers:

- Vocabulary: 616,872 features.
- Fit time: 71.6s. Transform of all A+B: 71.6s.
- Top-10 retrieval throughput: 285 A rows/sec.
- **Estimated full A (233k) top-10 run: ~14 minutes.**
- Top-1 cosine: median 0.314, p90 0.598.

Quality (on 2,000-row sample):
- 12.2% of A rows with a brand find a same-brand B candidate in top-10.
- 21.9% find a same `size_bucket` B candidate.
- 4.3% find a same `info_category_2` B candidate.
- 52.7% have at least one top-10 candidate with cosine ≥ 0.3.

**Conclusion**: TF-IDF is not a final matcher — same-brand recall at
top-10 is low because B's 55k catalog lacks coverage of A's 233k long
tail. But TF-IDF gives plausible B candidates for 100% of A rows
including the 45% with no brand at all, fast and cheaply. It belongs
at the top of the candidate pipeline; downstream rescoring decides.

### Learned A→B category bridge — soft prior

Using 8,195 shared-brand+size seed pairs at RapidFuzz ≥ 80, aggregated
co-occurring (A category, B category) at every taxonomy level:

| Level | Unique A cats | At least one seed | Support ≥ 5 | Top-B share ≥ 60% |
|---|---|---|---|---|
| `info_category_0` | 40 | 26 | 20 | 23 |
| `info_category_1` | 610 | 213 | 113 | 179 |
| `info_category_2` | 2,657 | 786 | 275 | 610 |
| `info_category_3` | 3,563 | 1,011 | 261 | 876 |

Where the bridge has seed support, it's usually decisive. For
`info_category_2`, 610/786 mapped A categories (77.6%) have a B
target representing ≥ 60% of seed pairs. Examples:

- A `Deodorants` → B `Deodorant & Antiperspirant` (96%)
- A `Tea` → B `Tea` (93%)
- A `Baby Food` → B `Baby Food & Snacks` (100%)
- A `Cookies` → B `Cookies` (77%)
- A `Energy Drinks` → B `Sports & Energy Drinks` (72%)

Noise is real where support is small. **Apply as a soft prior (boost
candidates with matching bridge), not a hard filter, and only at
support ≥ 5.** For the 70% of A categories with no seed mapping, fall
back to fuzzy text similarity between normalized category paths.

### B-side `ingredients` — unused signal for food/private-label

Per-store coverage of `item_info` fields the early EDA didn't exercise:

- Store A: `storage_type` 6.0%, `packaging_description` 7.9%,
  `ingredients` 3.4%.
- Store B: `storage_type` 0.0%, `packaging_description` 0.0%,
  **`ingredients` 52.7%**.

A is sparse on storage/packaging — rely on its category hierarchy
(`Frozen Foods`, `Refrigerated`) instead. B has effectively no
storage/packaging metadata, but the category hierarchy encodes form
(`Frozen`, `Dairy`).

**B's ingredients field is the standout**. Ingredient overlap (Jaccard
on normalized ingredient tokens) is a strong customer-equivalence
signal exactly where matching is hardest: fresh and private-label
products. Recommended use: when both A and B are food candidates,
compute ingredient-token Jaccard on A's name+description+size source
vs B.ingredients; require ≥ 0.4 for accepting a non-exact food match.

System 2 puts ingredients directly into the embedding text (v2
format).

### URL slugs — small but free signal

For 5,000 sampled A rows with both `url` and `name`: URL slug adds on
average 0.94 extra normalized tokens beyond `name`; 31.6% of rows get
at least one new token from the slug; p90 = 3 extra tokens. Most
useful for the long-tail Walmart catalog where the on-page slug is
more descriptive than the truncated display name. **Action**: include
slug in the text fed to TF-IDF (and System 2 embeddings).

### Brand alias map — unlocks thousands of extra matches

The earlier EDA reported 2,438 shared `brand_norm` values across A
and B. Running RapidFuzz `token_set_ratio` between top-2,000
most-frequent A and B brand_norm values surfaced **83 alias
candidates at score ≥ 90 and 77 at score ≥ 95** that aren't currently
shared. High-confidence examples (all score 100):

| B brand_norm | A brand_norm |
|---|---|
| `e l f` (223) | `e l f cosmetics` (151) |
| `l oreal` (66) | `l oreal paris` (26) |
| `amy s` (75) | `amy s kitchen` (20) |
| `bigelow` (40) | `bigelow tea` (39) |
| `stonyfield organic` (28) | `stonyfield` (15) |
| `toll house` (27) | `nestl toll house` (11) |
| `lindt lindor` (25) | `lindt` (106) |
| `suave essentials` (15) | `suave` (178) |

A few need manual review (`good ~ good to go`, `bell ~ bell and
evans`, `york ~ new york shuk`) because one side is generic.

System 1 ships with a hand-curated 20-entry subset. System 2 promotes
the broader 64-entry list into its knowledge base with category guards.

---

## 5. Parser audit — two correctable bugs

`08_parser_audit.py` confirms two systematic failure modes:

### Multipack indicator not multiplied into the bucket

The regex `(\d+)\s*(?:x|pk|pack)\s*(\d+(?:\.\d+)?)` only multiplies
when count and size are adjacent. So `(3 pack) Betty Crocker
... 15.25 oz` is parsed as per-unit `weight:15` instead of pack-total
`weight:45`.

- Store A: 67,475 multipack-indicator names; **31,358 rows (13.4% of A)
  end up with a bucket whose value doesn't account for the pack
  multiplier.**
- Store B: 1,891 such rows (3.4%).

This is the largest single fix available. System 1's `parse.py`
stores both `unit_canonical` (per-unit) and `total_canonical` (full
package) so multipacks are queryable both ways. The detached-pack
pattern remains in the deferred audit fix list (#7).

### Leading-decimal sizes silently dropped

382 Store A rows have a size source like `.5 L` where the leading `0`
is missing; the parser regex requires `\d+(?:\.\d+)?`. The parser
falls back to `name`, which usually contains another parsable size,
so the bucket isn't always empty. Still worth fixing.

### Trivial `count:1` buckets

1,562 A rows and 39 B rows have `count:1` buckets ("Stapler 1 each").
These convey no useful matching signal — they create false
high-coverage size matches if treated as real. Should be treated as
"unknown size".

These three parser issues (multipack detached pattern, leading
decimal, count:1) are in the audit's deferred Tier-3 list. Fixing
them requires rebuilding the Parquet cache.

---

## 6. The eval set as calibration anchor

97 hand-labeled pairs across 9 strata. Stratified sampling so each
slice of the candidate space gets coverage. Per-stratum positive label
rates *are* the precision baselines that drive routing:

| Stratum | n | Definition | Label match rate |
|---|---|---|---|
| `A_strong` | 20 | brand + size + RapidFuzz ≥ 95 | **100%** |
| `T_strong_tfidf` | 9 | TF-IDF cosine ≥ 0.6 (rank-1) | **89%** |
| `A_private_high` | 10 | private-label, size-only, RF ≥ 90 | 70% |
| `A_borderline` | 3 | brand-block, RF 75–85 | 33% |
| `H_hand` | 10 | hand-crafted edge cases | 30% |
| `T_weak_tfidf` | 10 | TF-IDF cosine 0.4–0.6 | 10% |
| `A_medium` | 20 | brand-block, RF 85–95 | 5% |
| `A_private_mid` | 10 | private-label, RF 85–90 | 0% |
| `A_low_score` | 5 | brand-block, RF < 75 | 0% |

The eval set is **read-only ground truth**. We tune against it, never
on it. Both systems' validation modules consume `eval_labels.csv` and
report per-stratum metrics.

---

## 7. LLM behavior on the eval set

Benchmarking `gpt-5.4-nano` on the 97 pairs (`11_eval_model_benchmark.py`):

| Metric | Value |
|---|---|
| Aggregate precision | 0.906 |
| Aggregate recall | 0.744 |
| F1 | 0.817 |
| **Precision at confidence ≥ 0.85** | **100%** |
| Precision at confidence ≥ 0.95 | 100% |
| JSON validity rate | 100% |

The model is **strict** — high precision, lower recall. That makes it
a great *filter* for borderline candidates, not a great *recall
saver*. This shaped System 1's central design choice: **LLM-as-veto,
not LLM-for-everything**.

Failure modes (recall losses):

1. **Multipack-of-same-SKU rejected** (e.g., Betty Crocker 3-pack
   15.25 oz vs single 13.25 oz).
2. **Naming-order drift rejected** ("Folgers Black Silk Ground Coffee
   Dark Roast" vs "Folgers Coffee Ground Dark Black Silk").
3. **Marketing-copy drift rejected** ("Palmer's for Dry Skin" vs
   "Daily Coconut Hydrate" on the same 8.5 oz Coconut Oil).
4. **Private-label Organic vs unmarked rejected** even when category
   and size align.

System 1 patches the first three with rubric clauses (with limited
effect — the rubric can't grow unbounded). System 2 attacks all four
by injecting per-pair RAG context with the relevant rules and similar
labeled examples.

---

## 8. Phase 0 feasibility — projection of the full run

`13_phase0_feasibility.py` built real T1+T3 candidate counts and
projected accepted matches under the winning routing policy:

| Source | Projected accepted matches |
|---|---|
| Auto-accept T1 (RapidFuzz ≥ 95) | 751 |
| Auto-accept T3 (cosine ≥ 0.6 + brand-aligned) | 9,374 |
| LLM-accepted T1 score 85–95 | 364 |
| LLM-accepted T1 score 70–85 | 17 |
| LLM-accepted T3 brand-unaligned cosine ≥ 0.6 | 2,896 |
| LLM-accepted T3 cosine 0.4–0.6 | 2,775 |
| LLM-accepted total | ~6,000 |
| **Projected total** | **~16,000** |
| Target | 4,000 |
| **Headroom** | **4×** |

System 1's actual full run produced **17,040 matches** — slightly
above the conservative projection because the routing was loosened
during execution.

---

## 9. How findings became architecture

| Finding | Architectural consequence |
|---|---|
| 12.9B all-pairs too expensive | Retrieval / blocking is mandatory before scoring |
| No UPC / ID overlap | No easy join; must use text/structure features |
| Top categories incompatible | Learned category bridge as soft prior |
| Brand + size + RapidFuzz ≥ 95 → 100% label rate | Direct auto-accept tier in `score.py` |
| TF-IDF cosine ≥ 0.6 + brand-aligned → 89% | Second auto-accept tier |
| LLM is 100% precise at conf ≥ 0.85 but 74% recall | LLM-as-veto, not recall saver |
| B has 52.7% ingredients, A has 0% | Ingredient overlap as System 2 signal; A side handled gracefully |
| Old parser missed 22k multipack rows | Unit + total size buckets in `parse.py` |
| 20 curated aliases lift coverage 10% | Locked alias map in `parse.py::BRAND_ALIASES` |
| Per-stratum label rates form a calibration curve | Routing thresholds anchored to eval baselines |

---

## 10. Open questions the EDA didn't close

- **Recall validation of TF-IDF top-10**: the 12.2% same-brand-in-top-10
  number conflates B catalog gaps with TF-IDF gaps. A held-out manually
  verified set of ~100 obvious matches would separate the two.
- **Sentence-transformer / OpenAI embeddings on cases where TF-IDF
  top-1 < 0.3**: addressed in System 2 (T7 semantic retriever).
- **Dedicated tier-3 fresh/loose-products pass** using produce/meat/
  cheese/deli categories from B against A's `Fresh Food`/`Produce`
  slices, with size + form constraints learned from `storage_type`.
  Defer; eval set has no fresh stratum.
- **Eval set expansion** to ≥ 300 labels with fresh + private-label
  oversampling. System 1's measured F1 0.79 and System 2's eval-mode
  0.81 on 97 pairs are noisy estimates.
