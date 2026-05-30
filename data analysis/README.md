# Data Analysis

The exploratory work that informed every design decision in System 1
and System 2. Read this if you want to understand *why* the matchers
look the way they do.

Scripts run independently and write to `outputs/`. None of them are
required for the end-to-end matcher to run (Systems 1 and 2 only
consume the eval set, the brand alias candidates, and the category
bridge from here).

---

## Findings, in one page

These are the load-bearing facts the matchers depend on.

### Catalog shape

| | Store A (Walmart) | Store B (Wegmans) |
|---|---|---|
| Rows | 233,195 | 55,516 |
| Brand coverage (`brand_raw` populated) | 54.1% | 90.0% |
| Old parsed-size coverage | 58.1% | 85.2% |
| Has `ingredients` field | No | Yes (52.7% of rows) |

The asymmetry matters. B is the smaller, better-structured catalog. A
is the larger, messier one — and A's data quality is the bottleneck
for most match decisions.

### What doesn't work as a join key

- **No useful ID overlap**: `item_id`, `datapoint_id`, `raw_data_id`
  share zero values across stores. (Probe in `03_match_signal_probe.py`.)
- **No UPC overlap**: UPC-like tokens have **zero** cross-store overlap.
- **Top-level categories are incompatible**: the two stores use
  different taxonomies, so direct category equality is meaningless.
- **`name` alone is too noisy**: pure name matching gets crowded
  by marketing variants.

### What does work

- **Brand + size strict block** with curated aliases — 100% match rate
  in the labeled `A_strong` stratum.
- **TF-IDF cosine over (name + brand + URL slug + category + selected
  description tokens)** — 89% rank-1 precision at cosine ≥ 0.6.
- **Learned A→B category bridge** — 610 high-confidence entries
  (support ≥ 10, share ≥ 0.6) from 8,195 shared-brand seed pairs.
- **Per-unit size matching for multipacks** — the old parser collapsed
  multipacks into one number, losing 22,353 A rows of valid per-unit
  candidates. The new parser stores both `unit_size` and `total_size`.
- **Brand alias curation** — 20 curated entries lift shared-block
  coverage by ~10% (e.g., `l oreal paris → l oreal`).
- **Ingredient overlap** — B has 52.7% coverage; critical for
  private-label-to-national-brand equivalence (mostly used by System 2).

### Eval set (the calibration anchor)

97 hand-labeled pairs across 9 strata. Per-stratum positive label
rates set the rule-precision baselines:

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

The eval set is **read-only ground truth**. We tune against it, never on it.

### The LLM is a high-precision veto, not a recall saver

Benchmarking `gpt-5.4-nano` against the 97 pairs:

| Metric | Value |
|---|---|
| Aggregate precision | 0.906 |
| Aggregate recall | 0.744 |
| Aggregate F1 | 0.817 |
| Precision at confidence ≥ 0.85 | **100%** |
| Precision at confidence ≥ 0.95 | **100%** |

Confidence ≥ 0.85 is the auto-accept gate used by both systems. The
model loses ~26% of true positives in the strict regions of the eval
set — which is what System 2's RAG context aims to recover.

### Phase-0 feasibility (projection)

Built from real T1+T3 candidate counts on full A:

| Source | Projected accepted matches |
|---|---|
| Auto-accept T1 (RapidFuzz ≥ 95) | 751 |
| Auto-accept T3 (cosine ≥ 0.6 + brand-aligned) | 9,374 |
| LLM-accepted total | ~6,000 |
| **Projected total** | **~16,000** |
| Target | 4,000 |

System 1's measured output (17,040 matches) overshot the projection
slightly because the routing was loosened after Phase 0. Either way:
4× the deliverable target, with headroom.

---

## Scripts, in run order

| # | Script | Purpose |
|---|---|---|
| 01 | `01_extract_assessment.py` | PDF → text for `[BetterBasket] Engineering Technical Assessment.pdf` |
| 02 | `02_profile_datasets.py` | Row counts, column missingness, top values per column |
| 03 | `03_match_signal_probe.py` | Cross-store ID overlap, UPC overlap, brand-overlap blocking counts |
| 04 | `04_high_confidence_estimate.py` | Strict brand+size shared-block A coverage |
| 05 | `05_text_signals_probe.py` | Description text, URL slug, item_info nested field analysis |
| 06 | `06_tfidf_blocking_probe.py` | TF-IDF top-K retrieval; wall-clock budgeting |
| 07 | `07_category_bridge.py` | Builds A→B category mapping from 8,195 shared-brand seed pairs |
| 08 | `08_parser_audit.py` | Catalogs parser failures + brand alias candidates |
| 09 | `09_llm_smoke_test.py` | Connectivity, structured-output validity, batching, Batch-API check |
| 10 | `10_build_eval_set.py` | Stratified sampling of 97 candidate pairs across 9 strata |
| (hand-labeling) | — | All 97 pairs labeled into `outputs/eval_labels.csv` |
| 11 | `11_eval_model_benchmark.py` | Runs gpt-5.4-nano on the labeled eval set; calibration tables |
| 12 | `12_executed_plan_followup.py` | Parser/alias deltas, routing-policy simulation on eval set |
| 13 | `13_phase0_feasibility.py` | Projected accept counts from real T1+T3 candidate volumes |

To re-run any of them:

```powershell
..\.venv\Scripts\python.exe "data analysis\<script>.py"
```

All write to `data analysis/outputs/`.

---

## What's in `outputs/`

The matchers depend on a small subset of these. The rest are
diagnostic.

### Consumed by Systems 1 and 2

| File | Used for | Consumer |
|---|---|---|
| `eval_candidates.csv` | 97-pair eval candidates | both validators |
| `eval_labels.csv` | Hand labels for those pairs | both validators |
| `eval_results.csv` | LLM model output × labels | knowledge bootstrap (S2) |
| `brand_alias_candidates.csv` | 83 candidate aliases | knowledge bootstrap (S2) |
| `category_bridge_a_to_b.csv` | 2,036 A→B category mappings | T5 retriever (both), knowledge bootstrap (S2) |
| `parser_failure_examples.csv` | Parser bug examples | knowledge bootstrap (S2) |

### Diagnostic outputs

| File | Insight it captures |
|---|---|
| `llm_smoke_test.md` | Endpoint latency, structured-output validity, batching savings |
| `eval_summary.md` | Aggregate model precision/recall/F1, confidence calibration, latency p50/p90/p99 |
| `executed_plan_followup.md` | Parser+alias deltas, routing-policy simulation |
| `eval_failure_review.csv` | Per-row review where model and label disagree |
| `phase0_feasibility.md` | Projected accept counts, exit-gate computation |
| `tfidf_retrieval_examples.csv` | Top-K examples for retrieval quality review |
| `high_confidence_examples.csv` | Strict-block matches at RapidFuzz ≥ 95 |
| `candidate_examples.csv` | Strict-block matches at RapidFuzz 85–95 |

---

## How findings became architecture

| Finding | Architectural consequence |
|---|---|
| 12.9B all-pairs too expensive | Retrieval/blocking is mandatory before scoring |
| No UPC/ID overlap | No easy join; must use text/structure features |
| Top categories incompatible | Need a learned category bridge for soft-matching |
| Brand + size + RapidFuzz ≥ 95 = 100% label rate | Direct auto-accept tier in `score.py` |
| TF-IDF cosine ≥ 0.6 + brand-aligned = ~89% | Second auto-accept tier |
| LLM is 100% precise at conf ≥ 0.85 but 74% recall | Use LLM as veto, not as recall saver |
| B has 52.7% ingredients, A has 0% | Ingredient overlap as a System 2 signal; A side must be handled gracefully |
| Old parser missed 22k multipack rows | Unit + total size buckets in `parse.py` |
| 20 curated aliases lift coverage 10% | Locked alias map in `parse.py::BRAND_ALIASES` |
| Per-stratum label rates form a calibration curve | Routing thresholds anchored to the eval baselines |

---

## Reproducing the eval set

If `outputs/eval_candidates.csv` or `outputs/eval_labels.csv` are
ever lost, regenerate as follows:

```powershell
# 1. Resample 97 candidates
..\.venv\Scripts\python.exe "data analysis\10_build_eval_set.py"

# 2. Hand-label them
#    Edit data analysis/outputs/eval_labels.csv manually.
#    Format: pair_id, label_match_type, label_is_match, label_confidence, label_notes

# 3. Re-run the benchmark
..\.venv\Scripts\python.exe "data analysis\11_eval_model_benchmark.py"
```

**Never overwrite `eval_labels.csv` blindly.** The labels are human
work and cannot be regenerated automatically.

---

## When to consult this folder vs the planning docs

- **Want to understand the *data*?** → here.
- **Want to understand the *design*?** → `../PLAN.md`, `../PLAN2.md`.
- **Want to understand the *audit*?** → `../FINAL_PLAN.md`, `../FINAL_AGENTIC_PLAN.md`.
- **Want to understand the *code*?** → `../SYSTEM 1 MVP/DEEPDIVE.md`, `../SYSTEM 2 RAG/DEEPDIVE.md`.
- **Want to understand the *execution history*?** → `../EXECUTED_PLAN.md`.
