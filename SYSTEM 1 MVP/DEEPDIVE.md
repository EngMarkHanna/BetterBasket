# SYSTEM 1 MVP — Deep Dive

Written for a human. Walks through every component, every decision,
with concrete examples from real eval-set rows. Reads top-to-bottom in
~25 minutes.

If you have less time: read **The big picture** and **Why we made
these choices**. Skip the rest until you hit a piece of code you need
to modify.

---

## The big picture

The problem is one of scale + ambiguity:

- **Scale**: 233,195 A × 55,516 B = **12.9 billion candidate pairs**.
  You cannot LLM-judge all of them ($500k+, years of clock time).
- **Ambiguity**: Walmart and Wegmans use different brand strings,
  different category trees, different size formats. UPC codes don't
  overlap. There's no easy join key.

System 1's strategy:

```
12.9B pairs  →  ~470k candidates  →  ~96k auto-accepted or
                (retrieval)            LLM-judged  →  17,040 matches
                                       (scoring + judge)   (selection)
```

Each arrow is a stage. Each stage gets cheaper-per-pair as we move
right. By the time we ask the LLM, we're only asking about ~96k pairs,
not 12.9B.

---

## The pipeline, stage by stage

### Stage 1 — Load (load.py)

Reads both CSVs, parses the nested `item_info` and `sizing_comp`
fields, and builds a canonical row per product with the columns
everything downstream depends on. Cached as Parquet so the second run
loads in seconds.

Example row, after load:

```python
item_id: "2069913"
name: "Pillsbury Banana Quick Bread & Muffin Mix, 14 oz"
brand_raw: "Pillsbury"
brand_canonical: "pillsbury"
unit_canonical: 14.0      # parsed from size text
unit_unit: "oz"
pack_count: 1
url_slug_norm: "pillsbury banana quick bread"
category_path_norm: "food baking mix"
is_private_label_inferred: False
is_food_like: True
```

The drop-duplicate step is important: the raw CSV has two malformed
rows where the name column overflowed and corrupted `item_id` (you'll
see this in `load.py`'s comment). We drop them at load.

### Stage 2 — Parse (parse.py)

Three responsibilities:

1. **Size parsing** with multipack support. The old EDA parser
   collapsed "12-pack 12 oz" into one number; the new one stores both
   `unit_canonical` (the per-unit size) and `total_canonical` (the
   total package size). This single change recovered 22,353 A rows
   of valid per-unit candidates.

   Worked examples:

   | Input | unit_bucket | total_bucket | pack_count |
   |---|---|---|---|
   | `12 fl oz` | `volume:12` | `volume:12` | 1 |
   | `12 x 12 fl oz` | `volume:12` | `volume:144` | 12 |
   | `4 pk 8.5 oz` | `weight:8.5` | `weight:34` | 4 |
   | `50 ct` | `count:50` | `count:50` | 50 |

2. **Brand canonicalization** via a 20-entry alias map. Examples:
   `"l oreal paris" → "l oreal"`, `"nestl toll house" → "toll house"`.
   The list is intentionally short; bigger lists let generic words
   like "good" become canonical, which creates spurious blocks.

3. **Inference flags**: `is_private_label_inferred` (via per-store
   brand denylist), `is_organic_inferred`, `is_food_like`,
   `is_fresh_like`.

### Stage 3 — Retrieve (retrieve.py)

Three retrievers, unioned and deduped by `(item_id_A, item_id_B)`.
Each carries a `source` label so the audit trail survives.

**T1 — strict brand + size blocks**

The conservative source. For each A row:
- Find B rows sharing `(brand_canonical, unit_bucket OR total_bucket)`.
- Score the names with `rapidfuzz.WRatio`.
- Keep top 3 per A row.

When A is a multipack and B is a single, the per-unit bucket matches
both → we catch it. Without that fix we'd lose ~22k legitimate pairs.

**T3 — TF-IDF top-K**

The universal recall engine. Two sklearn `TfidfVectorizer` instances
on the joint A∪B corpus:
- Word 1-grams + 2-grams (captures "kraft mac and cheese")
- Character n-grams 3-5 with `analyzer="char_wb"` (robust to typos,
  punctuation drift)

Combined into one ~530k-feature joint vector. For each A row, return
top-20 B rows by cosine with floor 0.4.

Why both analyzers? Word n-grams give semantic chunks; char n-grams
handle "ben&jerry" vs "ben & jerry's" without manual normalization.

**T5 — private-label + category bridge**

The most surgical source. For private-label products on both sides
(`is_private_label_inferred == True`), find B candidates with:
- Compatible size bucket
- Plausible B category according to the learned A→B `category_bridge`
  (built from 8,195 shared-brand seed pairs in
  `../data analysis/outputs/category_bridge_a_to_b.csv`)
- RapidFuzz name similarity ≥ 60

**Why three sources?** Different products fall through different
cracks:

| Source | Catches | Misses |
|---|---|---|
| T1 | National brands at standard sizes | Paraphrased names |
| T3 | Anything with literal token overlap | Pure paraphrase ("Marinara" vs "Tomato-Basil Sauce") |
| T5 | Store-specific brands matched by category + size + tokens | Cross-category equivalents |

Union them; you get coverage one source alone wouldn't.

### Stage 4 — Score (score.py)

For each candidate pair, compute 14 features:

| Feature | What it captures |
|---|---|
| `candidate_source` | T1 / T3 / T5 / combinations |
| `tfidf_cosine` | from the joint TF-IDF index |
| `rapidfuzz_wratio_name` | weighted fuzzy ratio on names |
| `token_jaccard_name` | name-token intersection / union |
| `brand_relation` | exact / alias / private-label-compatible / conflict / unknown |
| `unit_size_relation` | same / multipack-related / near / off / unknown |
| `total_size_relation` | same / off / unknown |
| `pack_count_relation` | matches / multipack / unknown |
| `category_token_jaccard` | category-string overlap |
| `organic_relation` | both / one / neither |
| `flavor_conflict` | does one side have disjoint flavor tokens? |
| `form_conflict` | liquid vs powder, ground vs whole bean |
| `veto_reasons` | comma-joined hard-veto flags |
| `final_score` | weighted combination in [0, 1] |

**Slot-aware vetoes (audit fix #8)**: flavor and form vetoes only fire
when the two sides have DISJOINT tokens for that slot. So
`"Cadbury Caramel Egg Chocolate"` vs `"Cadbury Caramel Egg"` does NOT
veto, because both share "caramel". A naive symmetric-difference test
would have rejected it.

**Routing bands**:

| Score | Action |
|---|---|
| Auto-accept (T1 strong: brand-aligned + size-aligned + RF ≥ 92) | Direct to matches.csv |
| Auto-accept (T3 strong: cosine ≥ 0.6 + brand-aligned + size aligned, audit #9) | Direct to matches.csv |
| Otherwise, not vetoed, not pure noise | Route to LLM |
| Vetoed OR cosine < 0.45 with no brand/size signal | Drop |

The bands are deliberately conservative: only "this is obviously
right" auto-accepts. Everything ambiguous goes to the model.

### Stage 5 — Judge (judge.py + rag.py)

The LLM is called on ~96k batches (one A row + top-5 B candidates per
batch), at K=5 batching, with `gpt-5.4-nano` on Azure OpenAI.

Why K=5 batching? The model can compare 5 candidates against one A
row in one structured call, picking the best (or none). That's 3.5×
cheaper in tokens vs five separate one-pair calls and ~3× faster.

Key flags:
- `response_format={"type":"json_schema","strict":true}` — parsing
  failures impossible by construction
- `reasoning_effort="minimal"` — measured 0 reasoning tokens on this
  deployment, no quality regression
- Bounded `ThreadPoolExecutor` with 5 workers
- Per-call sleep + exponential backoff (1/2/4/8/16/32s) on 429s

**RAG (the small kind)**: `rag.py` is a tiny TF-IDF index over 10
hand-written rules and the 97 labeled eval pairs. Before each LLM
call we retrieve top-3 most relevant snippets and inject them into
the prompt. This is *not* the same as System 2's full RAG — it's a
~200-line "show the model relevant precedents" layer. System 2
upgrades it.

**Cache**: every LLM call is keyed by `sha256(rubric_version +
item_id_A + sorted_B_ids)` and saved to `outputs/llm_judgments.jsonl`.
Re-running is free.

**Audit Tier-2 fix #11 lives in System 2**, not here, because System
1's pipeline doesn't reorder B candidates between runs.

### Stage 6 — Select (select.py)

One B per A. Sort by `final_score` desc, then `llm_confidence` desc,
then `candidate_score` desc, then `item_id_b` asc. Take the first
per A. Multiple A rows can map to the same B (legitimate when, e.g.,
distinct A SKUs map to a single fresh-produce B item).

Audit fix #15: the docstring used to claim ties used
`candidate_score`, but the code only sorted by `final_score`. Now the
sort matches the spec.

### Stage 7 — Validate (validate.py)

Joins `outputs/matches.csv` against the 97-pair labeled eval set,
computes precision/recall/F1 globally and per stratum.

Audit fix #16: when zero eval pairs overlap with shipped matches, we
return `precision=None` + `overlap_note="no shipped eval pairs"`
instead of silently reporting 0.0.

---

## Why we made these choices

### Why TF-IDF over embeddings (in System 1)?

| Property | TF-IDF | OpenAI embeddings |
|---|---|---|
| Setup | `pip install scikit-learn` | API key + rate limit + cache layer |
| Cost | $0 | ~$0.40 to embed full catalog |
| Wall clock | seconds to build, ~50 min retrieval | 80+ min to embed, ~10 min retrieval |
| Recall on literal overlap | excellent | excellent |
| Recall on paraphrase | weak | strong |

For a deliverable that must ship today, TF-IDF gets us to 4× the
required match volume with zero new ops surface. System 2 layers
embeddings on top.

### Why LLM-as-veto, not LLM-for-everything?

We measured `gpt-5.4-nano` against the eval set: at `confidence ≥
0.85` it has **100% precision** but only **74% aggregate recall**.
The model is strict; it's a great *filter* but a bad *recall saver*.

So: rules handle the obvious wins (`A_strong` stratum was labeled 100%
positive — auto-accept doesn't need the LLM). The LLM filters the
ambiguous middle. This is the inverse of asking the model about
everything, which would be 4× slower and miss legitimate matches that
the rubric is too strict about.

### Why a tiny RAG instead of a full vector index?

The rule corpus + eval examples fit in ~10 KB of text. A TF-IDF
nearest-neighbour over ~200 docs returns in <5ms. A full vector index
would be over-engineering. System 2 needs the upgrade because its
knowledge base grows past ~500 entries.

### Why no Batch API discount?

Our Azure SKU is `GlobalStandard`, not `globalbatch`. The Batch API
isn't available. At ~$0.65 for a full run, the 50% discount doesn't
matter.

---

## What we know System 1 gets wrong

Measured from the eval-set per-stratum recall:

| Stratum | Recall | What System 1 misses |
|---|---|---|
| `A_strong` | **1.00** | Nothing |
| `A_medium` | 1.00 | Nothing |
| `A_borderline` | 1.00 | Nothing |
| `A_private_high` | **0.29** | 5 private-label-equivalent positives — the LLM is too strict on cross-store private labels |
| `T_strong_tfidf` | **0.38** | 5 cases where TF-IDF retrieved correctly but the LLM rejected (likely multipack or word-order) |
| `H_hand` | **0.00** | 3 hand-crafted edge cases — exactly the long tail no rubric can cover |
| `T_weak_tfidf` | 0 | 1 weak cosine match the LLM declined |

This is the 13-positive recall gap System 2 was designed to close.

---

## What an interactive walk through one match looks like

A: `"Pillsbury Banana Quick Bread & Muffin Mix, 14 oz"` (item_id
`2069913`)

**Stage 1+2**: parsed →
`brand_canonical="pillsbury"`, `unit_bucket="weight:14"`,
`pack_count=1`, `is_food_like=True`.

**Stage 3 — Retrieve**:
- T1 finds B `"Pillsbury Quick Bread & Muffin Mix, Banana"` in the
  `("pillsbury", "weight:14")` block. RapidFuzz WRatio = 95.
- T3 finds the same B at cosine 0.71 (same brand canonical, ~7/8
  shared tokens).
- T5 doesn't fire (not private label).

Union: 1 candidate, `source="T1+T3"`, `score=95.0`.

**Stage 4 — Score**:
- `brand_relation = "exact"` (both `"pillsbury"`)
- `unit_size_relation = "same"` (both `weight:14`)
- `rapidfuzz_wratio_name = 95.0`
- `tfidf_cosine = 0.71`
- `final_score = 0.918`
- Route: **auto_accept** (brand-aligned + size-aligned + RF ≥ 92)

**Stage 5 — Judge**: skipped (auto-accept).

**Stage 6 — Select**: only one candidate for this A, picked.

**Stage 7 — Output**: `matches.csv` gets `2069913,98431`.

That whole chain takes <10ms. Most matches look like this. The
interesting cases are the ones where retrieval is ambiguous and the
judge has to decide.

---

## When to modify what

- **Want to raise recall?** Look at `score.py`'s routing bands or
  `judge.py`'s confidence threshold (currently 0.85). Be aware that
  loosening either trades precision.
- **Want to raise precision?** Add hard veto flags to `score.py`, or
  tighten the LLM rubric in `judge.py`.
- **New competitor store?** Update the private-label brand list in
  `parse.py::PRIVATE_LABEL_*` and re-run.
- **Better recall on multipacks?** The new parser already helps, but
  the audit's `repair_detached_pack_size` (`(3 pack) ... 15.25 oz`)
  is a known gap. See `FINAL_PLAN.md` audit Tier-3.

---

## Quick reference: every file in `solution/`

| File | LoC | One-sentence purpose |
|---|---|---|
| `load.py` | ~200 | Read CSVs, parse nested fields, build canonical table. |
| `parse.py` | ~250 | Size parser, brand alias, slug, flag inference. |
| `retrieve.py` | ~280 | T1 + T3 + T5 retrievers; union + dedupe. |
| `score.py` | ~370 | 14 features, slot-aware vetoes, routing bands. |
| `rag.py` | ~120 | TF-IDF over rules + eval examples for LLM prompt. |
| `judge.py` | ~280 | gpt-5.4-nano, K=5 batching, JSONL cache, backoff. |
| `select.py` | ~40 | One B per A with multi-key tie-break. |
| `validate.py` | ~130 | Eval-set comparison; per-stratum metrics. |
| `main.py` | ~250 | End-to-end driver with --mode rules/full. |
