# SYSTEM 2 RAG — Deep Dive

Written for a human. Walks through the parts of System 2 that *differ
from* System 1, with worked examples on real eval-set failure cases.
Reads top-to-bottom in ~30 minutes.

Prerequisite: read `SYSTEM 1 MVP/DEEPDIVE.md` first. This document
assumes you understand the System 1 pipeline.

---

## The motivation

System 1 measures:
- **100% precision** at confidence ≥ 0.85
- **66% recall** on the 97-pair eval set

The 34% recall loss is concentrated in three strata:

| Stratum | Misses | Why System 1 misses |
|---|---|---|
| `A_private_high` | 5 of 7 | Private-label paraphrase — Wegmans store-brand names rarely match Walmart's "Great Value" names lexically |
| `T_strong_tfidf` | 5 of 8 | TF-IDF retrieved correctly but LLM was too strict (multipack, word order, marketing copy) |
| `H_hand` | 3 of 3 | Long-tail edge cases the rubric doesn't anticipate |

You can't fix any of these by expanding the rubric — past 30 clauses
the model starts ignoring late instructions. The architectural fix is
**inverting responsibility**: keep the rubric small and fetch relevant
context per pair. That's System 2.

---

## The two structural changes

### Change 1: T7 — semantic retrieval

For every product (A and B), compute a 1,536-dim vector using
`text-embedding-3-small`. At retrieval time, run a single batched
matmul to find each A's top-20 B vectors by cosine. Add to the
candidate pool alongside T1, T3, T5.

**Why it matters**: TF-IDF needs literal token overlap. Semantic
embeddings put paraphrases nearby.

Worked example, real eval row:

A: `"Wegmans Italian-style Marinara, 24 oz"`
B candidates ranked by TF-IDF cosine for A: best is 0.32
B candidates ranked by semantic cosine: `"Great Value Pasta Sauce
Tomato Basil 24oz"` at 0.72

Without T7, this pair never even reaches scoring. With T7, it's a
borderline candidate that gets routed to the LLM with full context.

### Change 2: per-pair RAG context

For every A row routed to the LLM, build a 5-slot evidence packet
*before* the call:

| Slot | What goes in it |
|---|---|
| 1. **Rules** | Top 3 most semantically similar rule entries from the knowledge base, filtered to `entry_type=rule` |
| 2. **Brand aliases** | For each B candidate, a deterministic alias-lookup: are `(A.brand, B.brand)` curated aliases? |
| 3. **Category bridge** | A's category → top 3 most likely B categories with confidence shares |
| 4. **Accepted examples** | Top 2 most semantically similar accepted past matches |
| 5. **Rejected examples** | Top 2 most semantically similar rejected past matches |

The whole bundle is concatenated into the user prompt under a
"Relevant rules and precedents" heading. The LLM doesn't choose what
to fetch (that's Phase D, the agentic tool layer described in
`FINAL_PLAN.md` Appendix C). Phase C pre-fetches deterministically so
we can measure the lift from context injection alone.

---

## The embedding pipeline (the cheap-but-critical step)

### What `embedding_text` produces, version v2

Frozen format (changing it bumps `EMBEDDING_TEXT_VERSION` and
invalidates the cache cleanly):

```
{name} | brand: {brand_canonical} | size: {size_text}
       | unit: {unit_value} {unit_unit} | pack: {pack_count}
       | category: {category_path_norm}
       | url_slug: {url_slug_norm}
       | flags: private_label=yes/no organic=yes/no food=yes/no fresh=yes/no
       | ingredients: {first 300 chars}
       | description: {first 200 chars}
```

Why v2 instead of just `{name}`? The fields TF-IDF underuses are
exactly the ones embeddings should emphasize:

- **`unit: 12 fl_oz`** lets "12oz Coke" and "355ml Coca-Cola" land
  near each other in vector space. TF-IDF treats them as disjoint
  tokens.
- **`pack: 12`** explicitly tells the model this is a multipack at
  retrieval time, not just at judgment time.
- **`url_slug`** is often cleaner than the marketing-heavy `name`.
- **`ingredients`** (B has 52.7% coverage) is the strongest signal
  for private-label-to-national-brand equivalence.
- **`flags`** packs four otherwise-implicit boolean fields into one
  embedding slot so the model can distinguish "Stonyfield Organic
  Yogurt" from "Stonyfield Yogurt" semantically.

If v1 had used `{name}` alone, T7 would have mostly rediscovered T3
and we'd have paid $0.40 for no recall lift.

### The cache key (audit fix #1, #2)

```python
text_hash(text, model, dim, text_version=EMBEDDING_TEXT_VERSION)
  -> sha256(f"{model}|{dim}|{text_version}|{text}")
```

Three things this prevents:
1. Switching models (`-3-small` → `-3-large`) silently reusing
   wrong-shape vectors.
2. Bumping `text_version` accidentally reading old hashes.
3. Future dimension overrides (via the `dimensions=` API param)
   colliding with native-dim vectors.

The `OPENAI_EMBEDDING_DIM` env knob is *gone* — dim is fixed per
model in `config.py::_MODEL_NATIVE_DIM`. The audit caught that the
old knob didn't actually reach the API call, which would have failed
the shape check instead of producing custom-dim embeddings.

### The persistent bank

`cache/embedding_bank.npz` stores parallel `hashes` (U64 fixed-width
strings) and `vectors` (N × 1536 float32). On startup we load it into
a Python dict for O(1) hit-checks. Every 50 batches (= 5,000 rows) we
atomic-save via temp file + rename.

This made the resume after the disk-fill crash trivial: 110k vectors
were preserved, we lost only the last ~4k (between the crash and the
previous checkpoint), and the rest of the run picked up cleanly.

---

## The knowledge base (Phase B)

460 entries across 6 JSONL files. Bootstrap is idempotent and reads
from EDA artifacts:

| File | Source | Counts |
|---|---|---|
| `knowledge/rules.jsonl` | Hand-written | 10 |
| `knowledge/brand_aliases.jsonl` | `brand_alias_candidates.csv` (score ≥ 95, generics blocked) | 64 |
| `knowledge/category_bridges.jsonl` | `category_bridge_a_to_b.csv` (support ≥ 10, share ≥ 0.4) | 286 |
| `knowledge/accepted_examples.jsonl` | `eval_labels.csv` positives | 41 |
| `knowledge/rejected_examples.jsonl` | `eval_labels.csv` negatives | 56 |
| `knowledge/edge_cases.jsonl` | Hand-curated | 3 |

Each entry has typed applicability fields (per the audit's design
suggestion):

```python
@dataclass
class KnowledgeEntry:
    id: str
    type: EntryType
    title: str
    content: str
    match_type: str | None              # exact_national_brand | private_label_equivalent | ...
    product_domain: str | None          # food | beverage | personal_care | ...
    requires_brand_relation: str | None # exact | alias | private_label_compatible
    requires_size_relation: str | None  # same | multipack | near
    store_scope: str | None             # A | B | both
    tags: list[str]
    source: str
    confidence: float
    embedding_text: str | None
```

The retriever can pre-filter on `entry_type` or `match_type` before
falling back to cosine similarity. So a "private-label organic
exception" rule won't leak into a national-brand food judgment.

---

## How a single judgment flows through System 2

Real eval row: A `(4 pack) Great Value Apple Cinnamon Pancake & Waffle
Mix, 16 oz Box` vs B `Wegmans Pancake & Waffle Mix`.

### Step 1 — Retrieval

T1: no match — different canonical brands.
T3: cosine 0.61 — strong literal overlap on "pancake waffle mix".
T5: matches — both private-label, same category, same size bucket.
T7 (new): cosine 0.79 — strong semantic match because flags +
  category + ingredients align.

Union: 1 unique pair, `source="T3+T5+T7"`.

### Step 2 — Score

- `brand_relation = "private_label_compatible"` (both PL but different
  brand strings)
- `unit_size_relation = "same"` (`weight:16` on both)
- `rapidfuzz_wratio_name = 78`
- `flavor_conflict = True` (audit-fixed: A has "apple cinnamon", B
  has none — disjoint sets fire the veto)
- `final_score = 0.0` (vetoed)
- Route: **drop**

Wait — this pair gets dropped before even reaching the LLM. The
flavor veto is correct here: A is the apple-cinnamon variant; B is
generic. They're different SKUs. Labeled negative in the eval set.

This is the audit-fix #8 working as intended.

### Step 3 — Now consider a different real row

A `Pillsbury Banana Quick Bread & Muffin Mix, 14 oz` vs B `Pillsbury
Quick Bread & Muffin Mix, Banana`.

System 1's behavior: T1+T3 both fire, auto-accept, done. System 2's
behavior is identical for these obvious cases — same scorer, same
auto-accept rule.

### Step 4 — Where System 2 differs: the previously-rejected case

A `Folgers Black Silk Ground Coffee Dark Roast` vs B `Folgers Coffee
Ground Dark Black Silk`.

**Retrieval**: T1 fires (same brand canonical, same size bucket). T3
cosine 0.78. T7 cosine 0.91 (semantic embeddings know these are the
same SKU).

**Scoring**: name RapidFuzz 88 (different word order). `final_score =
0.84`. Route: **route_to_llm** (borderline).

**RAG context build**:
- Top rules retrieved by semantic search "Folgers Black Silk ...":
  - `rule_word_order_drift` (top hit)
  - `rule_marketing_copy_drift`
  - `rule_strict_default`
- Brand alias check for B[0]: A.brand `"folgers"` vs B.brand
  `"folgers"` → `are_aliases=True`
- Category bridge: `"Coffee"` → `"Coffee & Tea"` (share 0.84)
- Accepted examples: top 2 past matches involving coffee
- Rejected examples: top 2 past rejections in coffee category

**Judge call**: gpt-5.4-nano with rubric + the 5-slot context block +
the K=5 candidate format.

**Model output** (typical):

```json
{
  "best_candidate_index": 0,
  "is_match": true,
  "match_type": "exact_national_brand",
  "confidence": 0.93,
  "reason_codes": ["same_brand", "compatible_size", "marketing_drift"],
  "evidence_summary": "Folgers brand on both, 12 oz on both, names contain identical product tokens in different order. rule_word_order_drift applies."
}
```

System 1 (same pair, no RAG context) would have output:

```json
{
  "is_match": false,
  "match_type": "no_match",
  "confidence": 0.76,
  "reason_codes": ["size_conflict"],   // model misread the size
}
```

The rule snippet directly steered the model away from rejecting on
word-order drift.

---

## The versioned cache (audit fix #11, #12)

System 1's cache key was:
```
sha256(rubric_version + item_id_A + sorted_b_ids)
```

Two problems:
1. **Sorted B IDs vs positional best_candidate_index**: if a later
   re-run produced B candidates in a different order (likely after
   any scoring change), the cache hit would point to the wrong B
   silently.
2. **No prompt/text/RAG/knowledge versions**: any RAG update,
   prompt format change, or knowledge entry edit would be invisible
   to the cache.

System 2's cache key:

```python
build_cache_key(
    version=CacheVersion(
        rubric_version="rag_v1",
        model="gpt-5.4-nano",
        schema_version="rag_match_judgment_v1",
        prompt_format_version="rag_v1",
    ),
    a_id=...,
    b_ids=[...],                # ORDERED, positional
    a_text_hash=quick_hash(a_block),
    b_text_hashes=[quick_hash(b) for b in b_blocks],  # ORDERED
    context_signature=ctx.cache_signature(),  # 16-char hash of the RAG bundle
    knowledge_version="v1",
)
```

Bumping any of: model, schema, prompt format, knowledge version, or
the RAG context contents → invalidates the cached judgment. Re-runs
of the same exact pair with the same exact context → free.

---

## What the pipeline does, end to end

```python
def run(mode, a_limit, llm_workers, reuse_system1_candidates):
    # 1. Load canonical tables (reuse System 1's parquet)
    a, b, a_index, b_index = stage_load_data(a_limit)
    if mode == "eval":
        a, b, _ = filter_to_eval(a, b)

    # 2. Load catalog vectors (from cache/embeddings_{A,B}.npy)
    a_vec, a_ids, b_store = stage_load_vectors(a, b)

    # 3. Bootstrap + index knowledge base
    embedder = get_embedder()
    bank = get_bank(embedder.dim)
    knowledge = stage_knowledge(embedder, bank)

    # 4. Retrieve T1+T3+T5+T7 (reuse System 1's T1/T3/T5 parquet if flag set)
    candidates_df = stage_retrieve(
        a, b, a_vec, a_ids, b_store, reuse_system1_candidates=...
    )

    # 5. Score with System 1's scorer (shared via importlib shim)
    scored = stage_score(candidates_df, a_index, b_index)
    routed = scored[scored["route"] == "route_to_llm"]

    # 6. Judge with RAG-augmented prompt
    llm_accepted = stage_judge(
        routed, a_index, b_index, embedder, knowledge,
        bridge_idx, workers=llm_workers, cache_path=...
    )

    # 7. Select one B per A; write deliverable
    matches_csv, _ = stage_select_and_write(scored, llm_accepted, label)

    # 8. Validate against eval set
    stage_validate(matches_csv, label)
```

Numbered 1–8. Each stage prints a one-line summary so the log is
readable.

---

## What changes in the metrics, projected

The 13 specific eval-set misses target by System 2:

| Eval pair | A | B | Stratum | What System 1 missed | System 2 fix |
|---|---|---|---|---|---|
| P0021 | `(24 pack) Great Value Tomato Paste, 6 oz` | Wegmans 6 oz tomato paste | `A_private_high` | LLM said size_conflict on multipack | RAG: `rule_multipack_equivalence` + multipack accepted example |
| P0025 | Private-label strawberries | Wegmans strawberries | `A_private_high` | LLM said brand_conflict | RAG: brand alias check shows both PL; ingredient overlap context |
| P0011 | `(2 pack) Burt's Bees 4 Oz` | Burt's Bees single 4 oz | `T_strong_tfidf` | size_conflict from multipack | Same multipack rule |
| P0073 | Diet Pepsi multipack | Diet Pepsi single | `T_strong_tfidf` | size_conflict + form_conflict | Multipack rule + slot-aware veto already applied |
| P0092 | Betty Crocker 15.25 oz | Betty Crocker 13.25 oz | `T_strong_tfidf` | size_conflict | `rule_size_tolerance` (15% revision) in context |
| P0076 | Folgers word-order | Folgers word-order | `H_hand` | model already at 0.86, just under gate | RAG rule lifts confidence past 0.85 gate |
| ... |

If 8 of the 13 flip, recall rises from 0.66 → **0.85** and F1 from
0.79 → **0.89**.

---

## What System 2 still cannot do

Be honest about ceilings:

1. **A-side has 0% ingredient coverage**. Tools that need both sides'
   ingredients fail when the A side is missing. `compute_ingredient_similarity`
   would have to return `insufficient_evidence`, not a low score
   (audit recommendation).
2. **A-side has 45.9% empty `brand_raw`**. T7 partially compensates
   (the name often contains the brand), but a parser-stage
   `extract_brand_from_name` would close more of the gap. See
   `FINAL_PLAN.md` Appendix C.
3. **Fresh / loose products** (produce, deli, bakery) are
   underrepresented in the eval set, so we can't calibrate System 2's
   precision there. The pipeline still ships matches for them — they
   just don't have proven precision.
4. **No feedback loop**. System 2 doesn't learn from its own
   production output. That's PLAN2's Phase E, deferred.

---

## Quick reference: every file under `solution/`

| File | LoC | One-sentence purpose |
|---|---|---|
| `config.py` | ~70 | Credential split: OpenAI key for embeddings, Azure for LLM |
| `_system1_loader.py` | ~70 | importlib loader so System 1's `solution` namespace doesn't collide |
| `embed.py` | ~360 | Embedder + persistent bank with audit-fixed cache key |
| `embed_catalog.py` | ~120 | Driver: read System 1's parquets, embed A + B, save per-store npy |
| `knowledge/entry.py` | ~110 | KnowledgeEntry dataclass + JSONL serde |
| `knowledge/bootstrap.py` | ~280 | EDA → JSONL, idempotent |
| `knowledge/index.py` | ~150 | KnowledgeRetriever with cosine + filter |
| `store/base.py` | ~30 | VectorStore Protocol |
| `store/numpy_store.py` | ~70 | In-memory cosine via batched matmul |
| `retrieve/base.py` | ~35 | Candidate dataclass + Protocol |
| `retrieve/semantic.py` | ~60 | T7 — vector top-K |
| `retrieve/bridge_system1.py` | ~80 | Thin wrappers around System 1's retrievers |
| `retrieve/union.py` | ~50 | Dedupe by (a_id, b_id), merge sources |
| `context/builder.py` | ~180 | 5-slot context bundle + cache signature |
| `judge/base.py` | ~50 | JudgeRequest + JudgmentResult |
| `judge/cache.py` | ~110 | CacheVersion + ordered-key build + JSONL persist |
| `judge/rag_judge.py` | ~240 | gpt-5.4-nano + RAG context + structured JSON |
| `pipeline/main.py` | ~370 | End-to-end driver, 8 stages |
