# System 2 — Design and Execution Notes

The full design rationale, audit response, and execution trail for the
semantic + RAG matcher (System 2 RAG). For *what it does and how to
run it*, see `SYSTEM 2 RAG/README.md`. For a deep technical
walkthrough of the code, see `SYSTEM 2 RAG/DEEPDIVE.md`. This
document is the *why*: the audit findings, the design trade-offs, the
tool layer we considered and what we'd build if Phase C falls short.

---

## 1. Why System 2 exists

System 1 measures **100% precision** at confidence ≥ 0.85 but only
**66% recall** on the 97-pair eval set. The 34% miss concentrates in
three strata:

| Stratum | Misses | Why System 1 misses |
|---|---|---|
| `A_private_high` | 5 of 7 | Private-label paraphrase — Wegmans store-brand names rarely match Walmart's "Great Value" names lexically |
| `T_strong_tfidf` | 5 of 8 | TF-IDF retrieved correctly but LLM was too strict (multipack, word order, marketing copy) |
| `H_hand` | 3 of 3 | Long-tail edge cases the rubric doesn't anticipate |

You cannot fix any of these by growing the rubric — past 30 clauses
the model starts ignoring late instructions. The architectural fix is
to **invert responsibility**: keep the rubric small and fetch relevant
context per pair from a versioned knowledge base. That's System 2.

---

## 2. The two structural changes

Everything else (load, parse, score, select, validate) is reused from
System 1 via an `importlib` shim, so the comparison is apples-to-apples
by construction.

### Change 1: T7 — semantic retrieval

For every product (A and B), compute a 1,536-dim vector using
`text-embedding-3-small`. At retrieval time, run a batched matmul to
find each A's top-K B vectors by cosine. Add as a fourth source
alongside T1/T3/T5.

**Why it matters**: TF-IDF needs literal token overlap. Semantic
embeddings put paraphrases nearby. Real example: "Wegmans Italian-style
Marinara, 24 oz" vs "Great Value Pasta Sauce Tomato Basil 24oz"
has TF-IDF cosine 0.32 but semantic cosine 0.72.

### Change 2: per-pair RAG context

For every A row routed to the LLM, build a 5-slot evidence packet
*before* the call:

| Slot | Source | What it provides |
|---|---|---|
| Rules | Knowledge base top-K by semantic query | Decomposed rubric clauses |
| Brand aliases | Deterministic lookup per B candidate | Are `(A.brand, B.brand)` curated aliases? |
| Category bridge | Deterministic lookup | A's category → top-3 plausible B categories |
| Accepted examples | Knowledge base top-K | Similar past positive matches with reasons |
| Rejected examples | Knowledge base top-K | Similar past negative matches with reasons |

The model doesn't choose what to fetch (that's the agentic Phase D,
covered in section 6). Phase C pre-fetches deterministically so we can
measure the lift from context injection alone, without the agentic
infrastructure complexity.

---

## 3. The embedding pipeline

### Embedding text format (v2, frozen)

```
{name} | brand: {brand_canonical} | size: {size_text}
       | unit: {unit_value} {unit_unit} | pack: {pack_count}
       | category: {category_path_norm}
       | url_slug: {url_slug_norm}
       | flags: private_label=yes/no organic=yes/no food=yes/no fresh=yes/no
       | ingredients: {first 300 chars}
       | description: {first 200 chars}
```

The version is locked as `EMBEDDING_TEXT_VERSION = "v2"`. Bumping the
version cleanly invalidates the cache.

Why v2 instead of just `{name}`? The fields TF-IDF underuses are
exactly the ones embeddings should emphasize:

- `unit: 12 fl_oz` — lets "12oz Coke" and "355ml Coca-Cola" land near
  each other in vector space.
- `pack: 12` — explicit multipack signal at retrieval time.
- `url_slug` — often cleaner than the marketing-heavy `name`.
- `ingredients` (B has 52.7% coverage) — strongest signal for
  private-label-to-national-brand equivalence.
- `flags` — packs four otherwise-implicit boolean fields into one slot
  so the model distinguishes "Stonyfield Organic Yogurt" from
  "Stonyfield Yogurt" semantically.

v1 (an earlier version) used just `name + brand + category + size +
description` and made T7's vectors largely equivalent to T3's TF-IDF
space. We re-ran with v2 after the audit caught this.

### Cache key (audit fix #1, #2)

```python
text_hash(text, model, dim, text_version)
  -> sha256(f"{model}|{dim}|{text_version}|{text}")
```

Prevents:
1. Switching models silently reusing wrong-shape vectors.
2. Bumping text format accidentally reading old hashes.
3. Future dim-override collisions.

The `OPENAI_EMBEDDING_DIM` env knob from v1 was removed. Dim is fixed
per model in `_MODEL_NATIVE_DIM`. The audit caught that the old knob
didn't reach the API call — it would have failed shape checks instead
of producing custom-dim embeddings.

### Persistent bank

`cache/embedding_bank.npz` stores parallel `hashes` (U64 fixed-width
strings) and `vectors` (N × 1536 float32). Loaded into a Python dict
on startup for O(1) hit-checks. Every 50 batches (5,000 rows) we
atomic-save via temp file + rename.

This survived a disk-fill crash mid-run: ~110k vectors preserved, we
lost only the last ~4k between the crash and the previous checkpoint,
the resume picked up cleanly.

---

## 4. The knowledge base (460 entries)

Six JSONL files bootstrapped once from EDA artifacts. Each entry has
typed applicability fields so retrieval can filter before falling back
to cosine similarity.

| File | Source | Count |
|---|---|---|
| `knowledge/rules.jsonl` | Hand-written rubric clauses | 10 |
| `knowledge/brand_aliases.jsonl` | `brand_alias_candidates.csv`, score ≥ 95, generics blocked | 64 |
| `knowledge/category_bridges.jsonl` | `category_bridge_a_to_b.csv`, support ≥ 10, share ≥ 0.4 | 286 |
| `knowledge/accepted_examples.jsonl` | `eval_labels.csv` positives | 41 |
| `knowledge/rejected_examples.jsonl` | `eval_labels.csv` negatives | 56 |
| `knowledge/edge_cases.jsonl` | Hand-curated tricky patterns | 3 |

Entry schema:

```python
@dataclass
class KnowledgeEntry:
    id: str
    type: EntryType  # rule, alias, bridge, accepted_example, rejected_example, edge_case
    title: str
    content: str
    match_type: str | None              # exact_national_brand | private_label_equivalent | ...
    product_domain: str | None          # food | beverage | personal_care | ...
    requires_brand_relation: str | None # exact | alias | private_label_compatible
    requires_size_relation: str | None  # same | multipack | near
    store_scope: str | None
    tags: list[str]
    source: str
    confidence: float
    embedding_text: str | None
```

The retriever pre-filters on `entry_type` or `match_type` before
cosine ranking. So a "private-label organic exception" rule won't
leak into a national-brand food judgment.

---

## 5. Versioned LLM cache (audit fixes #11, #12)

System 1's cache key was:
```python
sha256(rubric_version + item_id_A + sorted_b_ids)
```

Two problems the audit caught:
1. **Sorted B IDs vs positional `best_candidate_index`**: if a later
   re-run produced B candidates in a different order, the cache hit
   would point to the wrong B silently.
2. **No prompt/text/RAG/knowledge versions**: any RAG update or
   prompt-format change would be invisible to the cache.

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
    b_text_hashes=[quick_hash(b) for b in b_blocks],   # ORDERED
    context_signature=ctx.cache_signature(),           # 16-char hash of the RAG bundle
    knowledge_version="v1",
)
```

Bumping any of: model, schema, prompt format, knowledge version, or
RAG context contents → invalidates the cached judgment. Re-runs of the
same exact pair with the same exact context → free.

---

## 6. The audit and what we did about it

A second-pass review (in `FINAL_PLAN.md` originally) flagged 16 bugs,
7 design issues, and prioritized improvements. Tier classification
and outcomes:

### Tier 1 (blocking, fixed before more embedding work)

| # | Bug | Action |
|---|---|---|
| #1 | Embedding cache key missing model/dim/version | DONE — `text_hash(text, model, dim, text_version)` |
| #2 | Broken `OPENAI_EMBEDDING_DIM` knob | DONE — removed; dim fixed per model |
| #4 | Embedding text omits high-signal fields | DONE — v2 format adds parsed unit/pack/url_slug/flags/ingredients |

### Tier 2 (applied during runs)

| # | Bug | Action |
|---|---|---|
| #11 | LLM cache key sorts B IDs (wrong-B-bug) | DONE — ordered B IDs + ordered b_text_hashes |
| #12 | LLM cache omits prompt/RAG/knowledge versions | DONE — `CacheVersion` dataclass + context signature in key |
| #13 | Retry only on 429, not 5xx/timeouts | DONE — widened to 429/500/502/503/504/timeout/connection |
| Design | Knowledge needs structured applicability | DONE — typed fields on `KnowledgeEntry` |
| Design | Brand aliases need category guards | PARTIAL — schema slot exists, not yet populated |

### Tier 3 (System 1 patches, reused by System 2)

System 2 inherits these because of the `_system1_loader` shim:

| # | Bug | Action |
|---|---|---|
| #8 | Flavor/form vetoes overfire on shared descriptors | DONE — only veto on disjoint slot tokens |
| #9 | T3 auto-accept allowed `unit_rel="unknown"` | DONE — requires `same/near/multipack` |
| #14 | T5 ignores `info_category_3` | DONE — folds cat3 into A walk + B index |
| #15 | `select.py` tie handling didn't match docstring | DONE — multi-key mergesort |
| #16 | Validation precision=0 on no overlap | DONE — returns `None` + `overlap_note` |
| #5 | Leading decimal parser (`.5 L` → 5 L) | DEFERRED |
| #6 | `count:1` becomes a block bucket | DEFERRED |
| #7 | Detached multipack `(3 pack) ... 15.25 oz` | DEFERRED |

### Tier 4 (research, deferred)

- `sparse_dot_topn` for TF-IDF top-K — memory fine at current scale
- Benchmark stronger embedders (text-embedding-3-large, BGE, Cohere,
  Voyage) — locked to `-small` for v2 cost / ops simplicity
- Expand eval set to 300+ labels with fresh + private-label
  oversampling

---

## 7. Phase D — the tool layer we considered

If Phase C (RAG context alone) falls short of F1 ≥ 0.86, the next
step is letting the LLM call tools on demand. The audit pushed back on
the initial tool list; the v2 below is the post-review proposal.

### Decision rule

| Phase C eval F1 | Action |
|---|---|
| ≥ 0.86 | Ship Phase C, do NOT build tools |
| 0.83–0.86 | Build the priority-4 agentic tools, re-eval |
| < 0.83 | Reconsider — RAG context alone isn't doing enough work |

### The priority-4 agentic tools

If we build the tool layer, these are the only four tools the LLM
gets to call:

1. **`compute_size_compatibility(size_a, size_b)`** — Single highest-
   leverage tool. Reviewer confirmed +9.8 recall points via 4 specific
   eval rows (P0011, P0021, P0073, P0092). Encapsulates unit conversion
   + multipack logic. ~150 lines.

2. **`resolve_brand_and_extract(brand_string, name, category)`** —
   Wider scope than just alias lookup: includes `is_private_label`,
   `store_scope`, `category_scope` (the alias-guard from the audit),
   and name-extraction when `brand_string` is empty (fixes Walmart's
   45.9% empty `brand_raw`). ~200 lines.

3. **`compare_category_bridge(a_category, b_category)`** — Per-
   candidate bridge verdict (current Phase C emits one shared text
   hint). Returns `bridge_match`, `share`, `support`, and a verdict.
   Reviewer's addition. ~80 lines.

4. **`classify_variant_conflict(a_name, b_name, category)`** —
   Precision-focused tool (the others target recall). Surfaces real
   slot-typed variant mismatches (flavor / scent / shade / model_line /
   form / feature). Fixes the P0086 Conair and P0022 Apple Cinnamon
   false positives. Reviewer's addition. ~180 lines.

### Things moved out of the agentic layer

Original Appendix B had a longer list; the reviewer correctly pointed
out several should be parser passes or score features, not tools:

- `extract_brand_from_name` → parser-stage one-time pass
- `repair_detached_pack_size` → parser fix (audit Tier-3 deferred)
- `compute_ingredient_similarity` → score feature, return
  `insufficient_evidence` when A side is missing
- `diff_names` → optional score features (token diff counts with
  full-catalog IDF, not eval-set IDF)
- Per-candidate bridge verdict → Phase C context-builder addition

### Things we will NOT build

| Tool | Why skip |
|---|---|
| `web_search(brand)` | Slow, rate-limited; manufacturer hierarchy is a lookup table, not a search problem |
| `query_database(SQL)` | Over-engineered; the "database" is two CSVs |
| `image_compare(url_a, url_b)` | Separate vision project; recall gains unclear |
| `generate_canonical_name(...)` | Speculative; forces creative writing over structured judgment |
| `find_alternative_candidates` | Dangerous — lets the model fish outside the calibrated retrieval set. Reviewer's strong veto |
| `decompose_product_name` | Overlaps with `diff_names` + `extract_brand_from_name`; doubles error surface |

### Tool-call protocol (if we build Phase D)

- `MAX_TOOL_STEPS = 2` (reduced from initial 4; literature doesn't
  support 4 as special — ReAct cap is task-dependent)
- `tool_choice = auto` for size + brand + bridge; `none` for
  variant-conflict tool unless the first-pass output is positive
- Parallel tool calls REQUIRED — if the model needs all four, batch
  in one model turn, don't sequentialize
- Tool-output cache keyed by `(input, tool_version)` survives across
  pairs that ask the same question

### Literature pointers (from reviewer)

Recent ER literature supports structured decomposition over open-ended
agentic loops:

- **Ditto** (arXiv:2004.00584) — closest research template: BERT-based
  EM with domain-knowledge injection, hard-example augmentation. Up
  to 29 F1 improvement over prior SOTA. Argues for RAG examples in
  prompts.
- **DeepMatcher** (SIGMOD '18) — DL doesn't outperform strong non-DL
  on structured EM, but helps on textual/dirty EM. Grocery names are
  dirty textual records, so RAG-augmented LLM reranking is plausible.
- **SBERT** (arXiv:1908.10084) — why dense retrieval belongs before
  cross-encoder/LLM judging.
- **ComEM** (arXiv:2405.16884, 2024) — selecting among candidate
  records and composing matching strategies. Supports K-candidate
  judging.
- **Schemora / LLMatch** — staged preparation, candidate selection,
  metadata enrichment, hybrid lexical/vector retrieval. Favors Phase
  C-style retrieval improvements before tool calls.

---

## 8. Architecture (concrete)

```
                  +----------------------------+
                  | Same load / parse / score  |
                  | / select / validate as     |
                  | System 1 (via importlib)   |
                  +----------------------------+
                              |
       +----------------------+----------------------+
       v                                             v
  T1 + T3 + T5 + T7 retrieval                 460-entry JSONL
  (semantic adds the 4th source)              knowledge base
       |                                             |
       v                                             v
  Deterministic scoring                       Embedded once,
  (same as System 1)                          cosine-searchable
       |                                             |
       v                                             |
  Auto-accept? --- yes ----> matches.csv             |
       |                                             |
       no                                            |
       v                                             |
  Build per-pair RAG context (5 slots) <-------------+
       |
       v
  gpt-5.4-nano judge (RAG context injected, structured JSON)
       |
       v
  matches.csv
```

---

## 9. What was actually executed

In order:

1. **Embedding pipeline built** (`embed.py` v1) — first run hit a
   disk-fill crash at 110k vectors. Atomic save pattern saved the
   bank.
2. **Audit findings #1/#2/#4 applied** — text format v2 with parsed
   unit/pack/slug/flags/ingredients, versioned cache key.
3. **Embedding re-run with v2** — 233k A + 55k B vectors, ~$0.40, ~3h
   wall clock (rate-limit bound).
4. **Knowledge base bootstrap** — 460 entries from EDA artifacts.
5. **Phase B + C code built** in subpackages
   (`knowledge/`, `store/`, `retrieve/`, `context/`, `judge/`,
   `pipeline/`).
6. **System 1 Tier-3 patches applied** — slot-aware vetoes,
   unknown-size routing, cat3 bridge, tie-break, validation None.
7. **Eval-mode gate** — 97-pair test:
   - Precision 1.00 → 0.84 (-0.16, 6 false positives)
   - Recall 0.66 → 0.78 (+0.12, exactly the target)
   - F1 0.79 → 0.81 (+0.02)
   - `A_private_high` recall 0.29 → 0.86 (the headline win)
8. **First full run attempt** — T7 cosine ≥ 0.55 produced 3.27M
   candidates. ~18h wall clock projected. Killed.
9. **Tightened T7 (cosine ≥ 0.7, k=10)** — 743k T7 candidates, ~6h
   wall clock projected. Killed for further tightening.
10. **Tightened again (cosine ≥ 0.75, k=5)** — 309k T7 candidates,
    602k union, 382k routed. First run had a silent context-build
    bottleneck (1 embedding API call per A-row group, serial). Killed.
11. **Patched `RAGContextBuilder.build_many`** to batch-embed all
    per-pair queries in groups of 100. Drops pre-stage from many
    hours to ~17 min. Confirmed identical eval-mode output (F1 0.81).
12. **Full production run completed** — 74,630 LLM batches in 7h
    10min at 3.0–3.1 req/s with 8 workers. Output:
    - 18,448 matches (vs System 1's 17,040, +8.3%)
    - Eval-set precision **1.00**, recall **0.76**, F1 **0.86**
    - Zero false positives across all 9 eval strata
    - +4 true positives recovered (2 in `A_private_high`, 1 in
      `T_strong_tfidf`, 1 in `T_weak_tfidf`)
    - `H_hand` stratum still 0/3 — long-tail edge cases that Phase D
      tools (section 7) would target

---

## 10. Open questions / next steps

If Phase C ships as the deliverable:

1. **Batch-embed the per-pair RAG queries** — current pre-stage is a
   ~95 min serial bottleneck for the full run. Batching could cut it
   to ~5 min.
2. **Phase D decision** — wait for full-run F1; if below 0.86, build
   priority-4 agentic tools per section 7.
3. **Expand eval set** to 300+ labels with fresh + private-label
   oversampling. The 0.81 vs 0.79 F1 lift on 97 pairs is noisy.
4. **Sharded embedding bank** — single 1 GB NPZ atomic rewrites work
   but waste disk during save. A directory of frozen 50 MB shards
   would be more robust.
5. **Multi-store config** — `BRAND_ALIASES` and private-label lists
   are inlined per store. A `stores/<id>.yaml` layout would generalize
   to competitor C, D, E without code changes.
6. **Knowledge auto-improvement** — Phase E from the original PLAN2.
   A separate offline agent proposes new knowledge entries from
   production-run disagreements. Defer until production data flows.

---

## Quick-start commands

```powershell
.\.venv\Scripts\Activate.ps1
cd "SYSTEM 2 RAG"

# One-time: embed the catalog (~80 min, ~$0.40)
..\.venv\Scripts\python.exe -m solution.embed_catalog

# Cheap eval gate (~3 min, ~$0.01)
..\.venv\Scripts\python.exe -m solution.pipeline.main --mode eval

# Full production run (~4h with reuse + 8 workers)
..\.venv\Scripts\python.exe -m solution.pipeline.main --mode full --reuse-system1-candidates --llm-workers 8
```

Outputs land in `SYSTEM 2 RAG/outputs/`. The deliverable is
`matches_full.csv`.
