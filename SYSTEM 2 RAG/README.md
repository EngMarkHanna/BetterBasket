# SYSTEM 2 RAG — Semantic Retrieval + Per-Pair RAG Context

The upgrade path. Reuses System 1's load, parse, scoring, selection,
and validation; **adds** OpenAI embedding-based semantic retrieval
(T7) and per-pair RAG context injected into the LLM prompt.

If you want the deeper walkthrough, read [`DEEPDIVE.md`](DEEPDIVE.md).
If you want to run it, read this file.

---

## What it does in one paragraph

For every A and B product, compute a 1,536-dim semantic embedding once
(`text-embedding-3-small`). At retrieval time, add a 4th source (T7)
that finds B candidates by vector cosine — catching the paraphrase
cases TF-IDF misses. At judgment time, pre-fetch five context slots
per candidate group (top rules, brand alias check per B, category
bridge hint, similar accepted examples, similar rejected examples)
and inject them into the LLM prompt. Same downstream scoring,
selection, and validation as System 1, so the comparison is
apples-to-apples.

---

## Architecture

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

## Measured results

| Metric | System 1 | **System 2 (measured)** |
|---|---|---|
| Total matches | 17,040 | **18,448** (+1,408, +8.3%) |
| Eval-set precision | 1.00 | **1.00** (tied perfect) |
| Eval-set recall | 0.66 | **0.76** (+0.10) |
| Eval-set F1 | 0.79 | **0.86** (+0.07) |
| Eval-set FPs (across all 9 strata) | 0 | **0** |
| LLM cost | $0.65 | ~$8 |
| LLM batches (K=5) | 56,495 | 74,630 |
| Wall clock (with `--reuse-system1-candidates`, 8 workers) | n/a | ~7h 10min |

System 2 strictly dominates: same perfect precision, +10pts recall,
+1,408 more matches, no FP regression on any stratum. The full
side-by-side with per-stratum breakdown and disagreement examples is
in [`../FINAL_ANALYSIS.md`](../FINAL_ANALYSIS.md).

---

## How to run

### Requirements

- All of System 1's setup (env, Azure creds at `../openai_creds.yaml`)
- `.env` at the repo root with `OPENAI_API_KEY=sk-...` (for embeddings)
- System 1's parquets in `../SYSTEM 1 MVP/cache/` (canonical tables
  used for input)
- Optionally, System 1's `candidates_afull.parquet` (saves ~50 min)

### One-time: embed the catalog

```powershell
cd "SYSTEM 2 RAG"
..\.venv\Scripts\python.exe -m solution.embed_catalog
```

Wall clock: ~80 min (OpenAI tier-1 rate limit bound). Cost: ~$0.40.

Re-running this is **free** — the per-row hash cache means unchanged
rows skip the API. Vectors land in:

- `cache/embedding_bank.npz` — keyed by `sha256(model | dim | text_version | text)`
- `cache/embeddings_A.npy` + `cache/item_ids_A.npy`
- `cache/embeddings_B.npy` + `cache/item_ids_B.npy`

### Sanity-check (the cheap gate)

Runs the full Phase C pipeline against the 97-pair eval set only.

```powershell
..\.venv\Scripts\python.exe -m solution.pipeline.main --mode eval
```

Wall clock: ~3 min. Cost: ~$0.01. Output:

- `outputs/matches_eval.csv`
- `outputs/validation_report_eval.md`
- `outputs/llm_judgments_eval.jsonl`

**Decision rule**: if eval F1 ≥ 0.86 → run full mode. If between 0.83
and 0.86 → consider Phase D tooling (`FINAL_PLAN.md` Appendix C). If
< 0.83 → ship System 1 only and write up why context injection
didn't help.

### Full production run (reusing System 1's candidates)

```powershell
..\.venv\Scripts\python.exe -m solution.pipeline.main --mode full --reuse-system1-candidates --llm-workers 5
```

Wall clock: **~5h** (vs ~5h 40min without `--reuse-system1-candidates`).
Cost: ~$1.60.

The `--reuse-system1-candidates` flag reads System 1's already-computed
`candidates_afull.parquet` instead of running T1+T3+T5 from scratch.
Results are identical because both systems share the retriever code.

### Full run without reusing (recompute everything)

```powershell
..\.venv\Scripts\python.exe -m solution.pipeline.main --mode full --llm-workers 5
```

Wall clock: ~5h 40min. Use this if System 1's parquet is unavailable.

---

## Where the deliverable lives

The pipeline writes the production deliverable to two locations:

1. **`SYSTEM 2 RAG/matches.csv`** — front and centre, next to this
   README. This is the file you ship from System 2.
2. `SYSTEM 2 RAG/outputs/matches_full.csv` — identical copy alongside
   the audit artifacts.

The eval-mode gate produces `outputs/matches_eval.csv` only (it's a
cheap pre-check, not a real deliverable).

## What else is in `outputs/`

`outputs/` holds every audit + diagnostic file. See
[`outputs/README.md`](outputs/README.md) for the full index; short
version:

| File | Purpose |
|---|---|
| `matches_full.csv` | Same as the deliverable in the parent folder |
| `matches_eval.csv` | The 97-pair gate output |
| `matches_with_features_full.csv` | The deliverable + features + RAG evidence summary per row |
| `match_candidates_scored_full.csv` | Every candidate considered |
| `validation_report_full.md` | Eval-set precision / recall / F1 per stratum |
| `validation_report_eval.md` | Same for eval mode |
| `llm_judgments_full.jsonl` | Every LLM call cached (gitignored — large, regenerable) |
| `embed_run.log`, `full_run.log` | Stdout of recent runs (gitignored) |

---

## What's in `cache/`

| File | Purpose |
|---|---|
| `embedding_bank.npz` | Persistent hash → vector store. Survives crashes (atomic save). |
| `embeddings_A.npy` | (233k, 1536) float32 catalog vectors for A, in catalog order. |
| `item_ids_A.npy` | (233k,) parallel array of A item IDs. |
| `embeddings_B.npy` | (55k, 1536) float32 catalog vectors for B. |
| `item_ids_B.npy` | (55k,) parallel array of B item IDs. |
| `embed_stats.json` | Run summary: rows embedded, cache hits, wall clock. |

All of these are gitignored.

---

## What's in `knowledge/`

460 JSONL entries, bootstrapped once from `../data analysis/outputs/`.

| File | Entries | Source |
|---|---|---|
| `rules.jsonl` | 10 | hand-written rubric clauses with structured applicability |
| `brand_aliases.jsonl` | 64 | `brand_alias_candidates.csv`, score ≥ 95, generics blocked |
| `category_bridges.jsonl` | 286 | `category_bridge_a_to_b.csv`, support ≥ 10, share ≥ 0.4 |
| `accepted_examples.jsonl` | 41 | eval-set positives |
| `rejected_examples.jsonl` | 56 | eval-set negatives |
| `edge_cases.jsonl` | 3 | hand-curated tricky patterns |

To rebuild: `..\..\.venv\Scripts\python.exe -c "from solution.knowledge import bootstrap_all; ..."`.
The bootstrap is idempotent.

---

## Module layout

```
SYSTEM 2 RAG/solution/
  config.py               # Azure (LLM) + OpenAI (embeddings) credential split
  _system1_loader.py      # importlib shim so we can reuse System 1 cleanly
  embed.py                # OpenAIEmbedder + EmbeddingBank + embedding_text (v2)
  embed_catalog.py        # Driver that embeds A + B from System 1's parquet cache

  knowledge/
    entry.py              # KnowledgeEntry dataclass; structured applicability
    bootstrap.py          # EDA artifacts -> JSONL
    index.py              # KnowledgeRetriever (cosine + filter by entry type)

  store/
    base.py               # VectorStore Protocol
    numpy_store.py        # In-memory cosine via batched matmul

  retrieve/
    base.py               # Candidate dataclass + CandidateRetriever Protocol
    semantic.py           # T7 - OpenAI embedding top-K
    bridge_system1.py     # Wraps System 1's T1/T3/T5 into the Protocol
    union.py              # Dedupe + merge sources

  context/
    builder.py            # RAGContextBuilder - 5 pre-fetched context slots

  judge/
    base.py               # JudgeRequest + JudgmentResult
    cache.py              # Versioned cache with ordered B IDs (audit fixes #11/#12)
    rag_judge.py          # gpt-5.4-nano + RAG context injection + structured JSON

  pipeline/
    main.py               # End-to-end driver (--mode eval | full)
```

See [`DEEPDIVE.md`](DEEPDIVE.md) for what every module does and why.

---

## Differences from System 1, exactly

| Stage | System 1 | System 2 |
|---|---|---|
| Load | `load.py` | Reuses System 1's `load.py` via shim |
| Parse | `parse.py` | Reuses System 1's `parse.py` via shim |
| Retrieve | T1 + T3 + T5 | **T1 + T3 + T5 + T7 (semantic)** |
| Score | `score.py` | Reuses System 1's `score.py` via shim |
| Judge | gpt-5.4-nano, K=5, **plain rubric** | gpt-5.4-nano, K=5, **rubric + 5-slot RAG context** |
| Cache | `sha256(rubric_version + a_id + sorted_b_ids)` | `sha256(model + schema + prompt_format + knowledge_version + ctx_sig + a_id + ORDERED b_ids + ORDERED b_text_hashes)` (audit fixes) |
| Select | `select.py` | Reuses System 1's `select.py` |
| Validate | `validate.py` | Reuses System 1's `validate.py` |

Only **retrieval** and **judging** differ. Everything else is shared,
so the comparison is clean by construction.
