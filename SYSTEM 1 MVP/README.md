# SYSTEM 1 MVP — TF-IDF + LLM-as-Veto Matcher

The shippable baseline. Produces the required `matches.csv`
deliverable end-to-end. Designed to be auditable, cheap, and fast to
iterate on.

If you want to understand *how* it works, read [`DEEPDIVE.md`](DEEPDIVE.md).
If you want to *run* it, read this file.

---

## What it does in one paragraph

For each of the 233k Walmart (A) products, generate candidate matches
in the 55k Wegmans (B) catalog using three retrieval sources (strict
brand+size blocks, TF-IDF top-K, private-label via category bridge).
Score every candidate with 14 deterministic features and route it: the
strongest candidates auto-accept; the borderline ones go to a strict
LLM judge (`gpt-5.4-nano`); the rest drop. Pick the best B per A and
write `matches.csv`.

---

## Measured results on full A × B

| Metric | Value |
|---|---|
| **Total matches** | **17,040** (target was 4,000 — 4.26× headroom) |
| Unique A rows in output | 17,040 (no duplicates) |
| Unique B rows referenced | 11,513 |
| Source mix | 22% rules-only, 78% LLM-confirmed |
| Eval-set precision (97 hand-labeled pairs) | **1.00** |
| Eval-set recall | 0.66 |
| Eval-set F1 | 0.79 |
| LLM cost | ~$0.65 |
| Wall clock (5 concurrent workers) | ~5h 30min |

The eval-set numbers come from joining `outputs/matches.csv` with
`../data analysis/outputs/eval_labels.csv`. Of the 27 labeled pairs
that overlap with shipped matches, all 27 are true positives.

---

## How to run

### Requirements

- Python 3.11+ virtual env activated (see top-level README for setup)
- `openai_creds.yaml` at the repo root with Azure OpenAI credentials
- The two CSVs in `../dataset/` (already there)

### Full production run

```powershell
cd "SYSTEM 1 MVP"
..\.venv\Scripts\python.exe -m solution.main --mode full --llm-workers 5
```

Time: ~5h 30min. Output: `outputs/matches.csv`.

### Cheap rules-only run (no LLM)

For sanity-checking the deterministic layer alone:

```powershell
..\.venv\Scripts\python.exe -m solution.main --mode rules
```

Time: ~55 min. Output: subset of matches (~3,400 rows) using only
auto-accept rules.

### Dev runs

`--a-limit N` caps the A catalog for fast iteration:

```powershell
..\.venv\Scripts\python.exe -m solution.main --mode full --a-limit 5000 --llm-workers 5
```

---

## Where the deliverable lives

The pipeline writes the deliverable to two locations:

1. **`SYSTEM 1 MVP/matches.csv`** — front and centre, next to this
   README. This is the file you ship.
2. `SYSTEM 1 MVP/outputs/matches.csv` — identical copy, kept alongside
   all the audit artifacts.

## What else is in `outputs/`

`outputs/` holds every audit + diagnostic file. See
[`outputs/README.md`](outputs/README.md) for a complete index, but
the short version:

| File | Purpose |
|---|---|
| `matches.csv` | Same as the deliverable in the parent folder (kept here too for audit consistency) |
| `matches_with_features.csv` | The deliverable + every feature + LLM confidence + accept source |
| `match_candidates_scored.csv` | Every candidate the system considered |
| `rejected_borderline.csv` | Close-but-dropped candidates |
| `match_audit_sample.csv` | 50 random shipped matches for manual review |
| `validation_report.md` | Eval-set precision / recall / F1 per stratum |
| `llm_judgments.jsonl` | Every LLM call cached (gitignored — large, regenerable) |
| `run_*.log` | Stdout of the most recent runs (gitignored) |

---

## Module layout

```
SYSTEM 1 MVP/solution/
  load.py        # CSV -> canonical schema. Reusable.
  parse.py       # size parser (unit + total), brand alias, slug, flags.
  retrieve.py    # T1 strict + T3 TF-IDF + T5 private-label, unioned.
  score.py       # 14 features, hard veto flags, routing bands.
  rag.py         # Tiny TF-IDF over rules + examples for the LLM prompt.
  judge.py       # gpt-5.4-nano, K=5 batching, structured JSON, backoff cache.
  select.py      # One B per A (multi-key tie-break per audit fix #15).
  validate.py    # Eval-set comparison; reports per stratum.
  main.py        # End-to-end driver. --mode rules/full.
```

See [`DEEPDIVE.md`](DEEPDIVE.md) for what each module does and why.

---

## When to use System 1 vs System 2

| You should use… | If… |
|---|---|
| **System 1** | You want the deliverable, fast. You need a baseline. You're iterating on retrieval/scoring. |
| **System 2** | You care about recall on private-label paraphrase or hand-crafted edge cases. You want richer audit (RAG evidence per decision). You're building toward a production multi-tenant system. |

Both write the same `item_id_A,item_id_B` columns, so the deliverable
shape is interchangeable. The CSVs land in different directories so
they don't clobber each other.
