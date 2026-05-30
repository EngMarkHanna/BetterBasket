# BetterBasket — Grocery Product Matching

Match products from **Walmart (Store A, 233,195 items)** to
**Wegmans (Store B, 55,516 items)** for downstream price comparison.
Required deliverable: `item_id_A,item_id_B` CSV with at least
4,000 high-quality matches.

This repo contains **two complete matching systems** built on the
same EDA, so you can compare a fast rules+TF-IDF baseline against a
richer semantic + RAG architecture.

---

## The two systems

| | **SYSTEM 1 MVP** (deterministic) | **SYSTEM 2 RAG** (semantic + RAG) |
|---|---|---|
| Retrieval | T1 strict brand+size, T3 TF-IDF, T5 private-label | All of System 1 + T7 semantic (OpenAI embeddings) |
| Judge | gpt-5.4-nano, small rubric, TF-IDF over rules | gpt-5.4-nano, 5-slot RAG context per pair |
| Knowledge | Inline (10 rules) | 460 versioned JSONL entries |
| Cost per run | ~$0.65 | ~$8 |
| Wall clock | ~5h 30min | ~7h 10min |
| Matches shipped | 17,040 | **18,448 (+1,408)** |
| Eval precision | **1.00** | **1.00** |
| Eval recall | 0.66 | **0.76 (+0.10)** |
| Eval F1 | 0.79 | **0.86 (+0.07)** |
| Read me | [`SYSTEM 1 MVP/README.md`](SYSTEM%201%20MVP/README.md) | [`SYSTEM 2 RAG/README.md`](SYSTEM%202%20RAG/README.md) |
| Deep design | [`SYSTEM 1 MVP/DEEPDIVE.md`](SYSTEM%201%20MVP/DEEPDIVE.md) | [`SYSTEM 2 RAG/DEEPDIVE.md`](SYSTEM%202%20RAG/DEEPDIVE.md) |

**System 2 strictly dominates System 1** on the deliverable: same
perfect precision, +10 percentage points of recall, +8% more matches
shipped, zero false positives across all 9 eval-set strata.

System 2 imports System 1's retrieval, scoring, selection, and
validation modules via an `importlib` shim so the comparison is
apples-to-apples by construction.

The full side-by-side write-up with disagreement analysis lives in
[`FINAL_ANALYSIS.md`](FINAL_ANALYSIS.md).

---

## File structure

```
BetterBasket/
├── README.md ............................. (this file)
├── FINAL_ANALYSIS.md ..................... Side-by-side comparison + recommendation
│
├── SYSTEM 1 MVP/
│   ├── README.md ......................... How to run + measured results
│   ├── DEEPDIVE.md ....................... Every component explained, worked examples
│   └── solution/ ......................... Code: load, parse, retrieve, score, rag, judge, select, validate, main
│
├── SYSTEM 2 RAG/
│   ├── README.md ......................... How to run + how it differs from System 1
│   ├── DEEPDIVE.md ....................... Embedding pipeline, RAG context, versioned cache
│   ├── solution/ ......................... Code: config, embed, knowledge/, store/, retrieve/, context/, judge/, pipeline/
│   └── knowledge/ ........................ 460 JSONL entries (rules, aliases, bridges, examples, edge cases)
│
├── data analysis/
│   ├── README.md ......................... EDA findings overview + script index
│   └── 01_*.py … 13_*.py ................. EDA scripts that informed everything
│
├── planning/ ............................. Design rationale + execution trail
│   ├── SYSTEM_1_DESIGN.md ................ Full design + measured results + audit fixes for System 1
│   ├── SYSTEM_2_DESIGN.md ................ Full design + audit response + tool layer proposal for System 2
│   └── EDA_ANALYSIS.md ................... Consolidated EDA findings + how they became architecture
│
├── dataset/ .............................. Raw CSVs (unchanged)
├── references/ ........................... Papers + literature_review.md
│
├── openai_creds.yaml ..................... Azure OpenAI creds (gitignored)
├── .env .................................. OpenAI API key for embeddings (gitignored)
├── requirements.txt
└── .gitignore
```

---

## Setup (once)

### Python environment

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

### Credentials (both files gitignored)

**`openai_creds.yaml`** — Azure OpenAI deployment used for the LLM
judge (`gpt-5.4-nano`):

```yaml
openai:
  endpoint: "https://<your-deployment>.openai.azure.com/openai/v1/"
  api_key: "<your-key>"
  deployment_name: "gpt-5.4-nano"
```

**`.env`** — Public OpenAI API key for embeddings (System 2 only):

```
OPENAI_API_KEY=sk-...
```

---

## Quick start

### Just produce the deliverable (System 1)

```powershell
cd "SYSTEM 1 MVP"
..\.venv\Scripts\python.exe -m solution.main --mode full --llm-workers 5
```

Output: `SYSTEM 1 MVP/outputs/matches.csv`. Wall clock ~5h 30min,
cost ~$0.65.

### Produce the System 2 deliverable too (richer recall)

```powershell
# One-time: embed the catalog (~80 min, ~$0.40)
cd "SYSTEM 2 RAG"
..\.venv\Scripts\python.exe -m solution.embed_catalog

# Quick eval gate on the 97-pair labeled set (~3 min, ~$0.01)
..\.venv\Scripts\python.exe -m solution.pipeline.main --mode eval

# Full production run reusing System 1's candidates (~4h, ~$8)
..\.venv\Scripts\python.exe -m solution.pipeline.main \
    --mode full --reuse-system1-candidates --llm-workers 8
```

Output: `SYSTEM 2 RAG/outputs/matches_full.csv`.

---

## What to read, in order

1. [`data analysis/README.md`](data%20analysis/README.md) — what we
   learned about the data
2. [`SYSTEM 1 MVP/README.md`](SYSTEM%201%20MVP/README.md) +
   [`SYSTEM 1 MVP/DEEPDIVE.md`](SYSTEM%201%20MVP/DEEPDIVE.md) — the
   shippable matcher
3. [`SYSTEM 2 RAG/README.md`](SYSTEM%202%20RAG/README.md) +
   [`SYSTEM 2 RAG/DEEPDIVE.md`](SYSTEM%202%20RAG/DEEPDIVE.md) — the
   upgrade
4. [`FINAL_ANALYSIS.md`](FINAL_ANALYSIS.md) — side-by-side
   comparison and recommendation
5. [`planning/`](planning/) — design rationale + audit trail
   (optional, for the full story)

---

## Repository hygiene

`.gitignore` excludes:

- `.env`, `openai_creds.yaml` — credentials
- `**/cache/`, `**/__pycache__/` — derived artifacts
- `*.npy`, `*.npz`, `*.parquet` — large binary caches
- `.venv/`, IDE files, `*.log`

Push-safe out of the box.
