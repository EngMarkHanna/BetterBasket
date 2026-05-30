# How to Run

A step-by-step tutorial for someone who just cloned this repo and
wants to produce the deliverable `matches.csv`. Should take ~10
minutes to set up; the actual matchers take hours to run end-to-end.

If you get stuck, see the **Troubleshooting** section at the bottom.

---

## Step 1 — Clone the repo

```powershell
git clone <repo-url>
cd BetterBasket
```

You should see this structure (some folders empty until you complete
later steps):

```
BetterBasket/
├── README.md
├── HOW_TO_RUN.md          # this file
├── FINAL_ANALYSIS.md
├── requirements.txt
├── .env.example
├── openai_creds.yaml.example
├── dataset/               # empty until step 3
├── data analysis/
├── SYSTEM 1 MVP/
├── SYSTEM 2 RAG/
└── planning/
```

---

## Step 2 — Set up the Python environment

```powershell
# Create a virtual environment
python -m venv .venv

# Activate it (Windows PowerShell)
.\.venv\Scripts\Activate.ps1

# On macOS / Linux instead:
#   source .venv/bin/activate

# Upgrade pip and install dependencies
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Verify:

```powershell
.\.venv\Scripts\python.exe -c "import pandas, sklearn, rapidfuzz, openai; print('OK')"
```

You should see `OK`.

---

## Step 3 — Put the datasets in place

The two source CSVs are **not** in the repo (they're large and
distributed separately). Drop them into `dataset/`:

```
dataset/
├── grocery_store_a_items_final.csv   # Store A (Walmart), 233k rows, ~159 MB
└── grocery_store_b_items_final.csv   # Store B (Wegmans), 55k rows, ~65 MB
```

If you only have System 1 in mind, this is the only data step.

---

## Step 4 — Set up credentials

Two providers, two files. Both are gitignored, so they never get
committed.

### 4a — Azure OpenAI credentials (required for both systems)

The LLM judge calls Azure OpenAI. Copy the example file and fill it in:

```powershell
copy openai_creds.yaml.example openai_creds.yaml
```

Then open `openai_creds.yaml` in your editor and fill in:

```yaml
openai:
  endpoint: "https://<your-deployment>.openai.azure.com/openai/v1/"
  api_key: "your-azure-api-key"
  deployment_name: "gpt-5.4-nano"
```

### 4b — OpenAI public API key (only needed for System 2)

System 2 uses public OpenAI for embeddings (`text-embedding-3-small`).
If you only plan to run System 1, **skip this step**.

```powershell
copy .env.example .env
```

Then open `.env` and fill in:

```
OPENAI_API_KEY=sk-your-openai-api-key-here
```

Get a key at <https://platform.openai.com/api-keys>.

---

## Step 5 — Run System 1 (the safety-net deliverable)

This produces `matches.csv` with ~17,000 matches in ~5h 30min. Cost
~$0.65.

```powershell
cd "SYSTEM 1 MVP"
..\.venv\Scripts\python.exe -m solution.main --mode full --llm-workers 5
cd ..
```

Output lands in `SYSTEM 1 MVP/outputs/`:

| File | What it is |
|---|---|
| `matches.csv` | **The deliverable** (`item_id_A,item_id_B`) |
| `matches_with_features.csv` | Same rows + all scoring features (audit-friendly) |
| `match_candidates_scored.csv` | Every candidate considered, with route decision |
| `llm_judgments.jsonl` | Every LLM call cached (replayable) |
| `validation_report.md` | Eval-set precision / recall / F1 per stratum |

### If you want to test the pipeline quickly first

Run rules-only mode (no LLM, no Azure required, ~55 min):

```powershell
..\.venv\Scripts\python.exe -m solution.main --mode rules
```

Or cap the A catalog for fast iteration (~3 min for 5k A rows):

```powershell
..\.venv\Scripts\python.exe -m solution.main --mode full --a-limit 5000 --llm-workers 5
```

---

## Step 6 — Run System 2 (optional upgrade)

Only do this if you want the comparison story. It produces a second
`matches_full.csv` with similar volume but different recall/precision
characteristics. ~4h, ~$8.

```powershell
cd "SYSTEM 2 RAG"

# One-time: embed the entire catalog (~80 min, ~$0.40)
..\.venv\Scripts\python.exe -m solution.embed_catalog

# Quick sanity check on the 97-pair eval set (~3 min, ~$0.01)
..\.venv\Scripts\python.exe -m solution.pipeline.main --mode eval

# Full production run, reusing System 1's candidates to save time
..\.venv\Scripts\python.exe -m solution.pipeline.main --mode full \
    --reuse-system1-candidates --llm-workers 8

cd ..
```

Output lands in `SYSTEM 2 RAG/outputs/`:

| File | What it is |
|---|---|
| `matches_full.csv` | The System 2 deliverable |
| `matches_with_features_full.csv` | Same rows + features + RAG evidence summary per row |
| `match_candidates_scored_full.csv` | Every candidate considered |
| `llm_judgments_full.jsonl` | Every LLM call cached |
| `validation_report_full.md` | Eval-set metrics |

You'll also see persistent caches in `SYSTEM 2 RAG/cache/`:

- `embedding_bank.npz` — the hash → vector cache (re-runs are free)
- `embeddings_A.npy`, `item_ids_A.npy` — A catalog vectors
- `embeddings_B.npy`, `item_ids_B.npy` — B catalog vectors

These are gitignored.

---

## Step 7 — View the results

The deliverable CSVs are at:
- `SYSTEM 1 MVP/outputs/matches.csv`
- `SYSTEM 2 RAG/outputs/matches_full.csv` (if you ran System 2)

For the side-by-side comparison, read `FINAL_ANALYSIS.md` — it
includes precision / recall / F1 per stratum, disagreement analysis,
and a recommendation.

---

## Order summary (the lazy version)

If you just want a numbered checklist:

1. `python -m venv .venv && .\.venv\Scripts\Activate.ps1`
2. `pip install -r requirements.txt`
3. Drop the two CSVs into `dataset/`
4. Copy `openai_creds.yaml.example` → `openai_creds.yaml`, fill in
5. (System 2 only) Copy `.env.example` → `.env`, fill in
6. `cd "SYSTEM 1 MVP" && ..\.venv\Scripts\python.exe -m solution.main --mode full --llm-workers 5`
7. Look at `SYSTEM 1 MVP/outputs/matches.csv` — done

---

## Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `OPENAI_API_KEY is empty` | `.env` missing or unfilled | Copy `.env.example` → `.env`, fill in |
| `Azure creds file missing: openai_creds.yaml` | `openai_creds.yaml` not created | Copy `openai_creds.yaml.example` → `openai_creds.yaml`, fill in |
| `FileNotFoundError: ...grocery_store_*.csv` | Datasets not in `dataset/` | See Step 3 |
| Rate limit (429) errors in logs | Azure deployment quota hit | The pipeline retries automatically with exponential backoff. If it persists for >10 min, reduce `--llm-workers` from 5 to 3 |
| `OSError: No space left on device` during embedding | <2 GB free | Free up disk; the embedding bank can grow to ~1 GB and atomic save briefly doubles that |
| System 2 says canonical parquets missing | System 1 wasn't run first | Run System 1 (Step 5) first; System 2 reads the canonical Parquet cache from `SYSTEM 1 MVP/cache/` |
| `import solution.X` errors | venv not activated | Activate with `.\.venv\Scripts\Activate.ps1` |

### Re-runs are free

Both systems cache their LLM judgments to JSONL files. Re-running with
the same inputs produces 0 new API calls. If you change the rubric,
knowledge base, or prompt format, the cache version bumps and affected
pairs re-judge.

### Where each piece is documented

- **What was built** → `README.md`
- **How systems work** → `SYSTEM 1 MVP/DEEPDIVE.md`, `SYSTEM 2 RAG/DEEPDIVE.md`
- **EDA findings** → `data analysis/README.md`, `planning/EDA_ANALYSIS.md`
- **Design rationale** → `planning/SYSTEM_1_DESIGN.md`, `planning/SYSTEM_2_DESIGN.md`
- **Side-by-side comparison + recommendation** → `FINAL_ANALYSIS.md`
