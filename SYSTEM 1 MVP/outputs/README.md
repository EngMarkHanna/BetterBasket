# SYSTEM 1 — outputs/

This folder holds everything System 1's pipeline writes. The actual
**deliverable** is `matches.csv` (a copy lives in the parent folder
`../matches.csv` so it's prominently next to the README rather than
buried here).

| File | Size class | What it is | Committable? |
|---|---|---|---|
| **`matches.csv`** | small | **The deliverable.** Two columns: `item_id_A,item_id_B`. 17,040 rows. | Yes (also copied to `../matches.csv`) |
| `matches_with_features.csv` | small | Same rows + every deterministic feature + LLM confidence + accept source. Audit-friendly. | Yes |
| `match_candidates_scored.csv` | medium | Every candidate the system considered, with route decision (`auto_accept` / `route_to_llm` / `drop`). | Optional |
| `rejected_borderline.csv` | medium | Candidates that were close to acceptance but dropped (for audit + tuning). | Optional |
| `match_audit_sample.csv` | small | 50 random shipped matches for manual review. | Yes |
| `validation_report.md` | small | Eval-set precision / recall / F1 per stratum. | Yes |
| `llm_judgments.jsonl` | large (~30 MB) | Every LLM call cached: input, output, cache key. Re-runs are free because of this cache. | Gitignored (regenerable + large) |
| `run_full.log` | medium | Full stdout of the last `--mode full` run. | Gitignored |
| `run_rules.log` | small | Full stdout of the last `--mode rules` run, if any. | Gitignored |

## What goes back into the pipeline

The deliverable (`matches.csv`) is the only file other systems consume:

- `SYSTEM 2 RAG/solution/pipeline/main.py` uses
  `SYSTEM 1 MVP/cache/candidates_afull.parquet` (not the outputs here)
  when run with `--reuse-system1-candidates`.
- `data analysis/outputs/eval_*.csv` provide the ground truth that
  `validation_report.md` is computed against.

Nothing here is needed as input to subsequent runs of System 1
itself — the candidate Parquet cache (in `../cache/`) and the LLM
JSONL cache (here) together make re-runs cheap.

## Regenerating any file

```powershell
cd "SYSTEM 1 MVP"
..\.venv\Scripts\python.exe -m solution.main --mode full --llm-workers 5
```

All files in this folder get re-written.
