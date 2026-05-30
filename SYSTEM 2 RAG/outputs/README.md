# SYSTEM 2 — outputs/

This folder holds everything System 2's pipeline writes. The actual
**deliverable** is `matches.csv` (a copy lives in the parent folder
`../matches.csv` so it's prominently next to the README rather than
buried here).

The pipeline runs in two modes (`--mode eval` and `--mode full`),
each producing its own set of files distinguished by a `_eval` /
`_full` suffix.

| File | Size class | What it is | Committable? |
|---|---|---|---|
| **`matches_full.csv`** | small | **The production deliverable.** Two columns: `item_id_A,item_id_B`. Also copied to `../matches.csv`. | Yes |
| `matches_eval.csv` | tiny | The 97-pair eval-gate deliverable (`--mode eval`). Used to decide whether to commit to the full run. | Yes |
| `matches_with_features_full.csv` | small | Same rows + features + RAG evidence summary per row. Audit-friendly. | Yes |
| `matches_with_features_eval.csv` | tiny | Same shape for eval mode. | Yes |
| `match_candidates_scored_full.csv` | medium | Every candidate considered, with route decision. | Optional |
| `match_candidates_scored_eval.csv` | tiny | Same for eval mode. | Optional |
| `validation_report_full.md` | small | Eval-set precision / recall / F1 per stratum vs the production matches. | Yes |
| `validation_report_eval.md` | small | Same for eval mode. | Yes |
| `llm_judgments_full.jsonl` | large (~50 MB) | Every LLM call cached. Re-runs are free because of this cache. | Gitignored |
| `llm_judgments_eval.jsonl` | small (~70 KB) | Same for eval mode. | Gitignored |
| `embed_run.log` | small | Stdout of the most recent `embed_catalog` run. | Gitignored |
| `full_run.log` | medium | Stdout of the most recent `--mode full` run. | Gitignored |

## Distinguishing `_eval` vs `_full`

| | Eval mode | Full mode |
|---|---|---|
| A subset | 87 / 97 (intersected with eval-set A IDs) | All 233k A rows |
| B subset | 87 / 97 | All 55k B rows |
| Wall clock | ~3 min | ~4h |
| Cost | ~$0.01 | ~$8 |
| Use case | Go/no-go gate before committing to full run | The actual production deliverable |

If you only see `_eval` files, the full run hasn't completed yet.

## What goes back into the pipeline

The deliverable (`matches.csv` / `matches_full.csv`) is the file the
comparison in `../../FINAL_ANALYSIS.md` consumes. Nothing else here
feeds back into the pipeline; the LLM judgment cache (`*.jsonl`) and
the vector cache in `../cache/` together make re-runs cheap.

## Regenerating any file

```powershell
cd "SYSTEM 2 RAG"

# Eval gate (cheap)
..\.venv\Scripts\python.exe -m solution.pipeline.main --mode eval

# Full production run
..\.venv\Scripts\python.exe -m solution.pipeline.main --mode full \
    --reuse-system1-candidates --llm-workers 8
```
