# Final Analysis — System 1 vs System 2

Side-by-side comparison of the two matchers built for the BetterBasket
product-matching assessment. System 1 (deterministic) and System 2
(semantic + RAG) are evaluated on the same 97-pair labeled eval set
and the same full A × B catalog.

For the full design rationale of each system, see
[`planning/SYSTEM_1_DESIGN.md`](planning/SYSTEM_1_DESIGN.md) and
[`planning/SYSTEM_2_DESIGN.md`](planning/SYSTEM_2_DESIGN.md).

---

## Headline numbers

| Metric | System 1 | System 2 (eval-mode) | System 2 (full, pending) | Δ (S2 eval − S1) |
|---|---|---|---|---|
| Total matches shipped | **17,040** | 38 / 97 | _(filled when run completes)_ | _(filled)_ |
| Unique A rows in output | 17,040 | 38 | — | — |
| Eval-set precision | **1.00** | 0.84 | _(filled)_ | **−0.16** |
| Eval-set recall | 0.66 | **0.78** | _(filled)_ | **+0.12** |
| Eval-set F1 | 0.79 | **0.81** | _(filled)_ | **+0.02** |
| LLM cost | $0.65 | $0.01 | ~$8 | — |
| Wall clock | 5h 30min | 3 min | ~4h | — |

The eval-mode numbers above are the gate test on the 97-pair labeled
set. The full-mode column gets filled when the production run lands.

---

## Per-stratum eval recall — where each system wins

| Stratum | Positives | S1 shipped | S2 shipped | S1 recall | S2 recall | Verdict |
|---|---|---|---|---|---|---|
| `A_strong` | 20 | 20 | 20 | **1.00** | **1.00** | Tie — both perfect on rules-strong tier |
| `A_medium` | 1 | 1 | 5 | 1.00 | 1.00 (4 FP) | **S2 regresses on precision** |
| `A_borderline` | 1 | 1 | 1 | 1.00 | 1.00 | Tie |
| `A_private_high` | 7 | 2 | **6** | 0.29 | **0.86** | **S2 wins big** (RAG context fix) |
| `A_private_mid` | 0 | 0 | 0 | — | — | Both correctly empty |
| `A_low_score` | 0 | 0 | 1 | — | 0 (1 FP) | S2 regression |
| `H_hand` | 3 | 0 | 0 | 0 | 0 | Tie (both miss — long-tail edge cases) |
| `T_strong_tfidf` | 8 | 3 | 3 | 0.38 | 0.38 | Tie — neither catches the multipack/word-order misses |
| `T_weak_tfidf` | 1 | 0 | 2 | 0 | 1.00 (1 FP) | S2 catches the positive, also one FP |

**Headline read**: System 2 fixes the `A_private_high` recall gap
(the whole point of RAG context — private-label paraphrase) and
trades 6 false positives concentrated in `A_medium` and
`T_weak_tfidf`.

---

## Cost / time / quality envelope

| | System 1 | System 2 |
|---|---|---|
| Per-pair LLM cost | $0.000039 | $0.000095 |
| Per-pair LLM latency | ~1s | ~1.5s |
| One-time embedding cost | n/a | $0.40 |
| Full LLM volume | 56,495 batches | ~77,000 batches |
| Full run cost | $0.65 | ~$8 |
| Full run wall clock | 5h 30min | ~4h (with 8 workers + reuse) |
| Storage | parquet caches ~165 MB | + 1.4 GB A vectors + 340 MB B vectors |
| Code | ~1,500 lines | ~2,300 lines |

---

## Disagreement analysis _(pending full run)_

When the full System 2 run completes, this section will contain:

- **Top 20 System 2-only accepts** — matches S2 ships that S1 didn't.
  Expected dominant pattern: private-label paraphrase + multipack
  recovery.
- **Top 20 System 1-only accepts** — matches S1 ships that S2 either
  drops or sends to a different B. Watch for cases where S2's broader
  T7 retrieval changed which B won the per-A selection.
- **Top 20 System 2 false positives on the eval set** — the 6 FPs
  measured in the eval gate, broken down by stratum and feature
  profile. Tells us whether the RAG-context-induced precision drop is
  a calibration problem (LLM confidence threshold) or a routing
  problem (which candidates reach the LLM).

---

## Recommendation _(provisional — will firm up after full run)_

Based on what we have so far (eval-mode results):

1. **System 1 is the shippable deliverable.** 17,040 matches at 100%
   eval-set precision; 4.26× the required volume.

2. **System 2 is a measurable improvement on recall but a measurable
   regression on precision.** The headline F1 lift (+0.02) is real but
   modest. The per-stratum lift on `A_private_high` is large and
   exactly the target the architecture was designed for.

3. **The precision drop is fixable** with one of:
   - Higher LLM confidence threshold for stratum-like signals
     (e.g., 0.90 for cases where deterministic features look weak).
   - Per-stratum routing — don't send `A_medium` candidates to the
     LLM at all if the deterministic features predict ≤ 5% positive
     rate.
   - Build the priority-4 agentic tools (Phase D in
     `planning/SYSTEM_2_DESIGN.md` § 7), especially
     `classify_variant_conflict` which targets exactly this FP class.

4. **System 2's hardest unfixed gap is `H_hand`** (0/3 on hand-crafted
   edge cases). This is the long tail of "weird real cases" that
   neither rules nor RAG context covers. Phase D tools (especially
   `compute_size_compatibility`) are designed for this.

5. **Eval set expansion is the highest-leverage non-code task.**
   97 pairs is noisy. The +0.02 F1 lift could be noise at this sample
   size. Expanding to 300+ labels with fresh and private-label
   oversampling would let us calibrate System 2 properly and decide
   Phase D with confidence.

---

## What we built and audited

The full audit trail and design iteration lives in `planning/`. The
short version:

- A first-pass audit (in `planning/SYSTEM_2_DESIGN.md` § 6) found 16
  concrete bugs across both systems and 7 design issues. 14 of the
  16 were fixed; 3 parser-level fixes were deferred because they
  require Parquet rebuild + System 1 re-run.
- A second-pass review on the proposed agentic tool layer corrected
  the priority list (dropped 4 tools, added 2, moved 3 from agentic
  to parser/scorer).
- Both system READMEs and DEEPDIVE docs are written and ready for
  a human reader to consume cold.

---

## How to reproduce

```powershell
.\.venv\Scripts\Activate.ps1

# System 1 (the safety-net deliverable)
cd "SYSTEM 1 MVP"
..\.venv\Scripts\python.exe -m solution.main --mode full --llm-workers 5
# Output: SYSTEM 1 MVP/outputs/matches.csv

# System 2 (the upgrade comparison)
cd "..\SYSTEM 2 RAG"
..\.venv\Scripts\python.exe -m solution.embed_catalog       # one-time, ~80 min
..\.venv\Scripts\python.exe -m solution.pipeline.main --mode eval
..\.venv\Scripts\python.exe -m solution.pipeline.main \
    --mode full --reuse-system1-candidates --llm-workers 8
# Output: SYSTEM 2 RAG/outputs/matches_full.csv
```

Both runs use a JSONL judgment cache, so re-running with no changes
is free (cache hit). Changing the rubric, knowledge base, or prompt
format bumps the cache version and re-judges affected pairs.
