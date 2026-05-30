# Phase 0 Feasibility Simulation

Projects how many A->B matches the planned pipeline can produce, 
using real candidate counts and eval-set-derived precision baselines.

Brand alias map applied: 20 entries; aliased 325 A rows and 710 B rows.

## T1: shared brand_canonical + same size_bucket, scored by RapidFuzz WRatio
- Score >= 95 (auto-accept, A_strong-like, ~100% precision): **751**
- Score 85-95 (route to LLM, A_medium-like band): 7,288
- Score 70-85 (route to LLM, A_borderline-like): 866
- Score <70 (drop): 1,085
- Total scored: 9,990

## T3: TF-IDF top-1 retrieval (projected from 5,000-A sample)
- Sample size: 5,000; scale factor to full A: 46.64x
- Cosine >= 0.6 (any): 28,683
  - of which brand-aligned (auto-accept tier): **9,374**
  - of which size-aligned: 11,286
  - of which brand AND size aligned: 2,845
- Cosine 0.4-0.6 (route to LLM): 55,501

## Routing-policy projection
Policy: auto-accept T1 (score>=95) + T3 (cosine>=0.6, brand-aligned); 
route everything else to LLM and accept iff model_confidence >= 0.85.

Per-band LLM acceptance rates derived from `eval_results.csv`:
- T1 score 85-95 (~A_medium): 5%
- T1 score 70-85 (~A_borderline): 2%
- T3 brand-unaligned cosine>=0.6: 15%
- T3 cosine 0.4-0.6: 5%

| Source | Projected accept |
|---|---|
| Auto-accept T1 (score>=95) | 751 |
| Auto-accept T3 (cosine>=0.6 + brand) | 9,374 |
| LLM-accepted (all routed bands) | 6,052 |
| **Total projected matches** | **16,177** |
| Target | 4,000 |

### Exit gate: **PASS** (404% of target)

LLM call volume estimate: ~82,964 pairs routed; 
at ~1s/call sequential or ~5 pair/s with K=5 batching + 5 concurrent workers, 
this is ~276-1382 minutes wall-clock.