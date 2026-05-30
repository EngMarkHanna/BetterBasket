# Executed Plan Follow-up EDA

## Parser and brand-alias probe

Store A:
- Rows: 233,199
- Old parsed size buckets: 135,381 (58.1%)
- New unit-size buckets: 125,613 (53.9%)
- Old empty rows recovered by new parser: 63
- Old `count:1` rows removed: 1,547
- Multipack rows with parsed unit size: 22,353
- Rows where old bucket differs from new unit bucket: 3,304
- Rows with shared raw normalized brand: 141,502
- Rows with shared alias-canonical brand probe: 141,866

Store B:
- Rows: 55,516
- Old parsed size buckets: 47,304 (85.2%)
- New unit-size buckets: 48,163 (86.8%)
- Old empty rows recovered by new parser: 935
- Old `count:1` rows removed: 39
- Multipack rows with parsed unit size: 4
- Rows where old bucket differs from new unit bucket: 1,615
- Rows with shared raw normalized brand: 34,277
- Rows with shared alias-canonical brand probe: 35,024

Brand + size block coverage variants:

- `brand_norm` + `size_bucket`: A complete 54,686, A shared-B block 9,789 (4.2% of A)
- `brand_norm` + `new_unit_bucket`: A complete 49,158, A shared-B block 10,236 (4.4% of A)
- `brand_norm` + `new_total_bucket`: A complete 49,158, A shared-B block 9,340 (4.0% of A)
- `brand_norm` + `new_unit_bucket+new_total_bucket`: A complete 49,158, A shared-B block 10,575 (4.5% of A)
- `brand_canonical_probe` + `new_unit_bucket+new_total_bucket`: A complete 49,158, A shared-B block 10,782 (4.6% of A)

## Eval-set routing policy simulation

- `llm_any_positive`: TP 29, FP 3, FN 12, TN 53, P=0.9062, R=0.7073, F1=0.7945
- `llm_conf_gte_085`: TP 25, FP 0, FN 16, TN 56, P=1.0, R=0.6098, F1=0.7576
- `llm_conf_gte_070`: TP 29, FP 2, FN 12, TN 54, P=0.9355, R=0.7073, F1=0.8056
- `auto_strong_plus_llm085`: TP 28, FP 0, FN 13, TN 56, P=1.0, R=0.6829, F1=0.8116
- `auto_strong_tfidf_plus_llm085`: TP 32, FP 1, FN 9, TN 55, P=0.9697, R=0.7805, F1=0.8649

Stratum recap:

- `A_borderline`: n=3, label positive rate=0.3333, model positive rate=0.3333, high-conf model positives=0
- `A_low_score`: n=5, label positive rate=0.0, model positive rate=0.0, high-conf model positives=0
- `A_medium`: n=20, label positive rate=0.05, model positive rate=0.05, high-conf model positives=1
- `A_private_high`: n=10, label positive rate=0.7, model positive rate=0.5, high-conf model positives=2
- `A_private_mid`: n=10, label positive rate=0.0, model positive rate=0.0, high-conf model positives=0
- `A_strong`: n=20, label positive rate=1.0, model positive rate=0.85, high-conf model positives=17
- `H_hand`: n=10, label positive rate=0.3, model positive rate=0.2, high-conf model positives=1
- `T_strong_tfidf`: n=9, label positive rate=0.8889, model positive rate=0.4444, high-conf model positives=4
- `T_weak_tfidf`: n=10, label positive rate=0.1, model positive rate=0.2, high-conf model positives=0

## Planning implications

- Store the new parser as unit-size plus total-size features. Do not collapse multipacks into a single bucket only; the eval labels accept same per-unit SKU even when A is a multipack and B is a single.
- Keep LLM auto-accept at confidence >= 0.85. On this eval set it has perfect precision but lower recall, so deterministic auto-accept rules are still needed for obvious strong strata.
- Do not auto-accept weak TF-IDF or private-label mid-score candidates. They need stronger deterministic filters or LLM review.
- Batch API should be removed from the near-term plan for this deployment. Use K-candidate prompts, `reasoning_effort='minimal'`, concurrency, and backoff.