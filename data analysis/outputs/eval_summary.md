# Model Benchmark on Eval Set

Model: `gpt-5.4-nano`
Eval set: 97 pairs

## Aggregate
- JSON validity: 97.9%
- Match-type accuracy: 87.4%
- is_match accuracy: 86.3%
- Binary precision / recall / F1: 0.906 / 0.744 / 0.817
- Confusion (tp / fp / fn / tn): 29 / 3 / 10 / 53

## Latency
- p50 / p90 / p99: 1.048s / 4.348s / 74.438s

## Tokens & cost
- Avg prompt / completion per pair: 419.7 / 44.8
- Total reasoning tokens: 0
- Eval set cost: $0.003776
- Projected 10k pairs cost: $0.3892

## Per stratum
| stratum | n | label_pos_rate | model_pos_rate | precision | recall | match_type_acc |
|---|---|---|---|---|---|---|
| A_borderline | 3 | 0.333 | 0.333 | 1.0 | 1.0 | 1.0 |
| A_low_score | 5 | 0.0 | 0.0 | None | None | 1.0 |
| A_medium | 20 | 0.05 | 0.05 | 1.0 | 1.0 | 1.0 |
| A_private_high | 10 | 0.7 | 0.5 | 0.8 | 0.571 | 0.6 |
| A_private_mid | 10 | 0.0 | 0.0 | None | None | 1.0 |
| A_strong | 18 | 1.0 | 0.944 | 1.0 | 0.944 | 0.944 |
| H_hand | 10 | 0.3 | 0.2 | 1.0 | 0.667 | 0.9 |
| T_strong_tfidf | 9 | 0.889 | 0.444 | 1.0 | 0.5 | 0.556 |
| T_weak_tfidf | 10 | 0.1 | 0.2 | 0.0 | 0.0 | 0.8 |

## Confidence calibration (predicted positives)
| confidence band | n | empirical precision |
|---|---|---|
| 0.00-0.50 | 1 | 0.0 |
| 0.50-0.70 | 0 | None |
| 0.70-0.85 | 6 | 0.667 |
| 0.85-0.95 | 12 | 1.0 |
| 0.95-1.01 | 13 | 1.0 |

## Match-type confusion
```
{
  "exact_national_brand": {
    "exact_national_brand": 24,
    "no_match": 1,
    "private_label_equivalent": 0
  },
  "no_match": {
    "exact_national_brand": 7,
    "no_match": 54,
    "private_label_equivalent": 3
  },
  "private_label_equivalent": {
    "exact_national_brand": 0,
    "no_match": 1,
    "private_label_equivalent": 5
  }
}
```