# Eval-set Validation

## Overall

| metric | value |
|---|---|
| n | 97 |
| positives | 41 |
| negatives | 56 |
| shipped | 27 |
| tp | 27 |
| fp | 0 |
| fn | 14 |
| tn | 56 |
| precision | 1.0 |
| recall | 0.6585 |
| f1 | 0.7941 |

## Per stratum

| stratum | n | positives | shipped | tp | fp | fn | prec_on_shipped | recall_on_positives |
|---|---|---|---|---|---|---|---|---|
| A_borderline | 3 | 1 | 1 | 1 | 0 | 0 | 1.0 | 1.0 |
| A_low_score | 5 | 0 | 0 | 0 | 0 | 0 | 0.0 | 0.0 |
| A_medium | 20 | 1 | 1 | 1 | 0 | 0 | 1.0 | 1.0 |
| A_private_high | 10 | 7 | 2 | 2 | 0 | 5 | 1.0 | 0.2857 |
| A_private_mid | 10 | 0 | 0 | 0 | 0 | 0 | 0.0 | 0.0 |
| A_strong | 20 | 20 | 20 | 20 | 0 | 0 | 1.0 | 1.0 |
| H_hand | 10 | 3 | 0 | 0 | 0 | 3 | 0.0 | 0.0 |
| T_strong_tfidf | 9 | 8 | 3 | 3 | 0 | 5 | 1.0 | 0.375 |
| T_weak_tfidf | 10 | 1 | 0 | 0 | 0 | 1 | 0.0 | 0.0 |