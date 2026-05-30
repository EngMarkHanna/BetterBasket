# Eval-set Validation

## Overall

| metric | value |
|---|---|
| n | 97 |
| positives | 41 |
| negatives | 56 |
| shipped | 31 |
| tp | 31 |
| fp | 0 |
| fn | 10 |
| tn | 56 |
| precision | 1.0 |
| recall | 0.7561 |
| f1 | 0.8611 |
| overlap_note | None |

## Per stratum

| stratum | n | positives | shipped | tp | fp | fn | prec_on_shipped | recall_on_positives |
|---|---|---|---|---|---|---|---|---|
| A_borderline | 3 | 1 | 1 | 1 | 0 | 0 | 1.0 | 1.0 |
| A_low_score | 5 | 0 | 0 | 0 | 0 | 0 | None | None |
| A_medium | 20 | 1 | 1 | 1 | 0 | 0 | 1.0 | 1.0 |
| A_private_high | 10 | 7 | 4 | 4 | 0 | 3 | 1.0 | 0.5714 |
| A_private_mid | 10 | 0 | 0 | 0 | 0 | 0 | None | None |
| A_strong | 20 | 20 | 20 | 20 | 0 | 0 | 1.0 | 1.0 |
| H_hand | 10 | 3 | 0 | 0 | 0 | 3 | None | 0.0 |
| T_strong_tfidf | 9 | 8 | 4 | 4 | 0 | 4 | 1.0 | 0.5 |
| T_weak_tfidf | 10 | 1 | 1 | 1 | 0 | 0 | 1.0 | 1.0 |