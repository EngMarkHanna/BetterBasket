# Eval-set Validation

## Overall

| metric | value |
|---|---|
| n | 97 |
| positives | 41 |
| negatives | 56 |
| shipped | 38 |
| tp | 32 |
| fp | 6 |
| fn | 9 |
| tn | 50 |
| precision | 0.8421 |
| recall | 0.7805 |
| f1 | 0.8101 |
| overlap_note | None |

## Per stratum

| stratum | n | positives | shipped | tp | fp | fn | prec_on_shipped | recall_on_positives |
|---|---|---|---|---|---|---|---|---|
| A_borderline | 3 | 1 | 1 | 1 | 0 | 0 | 1.0 | 1.0 |
| A_low_score | 5 | 0 | 1 | 0 | 1 | 0 | 0.0 | None |
| A_medium | 20 | 1 | 5 | 1 | 4 | 0 | 0.2 | 1.0 |
| A_private_high | 10 | 7 | 6 | 6 | 0 | 1 | 1.0 | 0.8571 |
| A_private_mid | 10 | 0 | 0 | 0 | 0 | 0 | None | None |
| A_strong | 20 | 20 | 20 | 20 | 0 | 0 | 1.0 | 1.0 |
| H_hand | 10 | 3 | 0 | 0 | 0 | 3 | None | 0.0 |
| T_strong_tfidf | 9 | 8 | 3 | 3 | 0 | 5 | 1.0 | 0.375 |
| T_weak_tfidf | 10 | 1 | 2 | 1 | 1 | 0 | 0.5 | 1.0 |