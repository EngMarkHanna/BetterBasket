# Text Signals Probe

## Description coverage and content

Store A:
- Description coverage: 18,908 / 233,199 (8.1%)
- Rows with UPC-like token in description: 270
- Rows with size-like token in description: 5,123
- Description length p50/p90/p99: 621 / 1407 / 2753 chars

Store B:
- Description coverage: 51,938 / 55,516 (93.6%)
- Rows with UPC-like token in description: 47
- Rows with size-like token in description: 15,848
- Description length p50/p90/p99: 143 / 873 / 1667 chars

## UPC token cross-store overlap
- A rows with any 12-14 digit token across name+desc+item_info+tags+url: 1,122
- B rows with any 12-14 digit token: 366
- Unique normalized UPC codes in A: 1,150
- Unique normalized UPC codes shared with B: 0
- Cross-store UPC-linked pairs: 0
- Unique A items linked via UPC: 0
- Unique B items linked via UPC: 0

## URL slug informativeness (Store A sample)
- Sample rows: 5,000
- Mean extra tokens in slug beyond `name`: 0.937
- p50 / p90 extra: 0.0 / 3.0
- Rows where slug adds at least one token: 1,580

## item_info fields beyond categories

Store A:
- `storage_type` coverage: 6.0%
- `packaging_description` coverage: 7.9%
- `ingredients` coverage: 3.4%

Store B:
- `storage_type` coverage: 0.0%
- `packaging_description` coverage: 0.0%
- `ingredients` coverage: 52.7%