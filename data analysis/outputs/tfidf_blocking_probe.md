# TF-IDF Blocking Probe

## Vectorizer
- Feature dim (word + char): 616,872
- Fit time: 71.6s
- Transform time: 71.59s

## Top-10 retrieval
- Sample A rows: 2,000
- Retrieval time: 7.01s (285.2 rows/s)
- Estimated full A run: ~13.6 min

## Quality of top-10 (on sample)
- Brand match in top-10 (A has brand): 12.2%
- Category-2 match in top-10: 4.3%
- Size-bucket match in top-10: 21.9%
- At least one cosine >= 0.3 in top-10: 52.7%
- Top-1 cosine p50 / p90: 0.3138 / 0.5977