# Dataset

Place the two source CSVs here:

```
dataset/
├── grocery_store_a_items_final.csv   # Store A (Walmart), 233,195 rows, ~159 MB
└── grocery_store_b_items_final.csv   # Store B (Wegmans), 55,516 rows, ~65 MB
```

These files are **gitignored** because they're large (>100 MB) and
typically distributed separately. Obtain them from the assessment
package or the original data source.

The matchers load these files via:

- `SYSTEM 1 MVP/solution/load.py` (read on first run, cached as Parquet)
- `SYSTEM 2 RAG/solution/embed_catalog.py` (re-uses the Parquet cache
  produced by System 1)

Once loaded, canonical Parquet caches are written to
`SYSTEM 1 MVP/cache/store_A_full.parquet` and `store_B_full.parquet`.
Those caches are also gitignored.
