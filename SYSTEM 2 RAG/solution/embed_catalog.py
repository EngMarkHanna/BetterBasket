"""Embed the A and B catalogs into vector arrays.

Reads canonical product tables from SYSTEM 1's parquet caches (so we
don't re-parse the CSVs) and writes:

  SYSTEM 2 RAG/cache/embedding_bank.npz     (hash -> vector, persistent)
  SYSTEM 2 RAG/cache/embeddings_A.npy       (N_a x dim float32)
  SYSTEM 2 RAG/cache/item_ids_A.npy         (N_a object)
  SYSTEM 2 RAG/cache/embeddings_B.npy
  SYSTEM 2 RAG/cache/item_ids_B.npy
  SYSTEM 2 RAG/cache/embed_stats.json

Re-running this script with the same catalog is free: every row hits the
bank cache.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import pandas as pd

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from solution.config import openai_embedding_config
    from solution.embed import (
        EmbeddingBank,
        OpenAIEmbedder,
        embed_dataframe,
        save_store_embeddings,
    )
else:
    from .config import openai_embedding_config
    from .embed import EmbeddingBank, OpenAIEmbedder, embed_dataframe, save_store_embeddings


REPO_ROOT = Path(__file__).resolve().parents[2]
SYSTEM2_ROOT = Path(__file__).resolve().parents[1]
CACHE_DIR = SYSTEM2_ROOT / "cache"
BANK_PATH = CACHE_DIR / "embedding_bank.npz"
STATS_PATH = CACHE_DIR / "embed_stats.json"

# Read SYSTEM 1's canonical parquets - they already have the fields
# embedding_text() needs and load in seconds.
SYSTEM1_CACHE = REPO_ROOT / "SYSTEM 1 MVP" / "cache"


def main():
    cfg = openai_embedding_config()
    print(
        f"[config] model={cfg['model']} dim={cfg['dim']} batch={cfg['batch']}"
    )

    a_parquet = SYSTEM1_CACHE / "store_A_full.parquet"
    b_parquet = SYSTEM1_CACHE / "store_B_full.parquet"
    if not a_parquet.exists() or not b_parquet.exists():
        raise SystemExit(
            f"Canonical parquets missing. Expected:\n  {a_parquet}\n  {b_parquet}\n"
            "Run System 1 first (or wait for the background run to write them)."
        )

    print(f"[load] reading {a_parquet.name}")
    a = pd.read_parquet(a_parquet)
    print(f"  A rows: {len(a):,}")
    print(f"[load] reading {b_parquet.name}")
    b = pd.read_parquet(b_parquet)
    print(f"  B rows: {len(b):,}")

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    bank = EmbeddingBank(BANK_PATH, dim=cfg["dim"])
    embedder = OpenAIEmbedder(
        api_key=cfg["api_key"],
        model=cfg["model"],
        dim=cfg["dim"],
        batch_size=cfg["batch"],
    )

    overall_t0 = time.time()
    print("[A] embedding store A ...")
    a_vecs, a_ids, a_stats = embed_dataframe(a, bank, embedder)
    bank.save()
    save_store_embeddings(a_vecs, a_ids, CACHE_DIR, "A")
    print(
        f"  [A] done in {a_stats['wall_clock_s']:.0f}s  "
        f"new_embedded={a_stats['n_embedded']:,}  cache_hits={a_stats['cache_hits']:,}"
    )

    print("[B] embedding store B ...")
    b_vecs, b_ids, b_stats = embed_dataframe(b, bank, embedder)
    bank.save()
    save_store_embeddings(b_vecs, b_ids, CACHE_DIR, "B")
    print(
        f"  [B] done in {b_stats['wall_clock_s']:.0f}s  "
        f"new_embedded={b_stats['n_embedded']:,}  cache_hits={b_stats['cache_hits']:,}"
    )

    overall = round(time.time() - overall_t0, 1)
    out = {
        "model": cfg["model"],
        "dim": cfg["dim"],
        "batch": cfg["batch"],
        "A": {**a_stats, "rows": len(a)},
        "B": {**b_stats, "rows": len(b)},
        "bank_size": len(bank),
        "wall_clock_total_s": overall,
    }
    STATS_PATH.write_text(json.dumps(out, indent=2))
    print(f"[done] total wall clock: {overall:.0f}s")
    print(f"       bank size: {len(bank):,} unique vectors")
    print(f"       wrote stats to {STATS_PATH.name}")


if __name__ == "__main__":
    main()
