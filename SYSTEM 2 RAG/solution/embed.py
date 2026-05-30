"""OpenAI embedding pipeline with aggressive caching.

Design goals:
1. NEVER re-embed a row whose content hasn't changed. Cache is keyed by
   sha256(model | dim | text_version | embedding_text); any row whose
   key is in the bank reuses its vector. The model/dim/version prefix
   means model swaps and text-format changes never silently reuse stale
   vectors. First run = full ~$0.40; subsequent runs = $0.
2. Survive crashes. We checkpoint the bank to disk every
   `checkpoint_every` batches so a Ctrl-C / 429 storm / power blip
   doesn't waste work.
3. Per-store deliverables: parallel `embeddings_<store>.npy`
   (N x dim, float32) and `item_ids_<store>.npy` (N strings) in catalog
   order. Downstream code just loads these two arrays.

EMBEDDING TEXT VERSION = v2.
  v2 over v1: adds parsed unit size + pack count, url_slug, ingredients,
  and explicit organic/food/fresh flags. v1 had only name + brand +
  raw size + category + description, which made the resulting vectors
  largely equivalent to the System 1 TF-IDF retrieval space (T7
  rediscovered T3). v2 emphasizes fields TF-IDF underuses.
"""
from __future__ import annotations

import hashlib
import time
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from openai import OpenAI, RateLimitError, APIError, APIConnectionError


EMBEDDING_TEXT_VERSION = "v2"


def _safe_str(value) -> str:
    """Return a clean string from any cell value (handles NaN, None, etc)."""
    if value is None:
        return ""
    # NaN is a float; pd.isna also catches pandas NA sentinel.
    try:
        if isinstance(value, float) and value != value:
            return ""
    except Exception:
        pass
    s = str(value).strip()
    if s.lower() in {"nan", "none", "null"}:
        return ""
    return s


def _bool_flag(value) -> str:
    """Render an inferred boolean flag in a deterministic, embed-friendly form."""
    return "yes" if bool(value) else "no"


def _format_numeric(value, ndigits: int = 2) -> str:
    if value is None:
        return ""
    try:
        if isinstance(value, float) and value != value:
            return ""
    except Exception:
        pass
    try:
        f = float(value)
    except (TypeError, ValueError):
        return _safe_str(value)
    if f == int(f):
        return str(int(f))
    return f"{f:.{ndigits}f}"


def embedding_text(row: dict) -> str:
    """Canonical embedding-text format for a product row (version v2).

    Stable across runs. Changing this format requires bumping
    EMBEDDING_TEXT_VERSION above, which invalidates the cache cleanly.
    """
    parts: list[str] = []

    name = _safe_str(row.get("name"))
    if name:
        parts.append(name)

    brand = _safe_str(row.get("brand_canonical"))
    if brand:
        parts.append(f"brand: {brand}")

    size_text = _safe_str(row.get("size_text"))
    if size_text:
        parts.append(f"size: {size_text}")

    # Parsed per-unit size makes "12 fl oz" and "355 ml" land near each
    # other in vector space - TF-IDF treats them as disjoint tokens.
    unit_value = _format_numeric(row.get("unit_value"))
    unit_unit = _safe_str(row.get("unit_unit"))
    if unit_value and unit_unit:
        parts.append(f"unit: {unit_value} {unit_unit}")

    # Pack count when > 1 (multipack signal).
    pack = row.get("pack_count")
    try:
        if pack is not None and not (isinstance(pack, float) and pack != pack):
            pack_int = int(pack)
            if pack_int > 1:
                parts.append(f"pack: {pack_int}")
    except (TypeError, ValueError):
        pass

    cat = _safe_str(row.get("category_path_norm"))
    if cat:
        parts.append(f"category: {cat}")

    slug = _safe_str(row.get("url_slug_norm"))
    if slug:
        parts.append(f"url_slug: {slug}")

    # Flags packed together so the embedder sees them in one slot.
    parts.append(
        "flags: "
        f"private_label={_bool_flag(row.get('is_private_label_inferred'))} "
        f"organic={_bool_flag(row.get('is_organic_inferred'))} "
        f"food={_bool_flag(row.get('is_food_like'))} "
        f"fresh={_bool_flag(row.get('is_fresh_like'))}"
    )

    # Ingredients (B-side mostly; ~53% coverage on B). Critical for
    # private-label-to-national-brand semantic similarity.
    ingredients = _safe_str(row.get("ingredients_norm"))[:300]
    if ingredients:
        parts.append(f"ingredients: {ingredients}")

    desc = _safe_str(row.get("description_norm"))[:200]
    if desc:
        parts.append(f"description: {desc}")

    return " | ".join(parts) if parts else "(empty)"


def text_hash(text: str, model: str, dim: int, text_version: str = EMBEDDING_TEXT_VERSION) -> str:
    """Cache key: sha256(model | dim | text_version | text).

    Prevents silent stale-vector reuse across model/dim/format changes.
    """
    key = f"{model}|{dim}|{text_version}|{text}"
    return hashlib.sha256(key.encode("utf-8")).hexdigest()


class EmbeddingBank:
    """Persistent {sha256(text) -> vector} cache backed by a single NPZ file.

    We store hashes as a (N,) array of fixed-width bytes and vectors as
    (N, dim) float32. On load we build a Python dict for O(1) lookup.
    """

    def __init__(self, path: Path, dim: int):
        self.path = path
        self.dim = dim
        self._hash_to_idx: dict[str, int] = {}
        self._hashes: list[str] = []
        self._vectors: list[np.ndarray] = []  # list of (dim,) arrays
        self._dirty = False
        self._load()

    def _load(self) -> None:
        if not self.path.exists():
            return
        try:
            blob = np.load(self.path, allow_pickle=False)
            hashes = blob["hashes"].astype(str)
            vectors = blob["vectors"].astype(np.float32)
            if vectors.ndim != 2 or vectors.shape[1] != self.dim:
                raise ValueError(
                    f"cache dim mismatch: got {vectors.shape}, expected (*, {self.dim})"
                )
            for i, h in enumerate(hashes):
                self._hash_to_idx[h] = i
                self._hashes.append(h)
                self._vectors.append(vectors[i])
            print(f"  [bank] loaded {len(self._hashes):,} cached vectors from {self.path.name}")
        except Exception as e:
            print(f"  [bank] WARNING: failed to load {self.path}: {e!r} - starting empty")
            self._hash_to_idx.clear()
            self._hashes.clear()
            self._vectors.clear()

    def has(self, h: str) -> bool:
        return h in self._hash_to_idx

    def get(self, h: str) -> np.ndarray | None:
        idx = self._hash_to_idx.get(h)
        return self._vectors[idx] if idx is not None else None

    def put(self, h: str, vec: np.ndarray) -> None:
        if h in self._hash_to_idx:
            return
        self._hash_to_idx[h] = len(self._hashes)
        self._hashes.append(h)
        self._vectors.append(np.asarray(vec, dtype=np.float32))
        self._dirty = True

    def save(self) -> None:
        if not self._dirty and self.path.exists():
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # Atomic-ish write: save to temp, rename. Use fixed-width strings
        # for hashes (sha256 hex = 64 chars) so np.load works without pickle.
        tmp = self.path.with_suffix(".tmp.npz")
        hashes_arr = np.array(self._hashes, dtype="U64")
        vectors_arr = (
            np.stack(self._vectors).astype(np.float32)
            if self._vectors
            else np.zeros((0, self.dim), dtype=np.float32)
        )
        np.savez_compressed(tmp, hashes=hashes_arr, vectors=vectors_arr)
        tmp.replace(self.path)
        self._dirty = False

    def __len__(self) -> int:
        return len(self._hashes)


class OpenAIEmbedder:
    def __init__(self, api_key: str, model: str, dim: int, batch_size: int = 100):
        self.client = OpenAI(api_key=api_key, max_retries=4, timeout=120.0)
        self.model = model
        self.dim = dim
        self.batch_size = batch_size

    def embed_batch(self, texts: list[str]) -> np.ndarray:
        """Call OpenAI embeddings with bounded backoff on transient errors.
        Retries on 429, 500/502/503/504, timeouts, and connection errors.
        Returns float32 (N, dim).
        """
        delays = [0.0, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0]
        last_err: Exception | None = None
        for delay in delays:
            if delay:
                time.sleep(delay)
            try:
                resp = self.client.embeddings.create(
                    model=self.model, input=texts, encoding_format="float"
                )
                vecs = np.array(
                    [d.embedding for d in resp.data], dtype=np.float32
                )
                if vecs.shape != (len(texts), self.dim):
                    raise RuntimeError(
                        f"embedding shape {vecs.shape} != expected ({len(texts)}, {self.dim})"
                    )
                return vecs
            except (RateLimitError, APIConnectionError) as e:
                last_err = e
                continue
            except APIError as e:
                # APIError covers 4xx and 5xx; retry transient classes.
                msg = str(e).lower()
                transient = (
                    "429" in msg
                    or "too_many_requests" in msg
                    or "rate" in msg
                    or "500" in msg
                    or "502" in msg
                    or "503" in msg
                    or "504" in msg
                    or "timeout" in msg
                    or "service_unavailable" in msg
                )
                if transient:
                    last_err = e
                    continue
                raise
        raise RuntimeError(f"embed_batch exhausted retries: {last_err!r}")


def embed_dataframe(
    df: pd.DataFrame,
    bank: EmbeddingBank,
    embedder: OpenAIEmbedder,
    checkpoint_every: int = 50,
    progress_every: int = 10,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Embed every row of `df` (in catalog order). Reuses cached vectors
    when the per-row text hash is already in the bank; embeds the rest in
    batches; saves the bank periodically.

    Returns (vectors (N, dim) float32, item_ids (N,) str array, stats dict).
    """
    # 1. Compute texts + hashes for every row. Hashes include model + dim
    # + text_version so model/format changes invalidate cleanly.
    texts: list[str] = []
    hashes: list[str] = []
    for r in df.itertuples(index=False):
        t = embedding_text(r._asdict())
        texts.append(t)
        hashes.append(text_hash(t, model=embedder.model, dim=embedder.dim))
    item_ids = df["item_id"].astype(str).to_numpy()

    n = len(df)
    cache_hits = sum(1 for h in hashes if bank.has(h))
    to_embed_idx = [i for i, h in enumerate(hashes) if not bank.has(h)]
    print(
        f"  [embed] rows={n:,}  cache_hits={cache_hits:,}  "
        f"to_embed={len(to_embed_idx):,}"
    )

    # 2. Embed the misses in batches, saving the bank periodically.
    n_batches = (len(to_embed_idx) + embedder.batch_size - 1) // embedder.batch_size
    stats = {
        "n_rows": n,
        "cache_hits": cache_hits,
        "n_embedded": 0,
        "n_batches": n_batches,
        "started": time.time(),
    }

    t0 = time.time()
    for b in range(n_batches):
        chunk = to_embed_idx[b * embedder.batch_size : (b + 1) * embedder.batch_size]
        chunk_texts = [texts[i] for i in chunk]
        vecs = embedder.embed_batch(chunk_texts)
        for local, global_i in enumerate(chunk):
            bank.put(hashes[global_i], vecs[local])
        stats["n_embedded"] += len(chunk)

        if (b + 1) % checkpoint_every == 0 or (b + 1) == n_batches:
            bank.save()

        if (b + 1) % progress_every == 0 or (b + 1) == n_batches:
            elapsed = time.time() - t0
            rate = stats["n_embedded"] / elapsed if elapsed > 0 else 0.0
            remaining = (n_batches - (b + 1)) * embedder.batch_size
            eta_s = remaining / rate if rate > 0 else 0
            print(
                f"    [embed] batch {b+1}/{n_batches}  "
                f"{stats['n_embedded']:,} rows embedded  "
                f"{rate:.0f} rows/s  ETA {eta_s/60:.1f} min"
            )

    stats["wall_clock_s"] = round(time.time() - stats["started"], 1)

    # 3. Assemble per-row vector array in catalog order.
    out = np.zeros((n, embedder.dim), dtype=np.float32)
    for i, h in enumerate(hashes):
        v = bank.get(h)
        if v is None:
            raise RuntimeError(f"row {i} has no vector after embedding pass; bug")
        out[i] = v

    return out, item_ids, stats


def save_store_embeddings(
    vectors: np.ndarray,
    item_ids: np.ndarray,
    out_dir: Path,
    store_label: str,
) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    vec_path = out_dir / f"embeddings_{store_label}.npy"
    ids_path = out_dir / f"item_ids_{store_label}.npy"
    np.save(vec_path, vectors)
    np.save(ids_path, item_ids)
    return vec_path, ids_path


def load_store_embeddings(out_dir: Path, store_label: str) -> tuple[np.ndarray, np.ndarray]:
    vec_path = out_dir / f"embeddings_{store_label}.npy"
    ids_path = out_dir / f"item_ids_{store_label}.npy"
    return np.load(vec_path), np.load(ids_path, allow_pickle=True)
