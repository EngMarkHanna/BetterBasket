"""Final selection: one B per A.

Highest final_score wins. Ties broken in this order (audit fix #15):
  1. final_score desc
  2. llm_confidence desc (if column present)
  3. candidate_score desc
  4. item_id_b asc (stable)

Multiple A rows can map to the same B (that's fine: distinct A SKUs can
share a B fresh-produce or single-pack equivalent).
"""
from __future__ import annotations

import pandas as pd


def select_one_b_per_a(scored: pd.DataFrame) -> pd.DataFrame:
    """Return one row per item_id_a using the documented tie-break order."""
    if scored.empty:
        return scored
    df = scored.copy()
    # Sort multi-key; ascending=False on score-like columns means desc.
    sort_cols: list[str] = ["final_score"]
    ascending: list[bool] = [False]
    if "llm_confidence" in df.columns:
        df["llm_confidence"] = pd.to_numeric(df["llm_confidence"], errors="coerce")
        sort_cols.append("llm_confidence")
        ascending.append(False)
    if "candidate_score" in df.columns:
        df["candidate_score"] = pd.to_numeric(df["candidate_score"], errors="coerce")
        sort_cols.append("candidate_score")
        ascending.append(False)
    sort_cols.append("item_id_b")
    ascending.append(True)
    df = df.sort_values(sort_cols, ascending=ascending, kind="mergesort", na_position="last")
    selected = df.drop_duplicates(subset=["item_id_a"], keep="first")
    return selected
