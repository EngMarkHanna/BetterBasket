"""Union + dedupe of candidates from multiple retrievers.

Same key (item_id_a, item_id_b) -> single Candidate with combined
`source` (e.g. 'T1+T3+T7') and merged features.
"""
from __future__ import annotations

from typing import Iterable

import pandas as pd

from .base import Candidate


def union_candidates(*lists: Iterable[Candidate]) -> list[Candidate]:
    merged: dict[tuple[str, str], Candidate] = {}
    for lst in lists:
        for cand in lst:
            key = (cand.item_id_a, cand.item_id_b)
            if key not in merged:
                merged[key] = Candidate(
                    item_id_a=cand.item_id_a,
                    item_id_b=cand.item_id_b,
                    source=cand.source,
                    score=cand.score,
                    features=dict(cand.features),
                )
            else:
                existing = merged[key]
                existing_sources = set(existing.source.split("+"))
                new_sources = set(cand.source.split("+"))
                combined = existing_sources | new_sources
                existing.source = "+".join(sorted(combined))
                existing.features.update(cand.features)
                # Keep the max so the union score is the strongest signal.
                existing.score = max(existing.score, cand.score)
    return list(merged.values())


def to_dataframe(candidates: list[Candidate]) -> pd.DataFrame:
    rows = []
    for c in candidates:
        row = {
            "item_id_a": c.item_id_a,
            "item_id_b": c.item_id_b,
            "candidate_source": c.source,
            "candidate_score": c.score,
        }
        row.update(c.features)
        rows.append(row)
    return pd.DataFrame(rows)
