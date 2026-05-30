"""Thin wrappers around System 1's T1/T3/T5 retrievers.

We reuse System 1's implementation so behaviour stays comparable; this
module only adapts their function-style API into the CandidateRetriever
Protocol.

If System 1's retrieve.py changes, behaviour here changes too. That's
intentional - we want comparison parity.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from .base import Candidate, CandidateRetriever
from .._system1_loader import system1_retrieve

t1_strict_blocks = system1_retrieve.t1_strict_blocks
t3_tfidf_topk = system1_retrieve.t3_tfidf_topk
t5_private_label = system1_retrieve.t5_private_label


def _wrap(cands: list[Any], source_label: str) -> list[Candidate]:
    out: list[Candidate] = []
    for c in cands:
        out.append(
            Candidate(
                item_id_a=c.item_id_a,
                item_id_b=c.item_id_b,
                source=source_label,
                score=float(c.score),
                features=dict(c.features),
            )
        )
    return out


class StrictBlockRetriever(CandidateRetriever):
    name = "T1"

    def __init__(self, a: pd.DataFrame, b: pd.DataFrame, top_per_a: int = 3):
        self._a = a
        self._b = b
        self._top_per_a = top_per_a

    def retrieve_all(self) -> list[Candidate]:
        return _wrap(t1_strict_blocks(self._a, self._b, self._top_per_a), self.name)


class TfidfRetriever(CandidateRetriever):
    name = "T3"

    def __init__(
        self,
        a: pd.DataFrame,
        b: pd.DataFrame,
        k: int = 20,
        cosine_floor: float = 0.4,
    ):
        self._a = a
        self._b = b
        self._k = k
        self._cosine_floor = cosine_floor

    def retrieve_all(self) -> list[Candidate]:
        return _wrap(
            t3_tfidf_topk(self._a, self._b, k=self._k, cosine_floor=self._cosine_floor),
            self.name,
        )


class PrivateLabelRetriever(CandidateRetriever):
    name = "T5"

    def __init__(self, a: pd.DataFrame, b: pd.DataFrame, bridge_csv: Path, top_per_a: int = 5):
        self._a = a
        self._b = b
        self._bridge_csv = bridge_csv
        self._top_per_a = top_per_a

    def retrieve_all(self) -> list[Candidate]:
        return _wrap(
            t5_private_label(self._a, self._b, self._bridge_csv, self._top_per_a),
            self.name,
        )
