"""Candidate dataclass + retriever Protocol.

Every retriever returns a list of Candidate objects. The `source` field
labels the retriever; the `features` dict carries source-specific
signals (cosine, RapidFuzz score, brand block key, etc.) that the
scorer consumes downstream.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol


@dataclass
class Candidate:
    """One A->B candidate with provenance."""

    item_id_a: str
    item_id_b: str
    source: str  # 'T1' | 'T3' | 'T5' | 'T7' | concatenation like 'T1+T7'
    score: float  # source-specific raw score
    features: dict = field(default_factory=dict)


class CandidateRetriever(Protocol):
    """Anything that turns A rows into candidate B rows."""

    name: str

    def retrieve_all(self) -> list[Candidate]:
        """Return all candidates this retriever surfaces.

        Implementations decide internally whether to stream or
        materialize all at once.
        """
