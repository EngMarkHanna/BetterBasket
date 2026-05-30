"""Candidate retrieval subpackage.

Each retriever returns Candidate objects with provenance (which source
surfaced them, what raw score) so the audit trail survives all the way
to `matches.csv`.
"""

from .base import Candidate, CandidateRetriever
from .semantic import SemanticRetriever
from .bridge_system1 import StrictBlockRetriever, TfidfRetriever, PrivateLabelRetriever
from .union import union_candidates, to_dataframe

__all__ = [
    "Candidate",
    "CandidateRetriever",
    "SemanticRetriever",
    "StrictBlockRetriever",
    "TfidfRetriever",
    "PrivateLabelRetriever",
    "union_candidates",
    "to_dataframe",
]
