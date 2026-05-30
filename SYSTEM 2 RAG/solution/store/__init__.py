"""Vector store subpackage.

A VectorStore is the minimal interface for ANN-style retrieval. We
ship a numpy implementation that handles <1M vectors comfortably. At
larger scales, swap in FAISS or pgvector; the rest of the pipeline
doesn't change.
"""

from .base import VectorStore, SearchHit
from .numpy_store import NumpyVectorStore

__all__ = ["VectorStore", "SearchHit", "NumpyVectorStore"]
