"""Knowledge base subpackage.

The knowledge base is a versioned corpus of small text entries the
RAG judge retrieves from per pair:
  - rules     : decomposed rubric clauses
  - aliases   : curated brand-equivalence entries
  - bridges   : learned A->B category mappings
  - examples  : labeled positive/negative pairs with reasons
  - edges     : known tricky patterns

Each entry has structured applicability fields so retrieval can be
filtered by `match_type` or `requires_brand_relation` rather than
relying solely on text similarity.
"""

from .entry import KnowledgeEntry, EntryType
from .bootstrap import BootstrapStats, bootstrap_all
from .index import KnowledgeRetriever, RetrievedEntry

__all__ = [
    "KnowledgeEntry",
    "EntryType",
    "BootstrapStats",
    "bootstrap_all",
    "KnowledgeRetriever",
    "RetrievedEntry",
]
