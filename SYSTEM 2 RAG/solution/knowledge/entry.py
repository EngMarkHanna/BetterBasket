"""Knowledge-base entry schema.

A KnowledgeEntry is one self-contained piece of context the RAG judge
might want to see when deciding a pair. The schema is deliberately
strict so retrieval can filter by structured applicability before
falling back to text similarity.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
from typing import Any, Iterable


class EntryType(str, Enum):
    """Type of knowledge entry. Drives default filtering behavior."""

    RULE = "rule"
    ALIAS = "alias"
    BRIDGE = "bridge"
    ACCEPTED_EXAMPLE = "accepted_example"
    REJECTED_EXAMPLE = "rejected_example"
    EDGE_CASE = "edge_case"


@dataclass
class KnowledgeEntry:
    """One knowledge entry.

    The `embedding_text` field is what we send to the embedder. If left
    None, falls back to `title + content`. We embed the entry once at
    bootstrap time and retrieve by cosine similarity at judge time.

    Applicability fields let us pre-filter retrieved entries so a
    private-label-organic exception does not leak into a national-brand
    food judgment.
    """

    # Stable identifier; appears in audit trails.
    id: str
    type: EntryType
    title: str
    content: str

    # Optional structured applicability fields. None means "applies to
    # any pair of this type"; a specific value restricts retrieval.
    match_type: str | None = None  # exact_national_brand | private_label_equivalent | fresh_equivalent
    product_domain: str | None = None  # food | beverage | personal_care | household | pet | ...
    requires_brand_relation: str | None = None  # exact | alias | private_label_compatible
    requires_size_relation: str | None = None  # same | multipack | near
    store_scope: str | None = None  # A | B | both | None (means both)

    # Free-form tags for soft-filtering / debugging.
    tags: list[str] = field(default_factory=list)

    # Provenance and confidence.
    source: str = "manual"
    confidence: float = 1.0

    # Optional override of what gets embedded. None -> title + content.
    embedding_text: str | None = None

    def text_for_embedding(self) -> str:
        if self.embedding_text:
            return self.embedding_text
        return f"{self.title}. {self.content}"

    def to_jsonl(self) -> str:
        d = asdict(self)
        d["type"] = self.type.value
        return json.dumps(d, ensure_ascii=False)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "KnowledgeEntry":
        d = dict(d)  # defensive copy
        d["type"] = EntryType(d["type"])
        d.setdefault("tags", [])
        d.setdefault("source", "manual")
        d.setdefault("confidence", 1.0)
        return cls(**d)


def write_jsonl(entries: Iterable[KnowledgeEntry], path: Path) -> int:
    """Write entries to a JSONL file. Returns count written."""
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8") as f:
        for e in entries:
            f.write(e.to_jsonl() + "\n")
            n += 1
    return n


def read_jsonl(path: Path) -> list[KnowledgeEntry]:
    """Read entries from a JSONL file. Missing path -> empty list."""
    if not path.exists():
        return []
    out: list[KnowledgeEntry] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(KnowledgeEntry.from_dict(json.loads(line)))
    return out


def read_all(knowledge_dir: Path) -> list[KnowledgeEntry]:
    """Load every JSONL file in a knowledge directory."""
    if not knowledge_dir.exists():
        return []
    out: list[KnowledgeEntry] = []
    for p in sorted(knowledge_dir.glob("*.jsonl")):
        out.extend(read_jsonl(p))
    return out
