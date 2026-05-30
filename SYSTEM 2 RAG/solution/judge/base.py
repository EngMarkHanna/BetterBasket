"""Judge data shapes: request and result."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class JudgeRequest:
    """One judgment call: A row, ordered B candidate rows, optional
    pre-built RAG context.

    `extra_context_text` lets the caller inject already-formatted RAG
    snippets (so judges can be swapped without changing the orchestration
    layer above).
    """

    a_id: str
    a_row: dict[str, Any]
    b_ids: list[str]
    b_rows: list[dict[str, Any]]
    extra_context_text: str = ""
    context_signature: str = ""  # included in cache key


@dataclass
class JudgmentResult:
    """Parsed LLM output. Mirrors the System 1 structured schema and
    adds `tools_used` / `evidence_summary` slots even though Phase C
    does not use them (Phase D fills them).
    """

    a_id: str
    b_ids: list[str]
    best_candidate_index: int | None
    is_match: bool
    match_type: str
    confidence: float
    reason_codes: list[str] = field(default_factory=list)
    tools_used: list[str] = field(default_factory=list)
    evidence_summary: str = ""

    # Diagnostics.
    latency_s: float = 0.0
    json_valid: bool = True
    prompt_tokens: int = 0
    completion_tokens: int = 0
    raw_response: str = ""
    error: str | None = None

    def selected_b_id(self) -> str | None:
        idx = self.best_candidate_index
        if idx is None or idx < 0 or idx >= len(self.b_ids):
            return None
        return self.b_ids[idx]
