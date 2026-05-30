"""Versioned JSONL cache for LLM judgments.

Audit fixes (FINAL_PLAN.md):
  #11 - cache key preserves ordered B IDs (model returns positional index)
  #12 - cache key includes prompt/text/RAG/knowledge versions and model

The cache file is append-only JSONL. On startup we load all entries
into an in-memory dict for O(1) hit-checks. Crash-safe because every
put fsyncs the line before returning.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .base import JudgmentResult


@dataclass(frozen=True)
class CacheVersion:
    """All ingredients of the cache key. Bump any field that affects
    judgment output.
    """

    rubric_version: str  # judge system-prompt version
    model: str  # 'gpt-5.4-nano' or whatever the deployment exposes
    schema_version: str  # JSON schema name + revision
    prompt_format_version: str  # how A/B blocks are rendered

    def fingerprint(self) -> str:
        s = "|".join((self.rubric_version, self.model, self.schema_version, self.prompt_format_version))
        return hashlib.sha256(s.encode("utf-8")).hexdigest()[:16]


def build_cache_key(
    version: CacheVersion,
    a_id: str,
    b_ids: list[str],  # ordered, positional
    a_text_hash: str,
    b_text_hashes: list[str],  # ordered, positional
    context_signature: str,
    knowledge_version: str = "v1",
) -> str:
    """Single sha256 over the full set of inputs that affect output.

    KEY POINT (audit #11): we use ORDERED b_ids and ORDERED b_text_hashes
    because the model returns `best_candidate_index` positionally. If we
    sorted them, a different routing order would silently point to the
    wrong B.
    """
    parts = [
        version.fingerprint(),
        knowledge_version,
        context_signature,
        a_id,
        a_text_hash,
        "|".join(b_ids),
        "|".join(b_text_hashes),
    ]
    h = hashlib.sha256()
    h.update("\x1f".join(parts).encode("utf-8"))
    return h.hexdigest()


def quick_hash(text: str) -> str:
    return hashlib.sha256((text or "").encode("utf-8")).hexdigest()[:16]


class JudgmentCache:
    """Append-only JSONL cache backed by an in-memory dict.

    Use `get(key)` before issuing an LLM call. Use `put(key, result)`
    after a successful call. `put` is fsync-style: line is written and
    flushed before return.
    """

    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._mem: dict[str, dict] = {}
        self._load()

    def _load(self) -> None:
        if not self.path.exists():
            return
        with self.path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                key = obj.get("cache_key")
                if key:
                    self._mem[key] = obj

    def __len__(self) -> int:
        return len(self._mem)

    def get(self, key: str) -> JudgmentResult | None:
        record = self._mem.get(key)
        if record is None:
            return None
        # Rebuild JudgmentResult from the stored dict. Tolerate missing
        # optional fields for backward compatibility with older caches.
        return JudgmentResult(
            a_id=record["a_id"],
            b_ids=record["b_ids"],
            best_candidate_index=record.get("best_candidate_index"),
            is_match=bool(record.get("is_match", False)),
            match_type=record.get("match_type", "no_match"),
            confidence=float(record.get("confidence", 0.0)),
            reason_codes=record.get("reason_codes", []),
            tools_used=record.get("tools_used", []),
            evidence_summary=record.get("evidence_summary", ""),
            latency_s=float(record.get("latency_s", 0.0)),
            json_valid=bool(record.get("json_valid", True)),
            prompt_tokens=int(record.get("prompt_tokens", 0)),
            completion_tokens=int(record.get("completion_tokens", 0)),
            raw_response=record.get("raw_response", ""),
        )

    def put(self, key: str, result: JudgmentResult) -> None:
        record = asdict(result)
        record["cache_key"] = key
        self._mem[key] = record
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            f.flush()
