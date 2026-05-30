"""Tool-less RAG judge: gpt-5.4-nano with pre-fetched context.

Differences vs System 1's judge:
  - RAG context block injected into the user prompt (deterministic
    pre-fetch, NOT model-driven tool calls).
  - Cache key includes prompt format version, model, schema version,
    rubric version, RAG context signature, and ORDERED candidate IDs
    + text hashes (audit #11, #12).
  - Retry logic covers 429, 5xx, timeouts, connection errors (audit #13).
"""
from __future__ import annotations

import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

from openai import OpenAI, APIError, APIConnectionError, RateLimitError

from .base import JudgeRequest, JudgmentResult
from .cache import CacheVersion, JudgmentCache, build_cache_key, quick_hash


JUDGE_RUBRIC = (
    "You are a grocery product matching judge. Given one A product and a "
    "numbered list of B candidates, pick the SINGLE best matching B (or "
    "none).\n"
    "Match types:\n"
    "- exact_national_brand: same brand (or curated alias), same specific "
    "product, compatible per-unit size, same form. Multipacks of the same "
    "per-unit SKU ARE acceptable. Word-order and marketing-copy drift are "
    "NOT match-breakers. Different flavors / formulations / forms / "
    "organic-vs-conventional ARE match-breakers for food.\n"
    "- private_label_equivalent: both store/private-label brands "
    "(Great Value, Marketside, Wegmans, Equate, etc.), same specific "
    "product, same size, same form. Different flavors or different "
    "products are NOT equivalent.\n"
    "- fresh_equivalent: produce, meat, deli, bakery equivalents with "
    "compatible size and form.\n"
    "- no_match: any meaningful difference. Be strict but do not reject "
    "over trivial naming or formatting differences when the underlying "
    "SKU is the same.\n"
    "Use the 'Relevant rules', 'Brand alias checks', 'Category bridge', "
    "and 'Similar examples' sections (when present) as evidence. They are "
    "pre-fetched for this pair specifically. Return ONLY the JSON object."
)

JUDGE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "best_candidate_index": {"type": ["integer", "null"]},
        "is_match": {"type": "boolean"},
        "match_type": {
            "type": "string",
            "enum": [
                "exact_national_brand",
                "private_label_equivalent",
                "fresh_equivalent",
                "no_match",
            ],
        },
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        "reason_codes": {
            "type": "array",
            "items": {
                "type": "string",
                "enum": [
                    "same_brand", "alias_brand", "compatible_size",
                    "multipack_equivalent", "same_form", "same_category",
                    "private_label_compatible", "marketing_drift",
                    "brand_conflict", "size_conflict", "category_conflict",
                    "form_conflict", "flavor_conflict", "organic_conflict",
                    "domain_conflict",
                ],
            },
        },
        "evidence_summary": {"type": "string"},
    },
    "required": [
        "best_candidate_index", "is_match", "match_type",
        "confidence", "reason_codes", "evidence_summary",
    ],
}

DEFAULT_CACHE_VERSION = CacheVersion(
    rubric_version="rag_v1",
    model="gpt-5.4-nano",
    schema_version="rag_match_judgment_v1",
    prompt_format_version="rag_v1",
)


def format_a_block(a_row: dict) -> str:
    return (
        "A:\n"
        f"  name: {a_row.get('name')}\n"
        f"  brand: {a_row.get('brand_raw')}\n"
        f"  size: {a_row.get('size_text')}\n"
        f"  category: {a_row.get('category_path_norm')}\n"
        f"  private_label: {bool(a_row.get('is_private_label_inferred'))}\n"
    )


def format_b_block(idx: int, b_row: dict) -> str:
    return (
        f"B[{idx}]:\n"
        f"  name: {b_row.get('name')}\n"
        f"  brand: {b_row.get('brand_raw')}\n"
        f"  size: {b_row.get('size_text')}\n"
        f"  category: {b_row.get('category_path_norm')}\n"
        f"  private_label: {bool(b_row.get('is_private_label_inferred'))}\n"
    )


def render_prompt(request: JudgeRequest) -> str:
    parts = [format_a_block(request.a_row), ""]
    for i, b in enumerate(request.b_rows):
        parts.append(format_b_block(i, b))
    parts.append("")
    if request.extra_context_text:
        parts.append(request.extra_context_text)
        parts.append("")
    parts.append(
        "Choose the single best B index (0..N-1) or null. Return JSON only."
    )
    return "\n".join(parts)


class RAGJudge:
    """Thread-pool driver for batched judgments.

    The class owns the cache so concurrent threads can hit it safely
    (the cache uses an internal dict and append-only writes).
    """

    def __init__(
        self,
        client: OpenAI,
        model: str,
        cache: JudgmentCache,
        version: CacheVersion = DEFAULT_CACHE_VERSION,
        knowledge_version: str = "v1",
        per_call_sleep: float = 0.4,
        max_tokens: int = 4000,
    ):
        self.client = client
        self.model = model
        self.cache = cache
        self.version = version
        self.knowledge_version = knowledge_version
        self.per_call_sleep = per_call_sleep
        self.max_tokens = max_tokens

    def _call_with_backoff(self, **kwargs) -> Any:
        delays = [0.0, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0]
        last_err: Exception | None = None
        for delay in delays:
            if delay:
                time.sleep(delay)
            try:
                return self.client.chat.completions.create(**kwargs)
            except (RateLimitError, APIConnectionError) as e:
                last_err = e
                continue
            except APIError as e:
                msg = str(e).lower()
                transient = (
                    "429" in msg
                    or "too_many_requests" in msg
                    or "rate" in msg
                    or "500" in msg
                    or "502" in msg
                    or "503" in msg
                    or "504" in msg
                    or "timeout" in msg
                )
                if transient:
                    last_err = e
                    continue
                raise
        raise RuntimeError(f"judge call exhausted retries: {last_err!r}")

    def _cache_key(self, req: JudgeRequest) -> str:
        a_text = format_a_block(req.a_row)
        b_texts = [format_b_block(i, b) for i, b in enumerate(req.b_rows)]
        return build_cache_key(
            version=self.version,
            a_id=req.a_id,
            b_ids=req.b_ids,
            a_text_hash=quick_hash(a_text),
            b_text_hashes=[quick_hash(t) for t in b_texts],
            context_signature=req.context_signature,
            knowledge_version=self.knowledge_version,
        )

    def judge_one(self, request: JudgeRequest) -> JudgmentResult:
        key = self._cache_key(request)
        cached = self.cache.get(key)
        if cached is not None:
            return cached

        if self.per_call_sleep:
            time.sleep(self.per_call_sleep)

        prompt = render_prompt(request)
        t0 = time.time()
        resp = self._call_with_backoff(
            model=self.model,
            messages=[
                {"role": "system", "content": JUDGE_RUBRIC},
                {"role": "user", "content": prompt},
            ],
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "rag_match_judgment",
                    "strict": True,
                    "schema": JUDGE_SCHEMA,
                },
            },
            max_completion_tokens=self.max_tokens,
            reasoning_effort="minimal",
        )
        dt = time.time() - t0
        raw = resp.choices[0].message.content
        try:
            parsed = json.loads(raw) if raw else None
        except json.JSONDecodeError:
            parsed = None

        result = JudgmentResult(
            a_id=request.a_id,
            b_ids=list(request.b_ids),
            best_candidate_index=(parsed or {}).get("best_candidate_index"),
            is_match=bool((parsed or {}).get("is_match", False)),
            match_type=(parsed or {}).get("match_type", "no_match"),
            confidence=float((parsed or {}).get("confidence", 0.0)),
            reason_codes=(parsed or {}).get("reason_codes", []),
            evidence_summary=(parsed or {}).get("evidence_summary", ""),
            latency_s=round(dt, 3),
            json_valid=parsed is not None,
            prompt_tokens=int(resp.usage.prompt_tokens),
            completion_tokens=int(resp.usage.completion_tokens),
            raw_response=raw or "",
        )
        self.cache.put(key, result)
        return result

    def judge_many(
        self,
        requests: list[JudgeRequest],
        workers: int = 5,
        progress_every: int = 100,
    ) -> list[JudgmentResult]:
        results: list[JudgmentResult | None] = [None] * len(requests)
        completed = 0
        t0 = time.time()
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(self.judge_one, r): i for i, r in enumerate(requests)}
            for fut in as_completed(futures):
                i = futures[fut]
                try:
                    results[i] = fut.result()
                except Exception as e:
                    results[i] = JudgmentResult(
                        a_id=requests[i].a_id,
                        b_ids=list(requests[i].b_ids),
                        best_candidate_index=None,
                        is_match=False,
                        match_type="no_match",
                        confidence=0.0,
                        error=repr(e),
                    )
                completed += 1
                if completed % progress_every == 0 or completed == len(requests):
                    elapsed = time.time() - t0
                    rate = completed / elapsed if elapsed > 0 else 0.0
                    print(
                        f"  [judge] {completed}/{len(requests)}  "
                        f"({rate:.1f} req/s, {elapsed:.0f}s elapsed)"
                    )
        return [r for r in results if r is not None]
