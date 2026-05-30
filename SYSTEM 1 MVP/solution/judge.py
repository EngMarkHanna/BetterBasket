"""LLM judge: gpt-5.4-nano via Azure OpenAI, K=5 candidates per prompt,
structured JSON output, exponential backoff, JSONL cache.
"""
from __future__ import annotations

import hashlib
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import yaml
from openai import OpenAI, RateLimitError, APIError


RUBRIC_VERSION = "v2_2026_05_29"

JUDGE_RUBRIC = (
    "You are a grocery product matching judge. Given one A product and a numbered "
    "list of B candidates, pick the SINGLE best matching B (or none).\n"
    "Match types:\n"
    "- exact_national_brand: same brand (or curated alias), same specific product, "
    "compatible per-unit size, same form. Multipacks of the same per-unit SKU ARE "
    "acceptable. Word-order and marketing-copy drift are NOT match-breakers. "
    "Different flavors / formulations / forms / organic-vs-conventional ARE "
    "match-breakers for food.\n"
    "- private_label_equivalent: both store/private-label brands (Great Value, "
    "Marketside, Wegmans, Equate, etc.), same specific product, same size, same form. "
    "Different flavors or different products are NOT equivalent.\n"
    "- fresh_equivalent: produce, meat, deli, bakery equivalents with compatible "
    "size and form.\n"
    "- no_match: any meaningful difference. Be strict but do not reject over trivial "
    "naming or formatting differences when the underlying SKU is the same.\n"
    "Return ONLY the JSON object. If no candidate is acceptable, return "
    "best_candidate_index=null and is_match=false."
)


JUDGE_SCHEMA = {
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
                    "same_brand",
                    "alias_brand",
                    "compatible_size",
                    "multipack_equivalent",
                    "same_form",
                    "same_category",
                    "private_label_compatible",
                    "marketing_drift",
                    "brand_conflict",
                    "size_conflict",
                    "category_conflict",
                    "form_conflict",
                    "flavor_conflict",
                    "organic_conflict",
                    "domain_conflict",
                ],
            },
        },
    },
    "required": ["best_candidate_index", "is_match", "match_type", "confidence", "reason_codes"],
}


def load_creds(creds_path: Path) -> tuple[OpenAI, str]:
    creds = yaml.safe_load(creds_path.read_text())["openai"]
    client = OpenAI(
        base_url=creds["endpoint"],
        api_key=creds["api_key"],
        max_retries=6,
        timeout=60.0,
    )
    return client, creds["deployment_name"]


def _call_with_backoff(client: OpenAI, **kwargs) -> Any:
    delays = [0.0, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0]
    last_err: Exception | None = None
    for delay in delays:
        if delay:
            time.sleep(delay)
        try:
            return client.chat.completions.create(**kwargs)
        except RateLimitError as e:
            last_err = e
            continue
        except APIError as e:
            if "429" in str(e) or "too_many_requests" in str(e).lower():
                last_err = e
                continue
            raise
    raise last_err if last_err else RuntimeError("retry loop exhausted")


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


def build_prompt(a_row: dict, b_rows: list[dict], rag_snippets: list[dict]) -> str:
    parts = [format_a_block(a_row), ""]
    for i, b in enumerate(b_rows):
        parts.append(format_b_block(i, b))
    parts.append("")
    if rag_snippets:
        parts.append("Relevant rules and precedents:")
        for s in rag_snippets:
            parts.append(f"- ({s['title']}) {s['content']}")
        parts.append("")
    parts.append("Choose the single best B index (0..N-1) or null. Return the JSON object only.")
    return "\n".join(parts)


def cache_key(a_id: str, b_ids: list[str], rubric_version: str) -> str:
    h = hashlib.sha256()
    h.update(rubric_version.encode())
    h.update(a_id.encode())
    for bid in sorted(b_ids):
        h.update(b"|")
        h.update(bid.encode())
    return h.hexdigest()


class JudgmentCache:
    def __init__(self, jsonl_path: Path):
        self.path = jsonl_path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.mem: dict[str, dict] = {}
        if jsonl_path.exists():
            with jsonl_path.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                    except Exception:
                        continue
                    key = obj.get("cache_key")
                    if key:
                        self.mem[key] = obj

    def get(self, key: str) -> dict | None:
        return self.mem.get(key)

    def put(self, key: str, record: dict) -> None:
        self.mem[key] = record
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")


def judge_batch(
    client: OpenAI,
    model: str,
    a_row: dict,
    b_rows: list[dict],
    rag_snippets: list[dict],
    cache: JudgmentCache,
    a_id: str,
    b_ids: list[str],
    per_call_sleep: float = 0.4,
) -> dict:
    key = cache_key(a_id, b_ids, RUBRIC_VERSION)
    cached = cache.get(key)
    if cached is not None:
        return cached

    if per_call_sleep:
        time.sleep(per_call_sleep)

    t0 = time.time()
    prompt = build_prompt(a_row, b_rows, rag_snippets)
    resp = _call_with_backoff(
        client,
        model=model,
        messages=[
            {"role": "system", "content": JUDGE_RUBRIC},
            {"role": "user", "content": prompt},
        ],
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "match_judgment",
                "strict": True,
                "schema": JUDGE_SCHEMA,
            },
        },
        max_completion_tokens=4000,
        reasoning_effort="minimal",
    )
    dt = time.time() - t0
    raw = resp.choices[0].message.content
    try:
        parsed = json.loads(raw) if raw else None
    except Exception:
        parsed = None

    usage = resp.usage
    record = {
        "cache_key": key,
        "a_id": a_id,
        "b_ids": b_ids,
        "latency_s": round(dt, 3),
        "json_valid": parsed is not None,
        "raw_response": raw,
        "best_candidate_index": (parsed or {}).get("best_candidate_index"),
        "is_match": (parsed or {}).get("is_match", False),
        "match_type": (parsed or {}).get("match_type", "no_match"),
        "confidence": float((parsed or {}).get("confidence", 0.0)),
        "reason_codes": (parsed or {}).get("reason_codes", []),
        "prompt_tokens": usage.prompt_tokens,
        "completion_tokens": usage.completion_tokens,
        "total_tokens": usage.total_tokens,
    }
    cache.put(key, record)
    return record


def judge_many(
    client: OpenAI,
    model: str,
    batches: list[tuple[str, dict, list[str], list[dict], list[dict]]],
    cache: JudgmentCache,
    workers: int = 5,
    per_call_sleep: float = 0.4,
    progress_every: int = 100,
) -> list[dict]:
    """Run a list of (a_id, a_row, b_ids, b_rows, rag_snippets) judgments
    concurrently. Returns the list of judgment records in submission order.
    """
    results: list[dict] = [None] * len(batches)  # type: ignore[list-item]
    submitted = 0
    completed = 0
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {}
        for i, (a_id, a_row, b_ids, b_rows, rag) in enumerate(batches):
            fut = pool.submit(
                judge_batch,
                client,
                model,
                a_row,
                b_rows,
                rag,
                cache,
                a_id,
                b_ids,
                per_call_sleep,
            )
            futures[fut] = i
            submitted += 1
        for fut in as_completed(futures):
            i = futures[fut]
            try:
                results[i] = fut.result()
            except Exception as e:
                results[i] = {"a_id": batches[i][0], "error": str(e), "is_match": False, "confidence": 0.0}
            completed += 1
            if completed % progress_every == 0 or completed == len(batches):
                elapsed = time.time() - t0
                rate = completed / elapsed if elapsed > 0 else 0.0
                print(
                    f"  judged {completed}/{len(batches)}  "
                    f"({rate:.1f} pair/s, {elapsed:.0f}s elapsed)"
                )
    return results
