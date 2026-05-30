"""LLM smoke test against the BetterBasket Azure OpenAI deployment.

Probes:
  1. Connectivity & token-usage shape (reasoning tokens?)
  2. Structured-output validity on 10 hand-curated pairs
  3. Same 10 pairs at reasoning_effort="minimal" (latency/cost comparison)
  4. Per-prompt candidate batching (1 A + K B) for the cost lever
  5. Batch API submission shape (syntactic check; not waited on)

Reads credentials from openai_creds.yaml at the repo root.
Writes outputs/llm_smoke_test.json and outputs/llm_smoke_test.md.

Reference: Azure OpenAI v1 surface (learn.microsoft.com) confirms
`OpenAI` client with `base_url=...` works, `max_completion_tokens` is
mandatory for gpt-5 family, reasoning tokens count toward completion,
and `response_format={"type":"json_schema",...,"strict":true}` is supported.
"""
from __future__ import annotations

import json
import statistics
import time
from pathlib import Path
from typing import Any

import yaml
from openai import BadRequestError, OpenAI

ROOT = Path(__file__).resolve().parents[1]
CREDS_PATH = ROOT / "openai_creds.yaml"
OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"
SUMMARY_JSON = OUTPUT_DIR / "llm_smoke_test.json"
SUMMARY_MD = OUTPUT_DIR / "llm_smoke_test.md"


JUDGE_RUBRIC = (
    "You are a grocery product matching judge. Given a pair of products from two "
    "different grocery stores, decide whether a shopper would treat them as "
    "essentially the same product.\n"
    "Match types:\n"
    "- exact_national_brand: same brand, same product, compatible size/form.\n"
    "- private_label_equivalent: both are store/private-label brands, same product "
    "category, compatible size and form.\n"
    "- fresh_equivalent: produce/meat/deli equivalents.\n"
    "- no_match: different category, different product family, incompatible size, "
    "different flavor/format, or non-food vs food.\n"
    "Be strict. When uncertain, prefer no_match.\n"
    "Return only the JSON object - no preamble."
)


JUDGE_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
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
                    "same_form",
                    "same_category",
                    "private_label_compatible",
                    "brand_conflict",
                    "size_conflict",
                    "category_conflict",
                    "form_conflict",
                    "flavor_conflict",
                    "domain_conflict",
                ],
            },
        },
    },
    "required": ["is_match", "match_type", "confidence", "reason_codes"],
}


RANK_SCHEMA = {
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
        "reason_codes": {"type": "array", "items": {"type": "string"}},
    },
    "required": [
        "best_candidate_index",
        "is_match",
        "match_type",
        "confidence",
        "reason_codes",
    ],
}


# Hand-curated pairs lifted from outputs/candidate_examples.csv plus
# obvious negatives. expected_label is *my* judgment of the right answer.
PAIRS: list[dict[str, Any]] = [
    {
        "name_A": "RUSSELL STOVER Christmas Sugar Free Assorted Chocolate Candy Gift Box, 6.2 oz.",
        "brand_A": "Russell Stover", "size_A": "6.2 oz", "category_A": "Food > Candy > Christmas Candy",
        "name_B": "Russell Stover Sugar Free Assorted Christmas Candy Gift Box",
        "brand_B": "Russell Stover", "size_B": "6.2 ounce", "category_B": "Grocery > Candy > Seasonal & Holiday Candy",
        "expected_label": "exact_national_brand",
    },
    {
        "name_A": "Bounty Paper Napkins, White, 200 Count",
        "brand_A": "Bounty", "size_A": "", "category_A": "Party & Occasions > Party Supplies > Party Tableware",
        "name_B": "Bounty Paper Napkins, White",
        "brand_B": "Bounty", "size_B": "200 ct.", "category_B": "More Departments > Household Essentials > Paper & Plastic",
        "expected_label": "exact_national_brand",
    },
    {
        "name_A": "Lotus Foods Gluten-Free Organic Brown Udon Rice Noodles, 3-Pack, 8 oz.",
        "brand_A": "Lotus Foods", "size_A": "8 oz", "category_A": "Food > Pantry > Pasta & pizza",
        "name_B": "Lotus Foods Rice Noodles, Udon, Brown",
        "brand_B": "Lotus Foods", "size_B": "8 ounce", "category_B": "Grocery > International Foods > Asian",
        "expected_label": "exact_national_brand",
    },
    {
        "name_A": "Soapbox Coconut & Shea Deep Moisture Shampoo with Vitamin E and Shea, 16 fl oz",
        "brand_A": "Soapbox", "size_A": "16 Fl Oz", "category_A": "Beauty > Hair Care > Shampoo",
        "name_B": "Soapbox Shampoo, Deep Moisture, Coconut & Shea",
        "brand_B": "Soapbox", "size_B": "16 fl. oz.", "category_B": "More Departments > Personal Care and Makeup > Hair Care",
        "expected_label": "exact_national_brand",
    },
    {
        "name_A": "(2 pack) Hershey's Kisses Grinch Milk Chocolate Christmas Candy, Bag 9.5 oz",
        "brand_A": "Hershey's", "size_A": "9.5 oz", "category_A": "Food > Candy > Shop by Brand",
        "name_B": "Hershey's Kisses Candy, Milk Chocolate",
        "brand_B": "Hershey's", "size_B": "9.5 ounce", "category_B": "Grocery > Candy > Seasonal & Holiday Candy",
        "expected_label": "exact_national_brand",
    },
    {
        "name_A": "M&M's Peanut Butter Chocolate Christmas Candy - 10 oz Bag",
        "brand_A": "M&M'S", "size_A": "10 Oz", "category_A": "Food > Candy > Chocolate",
        "name_B": "M&M'S Milk Chocolate Christmas Candy Bag",
        "brand_B": "M&M'S", "size_B": "10 ounce", "category_B": "Grocery > Candy > Seasonal & Holiday Candy",
        "expected_label": "no_match",
    },
    {
        "name_A": "Tide Pods HE Laundry Detergent Pods Free and Gentle 152 Count",
        "brand_A": "Tide", "size_A": "152 ct", "category_A": "Household Essentials > Laundry > Laundry Detergents",
        "name_B": "Tide Free & Gentle Liquid Laundry Detergent",
        "brand_B": "Tide", "size_B": "105 fl. oz.", "category_B": "More Departments > Household Essentials > Laundry & Laundry Supplies",
        "expected_label": "no_match",
    },
    {
        "name_A": "Diet Dr Pepper Soda Pop, .5 L bottles, 12 pack",
        "brand_A": "Diet Dr. Pepper", "size_A": "12 x 16.9 fl oz", "category_A": "Food > Beverages > Soda",
        "name_B": "Diet Dr Pepper Mini Soda",
        "brand_B": "Diet Dr Pepper", "size_B": "6 x 7.5 fl. oz.", "category_B": "Grocery > Beverages > Soda & Pop",
        "expected_label": "no_match",
    },
    {
        "name_A": "Q-Tips Cotton Swabs Purse Travel Size Pack, 30 Count Pack of 12",
        "brand_A": "Q-tips", "size_A": "30 ct x 12", "category_A": "Beauty > Makeup > Makeup Tools & Brushes",
        "name_B": "Q-Tips Cotton Swabs",
        "brand_B": "Q-Tips", "size_B": "500 ct.", "category_B": "More Departments > Personal Care and Makeup > Cotton Swabs, Rounds & Balls",
        "expected_label": "no_match",
    },
    {
        "name_A": "HART 140-Piece 1/4 and 3/8-inch Drive Mechanics Tool Set, Chrome Finish",
        "brand_A": "HART", "size_A": "140 pc", "category_A": "Home Improvement > Tools > Hand Tools",
        "name_B": "Wegmans Honey Ham, Thin Shaved Chipped",
        "brand_B": "Wegmans", "size_B": "", "category_B": "More Departments > Deli > Ham",
        "expected_label": "no_match",
    },
]


def format_pair(p: dict[str, Any]) -> str:
    return (
        "Store A product:\n"
        f"  name: {p['name_A']}\n"
        f"  brand: {p['brand_A']}\n"
        f"  size: {p['size_A']}\n"
        f"  category: {p['category_A']}\n"
        "Store B product:\n"
        f"  name: {p['name_B']}\n"
        f"  brand: {p['brand_B']}\n"
        f"  size: {p['size_B']}\n"
        f"  category: {p['category_B']}\n"
    )


def usage_dict(usage) -> dict[str, Any]:
    d = {
        "prompt_tokens": getattr(usage, "prompt_tokens", None),
        "completion_tokens": getattr(usage, "completion_tokens", None),
        "total_tokens": getattr(usage, "total_tokens", None),
    }
    details = getattr(usage, "completion_tokens_details", None)
    if details is not None:
        d["reasoning_tokens"] = getattr(details, "reasoning_tokens", None)
        d["accepted_prediction_tokens"] = getattr(details, "accepted_prediction_tokens", None)
    prompt_details = getattr(usage, "prompt_tokens_details", None)
    if prompt_details is not None:
        d["cached_prompt_tokens"] = getattr(prompt_details, "cached_tokens", None)
    return d


def make_client() -> tuple[OpenAI, str]:
    creds = yaml.safe_load(CREDS_PATH.read_text())["openai"]
    client = OpenAI(base_url=creds["endpoint"], api_key=creds["api_key"])
    return client, creds["deployment_name"]


def _try_call(client: OpenAI, **kwargs) -> Any:
    """Call chat.completions, falling back gracefully if a param is rejected."""
    try:
        return client.chat.completions.create(**kwargs)
    except BadRequestError as e:
        msg = str(e)
        if "reasoning_effort" in msg and "reasoning_effort" in kwargs:
            kwargs.pop("reasoning_effort")
            return client.chat.completions.create(**kwargs)
        raise


def probe_connectivity(client: OpenAI, model: str) -> dict[str, Any]:
    print("=== Probe 1: connectivity ===")
    t0 = time.time()
    resp = _try_call(
        client,
        model=model,
        messages=[{"role": "user", "content": "Reply with exactly the two letters: OK"}],
        max_completion_tokens=4000,
    )
    dt = time.time() - t0
    content = resp.choices[0].message.content
    finish = resp.choices[0].finish_reason
    usage = usage_dict(resp.usage)
    print(f"  latency: {dt:.2f}s  finish_reason: {finish}")
    print(f"  content: {content!r}")
    print(f"  usage: {usage}")
    return {"latency_s": round(dt, 3), "finish_reason": finish, "content": content, "usage": usage}


def probe_structured(client: OpenAI, model: str, effort: str | None) -> dict[str, Any]:
    label = effort or "default"
    print(f"\n=== Probe 2/3: structured output, reasoning_effort={label} ===")
    results = []
    for i, pair in enumerate(PAIRS):
        kwargs: dict[str, Any] = dict(
            model=model,
            messages=[
                {"role": "system", "content": JUDGE_RUBRIC},
                {"role": "user", "content": format_pair(pair)},
            ],
            response_format={
                "type": "json_schema",
                "json_schema": {"name": "match_judgment", "strict": True, "schema": JUDGE_SCHEMA},
            },
            max_completion_tokens=4000,
        )
        if effort:
            kwargs["reasoning_effort"] = effort
        t0 = time.time()
        resp = _try_call(client, **kwargs)
        dt = time.time() - t0
        raw = resp.choices[0].message.content
        usage = usage_dict(resp.usage)
        try:
            parsed = json.loads(raw) if raw else None
            valid = parsed is not None and isinstance(parsed, dict)
        except Exception:
            parsed = None
            valid = False
        agree = (
            parsed is not None
            and parsed.get("match_type") == pair["expected_label"]
        )
        results.append(
            {
                "pair_index": i,
                "latency_s": round(dt, 3),
                "json_valid": valid,
                "expected": pair["expected_label"],
                "predicted_match_type": parsed.get("match_type") if parsed else None,
                "predicted_is_match": parsed.get("is_match") if parsed else None,
                "confidence": parsed.get("confidence") if parsed else None,
                "agrees_with_expected": agree,
                "usage": usage,
                "finish_reason": resp.choices[0].finish_reason,
            }
        )
        print(
            f"  pair {i}: {dt:.2f}s "
            f"valid={valid} "
            f"agree={agree} "
            f"got={results[-1]['predicted_match_type']} "
            f"expected={pair['expected_label']} "
            f"reasoning_tok={usage.get('reasoning_tokens')}"
        )
    latencies = [r["latency_s"] for r in results]
    valid_rate = sum(r["json_valid"] for r in results) / len(results)
    agree_rate = sum(r["agrees_with_expected"] for r in results) / len(results)
    total_prompt = sum(r["usage"]["prompt_tokens"] or 0 for r in results)
    total_completion = sum(r["usage"]["completion_tokens"] or 0 for r in results)
    total_reasoning = sum(r["usage"].get("reasoning_tokens") or 0 for r in results)
    aggregate = {
        "n": len(results),
        "latency_p50": round(statistics.median(latencies), 3),
        "latency_p90": round(sorted(latencies)[int(0.9 * len(latencies)) - 1], 3),
        "json_valid_rate": round(valid_rate, 3),
        "label_agreement_rate": round(agree_rate, 3),
        "tokens_prompt_total": total_prompt,
        "tokens_completion_total": total_completion,
        "tokens_reasoning_total": total_reasoning,
        "tokens_per_pair_avg_prompt": round(total_prompt / len(results), 1),
        "tokens_per_pair_avg_completion": round(total_completion / len(results), 1),
    }
    print(f"  AGGREGATE: {aggregate}")
    return {"effort": label, "aggregate": aggregate, "rows": results}


def probe_batched(client: OpenAI, model: str, effort: str | None = "minimal") -> dict[str, Any]:
    print("\n=== Probe 4: per-prompt candidate batching (1 A + 5 B) ===")
    # Build one A from PAIRS[0] (Russell Stover), and 5 candidates:
    # the true match plus 4 distractors.
    query_a = PAIRS[0]
    candidates = [
        # idx 0: the true match
        {"name_B": PAIRS[0]["name_B"], "brand_B": PAIRS[0]["brand_B"], "size_B": PAIRS[0]["size_B"], "category_B": PAIRS[0]["category_B"]},
        # idx 1: same brand, wrong product
        {"name_B": "Russell Stover Pecan Delights Candy", "brand_B": "Russell Stover", "size_B": "9 ounce", "category_B": "Grocery > Candy > Boxed Chocolates"},
        # idx 2: wrong brand, similar product
        {"name_B": "Whitman's Sugar Free Assorted Chocolate Gift Box", "brand_B": "Whitman's", "size_B": "6.5 ounce", "category_B": "Grocery > Candy > Boxed Chocolates"},
        # idx 3: cross-domain distractor
        {"name_B": "Wegmans Honey Ham, Thin Shaved Chipped", "brand_B": "Wegmans", "size_B": "", "category_B": "More Departments > Deli > Ham"},
        # idx 4: M&M's
        {"name_B": "M&M'S Milk Chocolate Christmas Candy Bag", "brand_B": "M&M'S", "size_B": "10 ounce", "category_B": "Grocery > Candy > Seasonal & Holiday Candy"},
    ]
    body = (
        f"Store A product:\n"
        f"  name: {query_a['name_A']}\n"
        f"  brand: {query_a['brand_A']}\n"
        f"  size: {query_a['size_A']}\n"
        f"  category: {query_a['category_A']}\n\n"
        f"Candidate Store B products:\n"
    )
    for i, b in enumerate(candidates):
        body += (
            f"[{i}] name: {b['name_B']} | brand: {b['brand_B']} | "
            f"size: {b['size_B']} | category: {b['category_B']}\n"
        )
    body += (
        "\nPick the single best matching candidate index, or null if none match. "
        "Apply the same matching rules as before."
    )
    kwargs: dict[str, Any] = dict(
        model=model,
        messages=[
            {"role": "system", "content": JUDGE_RUBRIC},
            {"role": "user", "content": body},
        ],
        response_format={
            "type": "json_schema",
            "json_schema": {"name": "candidate_rank", "strict": True, "schema": RANK_SCHEMA},
        },
        max_completion_tokens=4000,
    )
    if effort:
        kwargs["reasoning_effort"] = effort
    t0 = time.time()
    resp = _try_call(client, **kwargs)
    dt = time.time() - t0
    raw = resp.choices[0].message.content
    usage = usage_dict(resp.usage)
    try:
        parsed = json.loads(raw) if raw else None
    except Exception:
        parsed = None
    print(f"  K=5 latency: {dt:.2f}s reasoning_tok={usage.get('reasoning_tokens')}")
    print(f"  parsed: {parsed}")
    print(f"  usage: {usage}")
    return {
        "K": 5,
        "latency_s": round(dt, 3),
        "usage": usage,
        "parsed": parsed,
        "expected_best_index": 0,
    }


def probe_batch_api_syntax(client: OpenAI, model: str) -> dict[str, Any]:
    """Submit a tiny 2-row Batch job to confirm the surface accepts it.
    We do NOT wait for completion (24h SLA). We cancel right after creation."""
    print("\n=== Probe 5: Batch API submission syntax ===")
    tmp = OUTPUT_DIR / "smoke_batch_input.jsonl"
    lines = []
    for i, pair in enumerate(PAIRS[:2]):
        lines.append(
            json.dumps(
                {
                    "custom_id": f"smoke-{i}",
                    "method": "POST",
                    "url": "/v1/chat/completions",
                    "body": {
                        "model": model,
                        "messages": [
                            {"role": "system", "content": JUDGE_RUBRIC},
                            {"role": "user", "content": format_pair(pair)},
                        ],
                        "response_format": {
                            "type": "json_schema",
                            "json_schema": {
                                "name": "match_judgment",
                                "strict": True,
                                "schema": JUDGE_SCHEMA,
                            },
                        },
                        "max_completion_tokens": 4000,
                    },
                }
            )
        )
    tmp.write_text("\n".join(lines), encoding="utf-8")
    try:
        with tmp.open("rb") as fh:
            file_obj = client.files.create(file=fh, purpose="batch")
        batch_obj = client.batches.create(
            input_file_id=file_obj.id,
            endpoint="/v1/chat/completions",
            completion_window="24h",
        )
        out = {
            "supported": True,
            "input_file_id": file_obj.id,
            "batch_id": batch_obj.id,
            "status": batch_obj.status,
        }
        # Cancel immediately so we don't accidentally consume hours of SLA.
        try:
            client.batches.cancel(batch_obj.id)
            out["cancel_attempted"] = True
        except Exception as e:
            out["cancel_attempted"] = False
            out["cancel_error"] = str(e)
        print(f"  Batch submission OK: batch_id={batch_obj.id} status={batch_obj.status}")
        return out
    except Exception as e:
        print(f"  Batch submission FAILED: {e}")
        return {"supported": False, "error": str(e)}


def build_markdown(summary: dict[str, Any]) -> str:
    lines = ["# LLM Smoke Test", "", f"Model deployment: `{summary['model']}`", f"Endpoint: `{summary['endpoint']}`", ""]
    p1 = summary["connectivity"]
    lines.append("## 1. Connectivity")
    lines.append(f"- Latency: {p1['latency_s']}s")
    lines.append(f"- finish_reason: `{p1['finish_reason']}`")
    lines.append(f"- content: `{p1['content']!r}`")
    lines.append(f"- usage: `{p1['usage']}`")
    for key in ("default_effort", "minimal_effort"):
        if key not in summary:
            continue
        probe = summary[key]
        agg = probe["aggregate"]
        lines.append(f"\n## 2/3. Single-pair structured output (effort={probe['effort']})")
        lines.append(f"- n: {agg['n']}")
        lines.append(f"- latency p50/p90: {agg['latency_p50']}s / {agg['latency_p90']}s")
        lines.append(f"- JSON validity: {agg['json_valid_rate']:.0%}")
        lines.append(f"- Label agreement (vs my expected): {agg['label_agreement_rate']:.0%}")
        lines.append(
            f"- Avg tokens per pair (prompt / completion): "
            f"{agg['tokens_per_pair_avg_prompt']} / {agg['tokens_per_pair_avg_completion']}"
        )
        lines.append(f"- Total reasoning tokens across n pairs: {agg['tokens_reasoning_total']}")
    if "batched" in summary:
        b = summary["batched"]
        lines.append("\n## 4. Per-prompt candidate batching")
        lines.append(f"- K (candidates per prompt): {b['K']}")
        lines.append(f"- Latency: {b['latency_s']}s")
        lines.append(f"- Usage: {b['usage']}")
        lines.append(f"- Parsed: `{b['parsed']}`")
        lines.append(f"- Expected best index: {b['expected_best_index']}")
    if "batch_api" in summary:
        bk = summary["batch_api"]
        lines.append("\n## 5. Batch API submission")
        if bk.get("supported"):
            lines.append(f"- Submission accepted. batch_id=`{bk['batch_id']}` status=`{bk['status']}`")
            lines.append(f"- Cancel attempted: {bk.get('cancel_attempted')}")
        else:
            lines.append(f"- Submission FAILED: `{bk.get('error')}`")
    return "\n".join(lines)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    client, model = make_client()
    print(f"Endpoint deployment: {model}")
    summary: dict[str, Any] = {
        "endpoint": yaml.safe_load(CREDS_PATH.read_text())["openai"]["endpoint"],
        "model": model,
    }
    summary["connectivity"] = probe_connectivity(client, model)
    summary["default_effort"] = probe_structured(client, model, effort=None)
    summary["minimal_effort"] = probe_structured(client, model, effort="minimal")
    summary["batched"] = probe_batched(client, model, effort="minimal")
    summary["batch_api"] = probe_batch_api_syntax(client, model)

    SUMMARY_JSON.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    SUMMARY_MD.write_text(build_markdown(summary), encoding="utf-8")
    print(f"\nWrote {SUMMARY_JSON}")
    print(f"Wrote {SUMMARY_MD}")


if __name__ == "__main__":
    main()
