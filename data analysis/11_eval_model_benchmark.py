"""Benchmark gpt-5.4-nano on the labeled eval set.

Runs single-pair structured-output judgments on every pair in
`eval_candidates.csv` joined with `eval_labels.csv`. Computes:

- Overall and per-stratum accuracy / agreement on match_type
- Binary classification metrics on is_match: precision, recall, F1
- Score-to-precision calibration on model confidence
- Latency p50/p90/p99, token totals, $/pair cost estimate
- Per-stratum confusion (what does the model get wrong, where)

Outputs:
- outputs/eval_results.csv          raw per-pair model output joined with labels
- outputs/eval_summary.md / .json   aggregate metrics + calibration table
"""
from __future__ import annotations

import json
import statistics
import time
from pathlib import Path
from typing import Any

import pandas as pd
import yaml
from openai import OpenAI, RateLimitError, APIError

ROOT = Path(__file__).resolve().parents[1]
CREDS_PATH = ROOT / "openai_creds.yaml"
OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"
EVAL_CANDIDATES = OUTPUT_DIR / "eval_candidates.csv"
EVAL_LABELS = OUTPUT_DIR / "eval_labels.csv"
RESULTS_CSV = OUTPUT_DIR / "eval_results.csv"
SUMMARY_MD = OUTPUT_DIR / "eval_summary.md"
SUMMARY_JSON = OUTPUT_DIR / "eval_summary.json"

# Public-pricing-ballpark estimate for gpt-5-nano-class.  Azure pricing may
# differ; this is purely an order-of-magnitude cost lens.
PRICE_PROMPT_PER_M = 0.05
PRICE_COMPLETION_PER_M = 0.40


JUDGE_RUBRIC = (
    "You are a grocery product matching judge. Given a pair of products from two "
    "different grocery stores, decide whether a shopper would treat them as "
    "essentially the same product for pricing purposes.\n"
    "Match types:\n"
    "- exact_national_brand: same brand (or curated alias), same specific product, "
    "compatible per-unit size, same form. Different flavors, different sub-product "
    "lines, conventional vs organic, and substantially different sizes are NOT exact.\n"
    "- private_label_equivalent: both are store/private-label brands "
    "(Great Value, Marketside, Wegmans, Equate, etc.), same specific product, "
    "same size, same form. Different flavors or different products do NOT qualify.\n"
    "- fresh_equivalent: produce, meat, deli, bakery equivalents with compatible "
    "size and form.\n"
    "- no_match: any meaningful difference in flavor, formulation, size, form, "
    "sub-product line, organic vs conventional, or category.\n"
    "Be strict. When uncertain, prefer no_match. Multipacks of the same SKU are "
    "an acceptable exact match (per-unit equivalence).\n"
    "Return only the JSON object."
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


def format_pair(row: pd.Series) -> str:
    return (
        "Store A product:\n"
        f"  name: {row['name_A']}\n"
        f"  brand: {row['brand_A']}\n"
        f"  size: {row['size_A']}\n"
        f"  category: {row['category_A']}\n"
        "Store B product:\n"
        f"  name: {row['name_B']}\n"
        f"  brand: {row['brand_B']}\n"
        f"  size: {row['size_B']}\n"
        f"  category: {row['category_B']}\n"
    )


def make_client() -> tuple[OpenAI, str]:
    creds = yaml.safe_load(CREDS_PATH.read_text())["openai"]
    return (
        OpenAI(
            base_url=creds["endpoint"],
            api_key=creds["api_key"],
            max_retries=6,  # SDK default is 2 - we need more for Azure 429s
            timeout=60.0,
        ),
        creds["deployment_name"],
    )


def _call_with_backoff(client: OpenAI, **kwargs) -> Any:
    """Call chat.completions with explicit exponential backoff on 429.
    The SDK already retries, but Azure 429s sometimes outlast the default count."""
    delays = [1.0, 2.0, 4.0, 8.0, 16.0, 32.0]
    last_err: Exception | None = None
    for attempt, delay in enumerate([0.0] + delays):
        if delay:
            time.sleep(delay)
        try:
            return client.chat.completions.create(**kwargs)
        except RateLimitError as e:
            last_err = e
            continue
        except APIError as e:
            # Some 429s come through as generic APIError on Azure
            if "429" in str(e) or "too_many_requests" in str(e).lower():
                last_err = e
                continue
            raise
    raise last_err if last_err else RuntimeError("retry loop exited without error")


def usage_dict(usage) -> dict[str, Any]:
    d = {
        "prompt_tokens": usage.prompt_tokens,
        "completion_tokens": usage.completion_tokens,
        "total_tokens": usage.total_tokens,
    }
    details = getattr(usage, "completion_tokens_details", None)
    if details is not None:
        d["reasoning_tokens"] = getattr(details, "reasoning_tokens", 0) or 0
    prompt_details = getattr(usage, "prompt_tokens_details", None)
    if prompt_details is not None:
        d["cached_prompt_tokens"] = getattr(prompt_details, "cached_tokens", 0) or 0
    return d


def judge_pair(client: OpenAI, model: str, row: pd.Series) -> dict[str, Any]:
    t0 = time.time()
    resp = _call_with_backoff(
        client,
        model=model,
        messages=[
            {"role": "system", "content": JUDGE_RUBRIC},
            {"role": "user", "content": format_pair(row)},
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
    usage = usage_dict(resp.usage)
    return {
        "latency_s": round(dt, 3),
        "json_valid": parsed is not None,
        "predicted_match_type": parsed.get("match_type") if parsed else None,
        "predicted_is_match": parsed.get("is_match") if parsed else None,
        "predicted_confidence": parsed.get("confidence") if parsed else None,
        "predicted_reasons": ";".join(parsed.get("reason_codes", [])) if parsed else "",
        "usage_prompt_tokens": usage["prompt_tokens"],
        "usage_completion_tokens": usage["completion_tokens"],
        "usage_reasoning_tokens": usage.get("reasoning_tokens", 0),
    }


def main() -> None:
    cands = pd.read_csv(EVAL_CANDIDATES)
    labels = pd.read_csv(EVAL_LABELS)
    df = cands.merge(labels, on="pair_id")
    print(f"Eval set size: {len(df)}")

    client, model = make_client()
    print(f"Model: {model}")

    rows: list[dict[str, Any]] = []
    per_call_sleep = 0.4  # be polite to the rate limit
    for idx, r in df.iterrows():
        try:
            res = judge_pair(client, model, r)
        except Exception as e:
            res = {
                "latency_s": None,
                "json_valid": False,
                "predicted_match_type": None,
                "predicted_is_match": None,
                "predicted_confidence": None,
                "predicted_reasons": f"ERROR:{e}",
                "usage_prompt_tokens": 0,
                "usage_completion_tokens": 0,
                "usage_reasoning_tokens": 0,
            }
        merged = {**r.to_dict(), **res}
        rows.append(merged)
        if (idx + 1) % 10 == 0 or idx == len(df) - 1:
            print(f"  [{idx + 1}/{len(df)}] last: pair={r['pair_id']} "
                  f"pred={res['predicted_match_type']} truth={r['label_match_type']} "
                  f"lat={res['latency_s']}s")
        time.sleep(per_call_sleep)

    out = pd.DataFrame(rows)
    out.to_csv(RESULTS_CSV, index=False)

    # ---- Aggregate metrics ----
    valid = out[out["json_valid"]]
    n = len(out)
    valid_rate = len(valid) / n if n else 0.0
    match_type_correct = (valid["predicted_match_type"] == valid["label_match_type"]).sum()
    is_match_correct = (valid["predicted_is_match"] == valid["label_is_match"]).sum()
    # Binary precision/recall on is_match
    tp = int(((valid["predicted_is_match"]) & (valid["label_is_match"])).sum())
    fp = int(((valid["predicted_is_match"]) & (~valid["label_is_match"])).sum())
    fn = int(((~valid["predicted_is_match"]) & (valid["label_is_match"])).sum())
    tn = int(((~valid["predicted_is_match"]) & (~valid["label_is_match"])).sum())
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

    # Latency
    latencies = [r["latency_s"] for r in rows if r["latency_s"] is not None]
    latency_p50 = round(statistics.median(latencies), 3) if latencies else None
    latency_p90 = round(sorted(latencies)[int(0.9 * len(latencies)) - 1], 3) if latencies else None
    latency_p99 = round(sorted(latencies)[max(int(0.99 * len(latencies)) - 1, 0)], 3) if latencies else None

    # Token / cost
    total_prompt = int(out["usage_prompt_tokens"].sum())
    total_completion = int(out["usage_completion_tokens"].sum())
    total_reasoning = int(out["usage_reasoning_tokens"].sum())
    cost = (
        total_prompt / 1_000_000 * PRICE_PROMPT_PER_M
        + total_completion / 1_000_000 * PRICE_COMPLETION_PER_M
    )

    # Per-stratum metrics
    per_stratum: list[dict[str, Any]] = []
    for stratum, sub in valid.groupby("stratum"):
        match_correct = int((sub["predicted_match_type"] == sub["label_match_type"]).sum())
        tp_s = int(((sub["predicted_is_match"]) & (sub["label_is_match"])).sum())
        fp_s = int(((sub["predicted_is_match"]) & (~sub["label_is_match"])).sum())
        fn_s = int(((~sub["predicted_is_match"]) & (sub["label_is_match"])).sum())
        prec_s = tp_s / (tp_s + fp_s) if (tp_s + fp_s) else None
        rec_s = tp_s / (tp_s + fn_s) if (tp_s + fn_s) else None
        per_stratum.append(
            {
                "stratum": stratum,
                "n": len(sub),
                "match_type_accuracy": round(match_correct / len(sub), 3),
                "label_positive_rate": round(float(sub["label_is_match"].mean()), 3),
                "model_positive_rate": round(float(sub["predicted_is_match"].mean()), 3),
                "precision": round(prec_s, 3) if prec_s is not None else None,
                "recall": round(rec_s, 3) if rec_s is not None else None,
            }
        )

    # Calibration: bin model confidence, compute empirical positive precision
    calib = []
    bins = [(0.0, 0.5), (0.5, 0.7), (0.7, 0.85), (0.85, 0.95), (0.95, 1.01)]
    pos = valid[valid["predicted_is_match"]]
    for lo, hi in bins:
        bucket = pos[(pos["predicted_confidence"] >= lo) & (pos["predicted_confidence"] < hi)]
        if bucket.empty:
            calib.append({"band": f"{lo:.2f}-{hi:.2f}", "n": 0, "empirical_precision": None})
            continue
        emp = float(bucket["label_is_match"].mean())
        calib.append({"band": f"{lo:.2f}-{hi:.2f}", "n": int(len(bucket)), "empirical_precision": round(emp, 3)})

    # Confusion matrix on match_type
    confusion = (
        valid.groupby(["label_match_type", "predicted_match_type"]).size().unstack(fill_value=0).to_dict()
    )

    summary = {
        "n": int(n),
        "json_valid_rate": round(valid_rate, 4),
        "match_type_accuracy": round(match_type_correct / len(valid), 4) if len(valid) else None,
        "is_match_accuracy": round(is_match_correct / len(valid), 4) if len(valid) else None,
        "binary": {
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
        },
        "latency": {
            "p50_s": latency_p50, "p90_s": latency_p90, "p99_s": latency_p99,
            "n": len(latencies),
        },
        "tokens": {
            "prompt_total": total_prompt,
            "completion_total": total_completion,
            "reasoning_total": total_reasoning,
            "avg_prompt_per_pair": round(total_prompt / n, 1) if n else 0,
            "avg_completion_per_pair": round(total_completion / n, 1) if n else 0,
        },
        "cost_estimate_usd": {
            "total_for_eval_set": round(cost, 6),
            "per_pair": round(cost / n, 6) if n else 0,
            "projected_10k_pairs": round(cost / n * 10_000, 4) if n else 0,
            "pricing_basis": f"${PRICE_PROMPT_PER_M}/M prompt + ${PRICE_COMPLETION_PER_M}/M completion (public gpt-5-nano ballpark)",
        },
        "per_stratum": per_stratum,
        "confidence_calibration": calib,
        "confusion_matrix": confusion,
    }
    SUMMARY_JSON.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    # Markdown
    md = ["# Model Benchmark on Eval Set", "",
          f"Model: `{model}`", f"Eval set: {n} pairs", ""]
    md.append("## Aggregate")
    md.append(f"- JSON validity: {summary['json_valid_rate']:.1%}")
    md.append(f"- Match-type accuracy: {summary['match_type_accuracy']:.1%}")
    md.append(f"- is_match accuracy: {summary['is_match_accuracy']:.1%}")
    md.append(f"- Binary precision / recall / F1: "
              f"{summary['binary']['precision']:.3f} / "
              f"{summary['binary']['recall']:.3f} / {summary['binary']['f1']:.3f}")
    md.append(f"- Confusion (tp / fp / fn / tn): "
              f"{summary['binary']['tp']} / {summary['binary']['fp']} / "
              f"{summary['binary']['fn']} / {summary['binary']['tn']}")
    md.append("")
    md.append("## Latency")
    md.append(f"- p50 / p90 / p99: {summary['latency']['p50_s']}s / {summary['latency']['p90_s']}s / {summary['latency']['p99_s']}s")
    md.append("")
    md.append("## Tokens & cost")
    md.append(f"- Avg prompt / completion per pair: {summary['tokens']['avg_prompt_per_pair']} / {summary['tokens']['avg_completion_per_pair']}")
    md.append(f"- Total reasoning tokens: {summary['tokens']['reasoning_total']}")
    md.append(f"- Eval set cost: ${summary['cost_estimate_usd']['total_for_eval_set']}")
    md.append(f"- Projected 10k pairs cost: ${summary['cost_estimate_usd']['projected_10k_pairs']}")
    md.append("")
    md.append("## Per stratum")
    md.append("| stratum | n | label_pos_rate | model_pos_rate | precision | recall | match_type_acc |")
    md.append("|---|---|---|---|---|---|---|")
    for p in per_stratum:
        md.append(
            f"| {p['stratum']} | {p['n']} | {p['label_positive_rate']} | {p['model_positive_rate']} | "
            f"{p['precision']} | {p['recall']} | {p['match_type_accuracy']} |"
        )
    md.append("")
    md.append("## Confidence calibration (predicted positives)")
    md.append("| confidence band | n | empirical precision |")
    md.append("|---|---|---|")
    for c in calib:
        md.append(f"| {c['band']} | {c['n']} | {c['empirical_precision']} |")
    md.append("")
    md.append("## Match-type confusion")
    md.append("```")
    md.append(json.dumps(confusion, indent=2))
    md.append("```")

    SUMMARY_MD.write_text("\n".join(md), encoding="utf-8")
    print(f"\nWrote {RESULTS_CSV}")
    print(f"Wrote {SUMMARY_JSON}")
    print(f"Wrote {SUMMARY_MD}")


if __name__ == "__main__":
    main()
