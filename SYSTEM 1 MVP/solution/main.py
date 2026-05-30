"""End-to-end driver.

Stages:
  1. load A and B  (cached as parquet)
  2. retrieve T1 + T3 + T5  (cached as parquet)
  3. score deterministic features + routing
  4. judge routed cases with the LLM (cached as JSONL)
  5. select one B per A from accepted candidates
  6. validate against the eval set

Usage:
  python -m solution.main --mode rules        # no LLM, deterministic only
  python -m solution.main --mode full         # full pipeline including LLM
  python -m solution.main --mode full --a-limit 5000
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd

# Allow running both as module and as script.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from solution import judge as judge_mod
    from solution import rag as rag_mod
    from solution import retrieve as retrieve_mod
    from solution import score as score_mod
    from solution import select as select_mod
    from solution import validate as validate_mod
    from solution.load import load_store
else:
    from . import judge as judge_mod
    from . import rag as rag_mod
    from . import retrieve as retrieve_mod
    from . import score as score_mod
    from . import select as select_mod
    from . import validate as validate_mod
    from .load import load_store


ROOT = Path(__file__).resolve().parents[2]
SYSTEM_ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = ROOT / "dataset"
CACHE_DIR = SYSTEM_ROOT / "cache"
OUTPUTS_DIR = SYSTEM_ROOT / "outputs"
BRIDGE_CSV = ROOT / "data analysis" / "outputs" / "category_bridge_a_to_b.csv"
EVAL_CANDIDATES = ROOT / "data analysis" / "outputs" / "eval_candidates.csv"
EVAL_LABELS = ROOT / "data analysis" / "outputs" / "eval_labels.csv"
EVAL_RESULTS = ROOT / "data analysis" / "outputs" / "eval_results.csv"
CREDS_PATH = ROOT / "openai_creds.yaml"


def load_or_cache(label: str, nrows: int | None) -> pd.DataFrame:
    """Load a store from CSV with a parquet cache layer."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    suffix = f"_n{nrows}" if nrows else "_full"
    cache_path = CACHE_DIR / f"store_{label}{suffix}.parquet"
    if cache_path.exists():
        print(f"  loading {label} from cache {cache_path.name}")
        return pd.read_parquet(cache_path)
    print(f"  loading {label} from CSV (and caching)")
    df = load_store(label, nrows=nrows)
    # Parquet can't handle list columns containing nulls without conversion.
    if "tags_list" in df.columns:
        df["tags_list"] = df["tags_list"].apply(lambda x: x if isinstance(x, list) else [])
    df.to_parquet(cache_path, index=False)
    return df


def cached_retrieve(a: pd.DataFrame, b: pd.DataFrame, a_limit_tag: str) -> pd.DataFrame:
    """Cache the candidate dataframe to parquet so iteration is fast."""
    cache_path = CACHE_DIR / f"candidates_{a_limit_tag}.parquet"
    if cache_path.exists():
        print(f"  loading candidates from cache {cache_path.name}")
        return pd.read_parquet(cache_path)

    print("  T1: strict brand+size blocks ...")
    t0 = time.time()
    t1 = retrieve_mod.t1_strict_blocks(a, b, top_per_a=3)
    print(f"    T1: {len(t1)} candidates in {time.time()-t0:.0f}s")

    print("  T3: TF-IDF top-K ...")
    t0 = time.time()
    t3 = retrieve_mod.t3_tfidf_topk(a, b, k=20, cosine_floor=0.4)
    print(f"    T3: {len(t3)} candidates in {time.time()-t0:.0f}s")

    print("  T5: private-label + category bridge ...")
    t0 = time.time()
    t5 = retrieve_mod.t5_private_label(a, b, BRIDGE_CSV, top_per_a=5)
    print(f"    T5: {len(t5)} candidates in {time.time()-t0:.0f}s")

    union = retrieve_mod.union_candidates(t1, t3, t5)
    print(f"  union: {len(union)} unique candidate pairs")
    df = retrieve_mod.to_dataframe(union)
    df.to_parquet(cache_path, index=False)
    return df


def run(mode: str, a_limit: int | None, llm_workers: int) -> None:
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[stage 1] Loading datasets (a_limit={a_limit}) ...")
    a = load_or_cache("A", a_limit)
    b = load_or_cache("B", None)
    print(f"  A={len(a)} rows, B={len(b)} rows")

    a_index = score_mod.build_lookup_index(a)
    b_index = score_mod.build_lookup_index(b)

    print("[stage 2] Generating candidates ...")
    a_tag = f"a{a_limit}" if a_limit else "afull"
    candidates_df = cached_retrieve(a, b, a_tag)

    print("[stage 3] Scoring deterministic features ...")
    t0 = time.time()
    scored = score_mod.score_candidates(candidates_df, a_index, b_index)
    print(f"  scored {len(scored)} candidates in {time.time()-t0:.0f}s")

    auto = scored[scored["route"] == "auto_accept"]
    routed = scored[scored["route"] == "route_to_llm"]
    drop = scored[scored["route"] == "drop"]
    print(f"  routing: auto_accept={len(auto)}  route_to_llm={len(routed)}  drop={len(drop)}")

    # Save full scored table for audit.
    scored.to_csv(OUTPUTS_DIR / "match_candidates_scored.csv", index=False)
    drop.to_csv(OUTPUTS_DIR / "rejected_borderline.csv", index=False)

    if mode == "rules":
        print("[stage 4] Skipping LLM (rules mode)")
        accepted = auto.copy()
        accepted["accept_source"] = "rules"
        accepted["llm_confidence"] = None
    else:
        print(f"[stage 4] LLM judge on {len(routed)} routed candidates ...")
        if len(routed) == 0:
            llm_accepted = pd.DataFrame(columns=auto.columns.tolist() + ["llm_confidence", "accept_source"])
        else:
            llm_accepted = run_llm(routed, a_index, b_index, llm_workers)
        auto = auto.copy()
        auto["llm_confidence"] = None
        auto["accept_source"] = "rules"
        accepted = pd.concat([auto, llm_accepted], ignore_index=True)

    print(f"[stage 5] Selecting one B per A (over {len(accepted)} accepted candidates) ...")
    final = select_mod.select_one_b_per_a(accepted)
    print(f"  final matches: {len(final)} rows")

    # Deliverable.
    deliverable = final[["item_id_a", "item_id_b"]].copy()
    deliverable.columns = ["item_id_A", "item_id_B"]
    deliverable.to_csv(OUTPUTS_DIR / "matches.csv", index=False)
    # Also write the deliverable to the system root so it's prominent
    # next to the README rather than buried under outputs/.
    deliverable.to_csv(SYSTEM_ROOT / "matches.csv", index=False)
    print(f"  wrote {OUTPUTS_DIR / 'matches.csv'} ({len(deliverable)} rows)")
    print(f"  wrote {SYSTEM_ROOT / 'matches.csv'} (deliverable copy)")

    final.to_csv(OUTPUTS_DIR / "matches_with_features.csv", index=False)

    # Audit sample of 50.
    sample = final.sample(min(50, len(final)), random_state=42)
    sample.to_csv(OUTPUTS_DIR / "match_audit_sample.csv", index=False)

    print("[stage 6] Validating against eval set ...")
    report = validate_mod.validate_against_eval(
        OUTPUTS_DIR / "matches.csv", EVAL_CANDIDATES, EVAL_LABELS
    )
    validate_mod.write_report(report, OUTPUTS_DIR / "validation_report.md")
    if "overall" in report:
        o = report["overall"]
        print(
            f"  eval-set: n={o['n']} shipped={o['shipped']} "
            f"P={o['precision']} R={o['recall']} F1={o['f1']}"
        )

    print("Done.")


def run_llm(
    routed: pd.DataFrame, a_index: dict, b_index: dict, workers: int
) -> pd.DataFrame:
    """Group routed candidates by A id (K up to 5 per call), build RAG
    snippets per group, call the judge, fold accepted picks back into a
    DataFrame matching `routed`'s columns."""
    client, model = judge_mod.load_creds(CREDS_PATH)
    cache = judge_mod.JudgmentCache(OUTPUTS_DIR / "llm_judgments.jsonl")
    rag_store = rag_mod.build_store(EVAL_RESULTS)

    # Group routed candidates by item_id_a; cap each group at top-5 by
    # final_score (highest first).
    K = 5
    groups: list[tuple[str, pd.DataFrame]] = []
    for a_id, sub in routed.groupby("item_id_a"):
        sub_sorted = sub.sort_values("final_score", ascending=False).head(K)
        groups.append((a_id, sub_sorted))

    # Build judgment batches.
    batches: list[tuple] = []
    group_index: list[pd.DataFrame] = []
    for a_id, sub in groups:
        a_row = a_index.get(a_id)
        if a_row is None:
            continue
        b_ids = sub["item_id_b"].tolist()
        b_rows = [b_index.get(bid) or {} for bid in b_ids]
        query = (
            f"A: {a_row.get('name')} brand {a_row.get('brand_raw')} size {a_row.get('size_text')} | "
            f"B candidates: {' / '.join((b_row or {}).get('name') or '' for b_row in b_rows)}"
        )
        rag_snippets = rag_store.search(query, k=3)
        batches.append((a_id, a_row, b_ids, b_rows, rag_snippets))
        group_index.append(sub.reset_index(drop=True))

    print(f"  {len(batches)} LLM batches (K={K})")
    results = judge_mod.judge_many(client, model, batches, cache, workers=workers)

    # Fold back accepted picks.
    rows = []
    n_accepted = 0
    n_calls = 0
    for r, sub in zip(results, group_index):
        if r is None:
            continue
        n_calls += 1
        if not r.get("is_match"):
            continue
        if r.get("confidence", 0.0) < 0.85:
            continue
        idx = r.get("best_candidate_index")
        if idx is None or idx < 0 or idx >= len(sub):
            continue
        row = sub.iloc[idx].to_dict()
        row["llm_confidence"] = r.get("confidence")
        row["accept_source"] = "llm"
        rows.append(row)
        n_accepted += 1
    print(f"  LLM accepted {n_accepted} / {n_calls} batches (confidence>=0.85)")
    return pd.DataFrame(rows)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["rules", "full"], default="rules")
    p.add_argument("--a-limit", type=int, default=None, help="cap A rows for dev runs")
    p.add_argument("--llm-workers", type=int, default=5)
    args = p.parse_args()
    run(args.mode, args.a_limit, args.llm_workers)


if __name__ == "__main__":
    main()
