"""SYSTEM 2 RAG end-to-end driver.

Stages (mirror System 1's structure):
  1. Load A + B (read SYSTEM 1's parquet cache; no re-parsing).
  2. Load catalog embeddings (per-store .npy from cache/).
  3. Bootstrap + load knowledge base (idempotent).
  4. Build VectorStore over B; semantic retrieve top-K per A row.
  5. Run System 1's T1/T3/T5 retrievers in parallel (deterministic).
  6. Union + score deterministically (reuses System 1's score.py).
  7. Route: auto-accept high-confidence; route mid-confidence to RAG judge.
  8. Build per-batch RAG context, call gpt-5.4-nano (concurrent workers).
  9. Select one B per A; write matches.csv + audit.

Modes:
  --mode eval     : run only on the 97-pair labeled eval set (cheap gate)
  --mode full     : run on the full 233k A catalog (the deliverable)
  --a-limit N     : cap A rows for dev runs
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from openai import OpenAI

# Allow running both as module and as script.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from solution.config import azure_llm_config, openai_embedding_config
    from solution.embed import (
        EmbeddingBank,
        OpenAIEmbedder,
        load_store_embeddings,
    )
    from solution.knowledge import (
        KnowledgeRetriever,
        bootstrap_all,
    )
    from solution.store import NumpyVectorStore
    from solution.retrieve import (
        SemanticRetriever,
        StrictBlockRetriever,
        TfidfRetriever,
        PrivateLabelRetriever,
        union_candidates,
        to_dataframe,
    )
    from solution.context import RAGContextBuilder
    from solution.judge import JudgeRequest, JudgmentCache, RAGJudge
    from solution.judge.rag_judge import DEFAULT_CACHE_VERSION
else:
    from ..config import azure_llm_config, openai_embedding_config
    from ..embed import EmbeddingBank, OpenAIEmbedder, load_store_embeddings
    from ..knowledge import KnowledgeRetriever, bootstrap_all
    from ..store import NumpyVectorStore
    from ..retrieve import (
        SemanticRetriever,
        StrictBlockRetriever,
        TfidfRetriever,
        PrivateLabelRetriever,
        union_candidates,
        to_dataframe,
    )
    from ..context import RAGContextBuilder
    from ..judge import JudgeRequest, JudgmentCache, RAGJudge
    from ..judge.rag_judge import DEFAULT_CACHE_VERSION


REPO_ROOT = Path(__file__).resolve().parents[3]
SYSTEM1_ROOT = REPO_ROOT / "SYSTEM 1 MVP"
SYSTEM1_CACHE = SYSTEM1_ROOT / "cache"
SYSTEM2_ROOT = Path(__file__).resolve().parents[2]
SYSTEM2_CACHE = SYSTEM2_ROOT / "cache"
SYSTEM2_OUTPUTS = SYSTEM2_ROOT / "outputs"
KNOWLEDGE_DIR = SYSTEM2_ROOT / "knowledge"
EDA_OUTPUTS = REPO_ROOT / "data analysis" / "outputs"
BRIDGE_CSV = EDA_OUTPUTS / "category_bridge_a_to_b.csv"
EVAL_CANDIDATES = EDA_OUTPUTS / "eval_candidates.csv"
EVAL_LABELS = EDA_OUTPUTS / "eval_labels.csv"
CREDS_PATH = REPO_ROOT / "openai_creds.yaml"
BANK_PATH = SYSTEM2_CACHE / "embedding_bank.npz"


# ----------------------------------------------------------------------------
# System 1 borrowed helpers - loaded by file path to avoid `solution`
# namespace collision between System 1 and System 2.
# ----------------------------------------------------------------------------

if __package__ in (None, ""):
    from solution._system1_loader import (  # type: ignore  # noqa: E402
        system1_score,
        system1_select,
        system1_validate,
    )
else:
    from .._system1_loader import system1_score, system1_select, system1_validate

_score_candidates = system1_score.score_candidates
_build_lookup_index = system1_score.build_lookup_index
_select_one = system1_select.select_one_b_per_a
_validate = system1_validate.validate_against_eval
_write_report = system1_validate.write_report


# ----------------------------------------------------------------------------
# Loading helpers
# ----------------------------------------------------------------------------


def load_canonical_tables() -> tuple[pd.DataFrame, pd.DataFrame]:
    a_path = SYSTEM1_CACHE / "store_A_full.parquet"
    b_path = SYSTEM1_CACHE / "store_B_full.parquet"
    if not a_path.exists() or not b_path.exists():
        raise SystemExit(
            f"Canonical parquets missing.\n  {a_path}\n  {b_path}\n"
            "Run SYSTEM 1 MVP first."
        )
    return pd.read_parquet(a_path), pd.read_parquet(b_path)


def load_catalog_vectors(store_label: str) -> tuple[np.ndarray, np.ndarray]:
    vec, ids = load_store_embeddings(SYSTEM2_CACHE, store_label)
    return vec, ids


def filter_to_eval(a: pd.DataFrame, b: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if not EVAL_CANDIDATES.exists():
        raise SystemExit(f"Missing {EVAL_CANDIDATES}")
    ec = pd.read_csv(EVAL_CANDIDATES, dtype=str)
    a_ids = set(ec["item_id_A"].astype(str))
    b_ids = set(ec["item_id_B"].astype(str))
    a_sub = a[a["item_id"].astype(str).isin(a_ids)].reset_index(drop=True)
    b_sub = b[b["item_id"].astype(str).isin(b_ids)].reset_index(drop=True)
    return a_sub, b_sub, ec


def load_category_bridge(path: Path) -> dict[str, list[tuple[str, float, int]]]:
    if not path.exists():
        return {}
    df = pd.read_csv(path)
    df = df[df["support"] >= 10]
    out: dict[str, list[tuple[str, float, int]]] = {}
    for r in df.itertuples(index=False):
        cat = str(r.a_category).strip().lower()
        slot = out.setdefault(cat, [])
        for col_cat, col_share in (
            ("b_top1", "b_top1_share"),
            ("b_top2", "b_top2_share"),
            ("b_top3", "b_top3_share"),
        ):
            b_cat = getattr(r, col_cat)
            share = getattr(r, col_share)
            if pd.notna(b_cat) and pd.notna(share):
                slot.append((str(b_cat).strip().lower(), float(share), int(r.support)))
    return out


# ----------------------------------------------------------------------------
# Embedding helpers
# ----------------------------------------------------------------------------


def get_embedder() -> OpenAIEmbedder:
    cfg = openai_embedding_config()
    return OpenAIEmbedder(
        api_key=cfg["api_key"],
        model=cfg["model"],
        dim=cfg["dim"],
        batch_size=cfg["batch"],
    )


def get_judge_client() -> tuple[OpenAI, str]:
    cfg = azure_llm_config()
    client = OpenAI(
        base_url=cfg["endpoint"],
        api_key=cfg["api_key"],
        max_retries=6,
        timeout=60.0,
    )
    return client, cfg["deployment"]


def get_bank(dim: int) -> EmbeddingBank:
    return EmbeddingBank(BANK_PATH, dim=dim)


# ----------------------------------------------------------------------------
# Pipeline stages
# ----------------------------------------------------------------------------


def stage_load_data(a_limit: int | None) -> tuple[pd.DataFrame, pd.DataFrame, dict, dict]:
    print("[1/8] Loading canonical tables ...")
    a, b = load_canonical_tables()
    if a_limit:
        a = a.head(a_limit).reset_index(drop=True)
    print(f"  A={len(a):,} rows  B={len(b):,} rows")
    a_index = _build_lookup_index(a)
    b_index = _build_lookup_index(b)
    return a, b, a_index, b_index


def stage_load_vectors(
    a: pd.DataFrame, b: pd.DataFrame
) -> tuple[np.ndarray, np.ndarray, NumpyVectorStore]:
    print("[2/8] Loading catalog vectors ...")
    a_vec_full, a_ids_full = load_catalog_vectors("A")
    b_vec_full, b_ids_full = load_catalog_vectors("B")

    a_id_to_idx = {str(x): i for i, x in enumerate(a_ids_full)}
    keep_a = np.array(
        [a_id_to_idx[i] for i in a["item_id"].astype(str) if i in a_id_to_idx], dtype=np.int64
    )
    if len(keep_a) != len(a):
        print(
            f"  WARNING: {len(a) - len(keep_a)} A rows missing vectors (will be dropped from T7)"
        )
    a_vec = a_vec_full[keep_a]
    a_ids = a["item_id"].astype(str).to_numpy()[: len(a_vec)]

    b_id_to_idx = {str(x): i for i, x in enumerate(b_ids_full)}
    keep_b = np.array(
        [b_id_to_idx[i] for i in b["item_id"].astype(str) if i in b_id_to_idx], dtype=np.int64
    )
    b_vec = b_vec_full[keep_b]
    b_ids = b["item_id"].astype(str).to_numpy()[: len(b_vec)]

    print(f"  loaded A vectors: {a_vec.shape}, B vectors: {b_vec.shape}")
    b_store = NumpyVectorStore(ids=b_ids, vectors=b_vec)
    return a_vec, a_ids, b_store


def stage_knowledge(embedder: OpenAIEmbedder, bank: EmbeddingBank) -> KnowledgeRetriever:
    print("[3/8] Bootstrapping + indexing knowledge base ...")
    stats = bootstrap_all(EDA_OUTPUTS, KNOWLEDGE_DIR)
    print(
        f"  bootstrap: rules={stats.rules}  aliases={stats.aliases}  "
        f"bridges={stats.bridges}  acc={stats.accepted_examples}  "
        f"rej={stats.rejected_examples}  edges={stats.edge_cases}  total={stats.total}"
    )
    kr = KnowledgeRetriever.build(KNOWLEDGE_DIR, embedder, bank)
    print(f"  knowledge index: {len(kr)} entries")
    return kr


def _load_system1_candidates_as_t1_t3_t5(parquet_path: Path):
    """Reuse System 1's already-computed candidates parquet. Saves ~50 min
    of T1+T3+T5 recomputation; results are identical to running them fresh
    because both systems share the same load.py/parse.py/retrieve.py.
    """
    from solution.retrieve.base import Candidate
    df = pd.read_parquet(parquet_path)
    out: list[Candidate] = []
    for r in df.itertuples(index=False):
        features = {}
        for f in ("rapidfuzz_wratio_name", "tfidf_cosine", "block_brand", "private_label_both"):
            v = getattr(r, f, None)
            if v is not None and not (isinstance(v, float) and v != v):
                features[f] = v
        out.append(
            Candidate(
                item_id_a=str(r.item_id_a),
                item_id_b=str(r.item_id_b),
                source=str(r.candidate_source),
                score=float(r.candidate_score),
                features=features,
            )
        )
    return out


def stage_retrieve(
    a: pd.DataFrame,
    b: pd.DataFrame,
    a_vec: np.ndarray,
    a_ids: np.ndarray,
    b_store: NumpyVectorStore,
    reuse_system1_candidates: bool = False,
) -> pd.DataFrame:
    print("[4/8] Generating candidates ...")
    if reuse_system1_candidates:
        s1_parquet = REPO_ROOT / "SYSTEM 1 MVP" / "cache" / "candidates_afull.parquet"
        if not s1_parquet.exists():
            print(f"  WARNING: {s1_parquet} not found; falling back to recompute T1/T3/T5")
        else:
            t0 = time.time()
            reused = _load_system1_candidates_as_t1_t3_t5(s1_parquet)
            print(f"  T1+T3+T5 reused from System 1: {len(reused)} ({time.time()-t0:.0f}s)")
            t0 = time.time()
            # Tightened in two passes at full scale:
            #   v1 (cosine>=0.55, k=20)  -> 3.27M T7 candidates (too many)
            #   v2 (cosine>=0.70, k=10)  -> 743k candidates / 6h wall clock
            #   v3 (cosine>=0.75, k=5)   -> ~300k candidates / ~3h target
            # Cosine 0.70-0.75 is the noisiest semantic-neighbor band;
            # tightening here mostly drops likely-false-positive pairs.
            t7 = SemanticRetriever(a_ids, a_vec, b_store, k=5, cosine_floor=0.75).retrieve_all()
            print(f"  T7: {len(t7)} ({time.time()-t0:.0f}s)")
            union = union_candidates(reused, t7)
            print(f"  union: {len(union)} unique pairs")
            return to_dataframe(union)

    t0 = time.time()
    t1 = StrictBlockRetriever(a, b).retrieve_all()
    print(f"  T1: {len(t1)} ({time.time()-t0:.0f}s)")
    t0 = time.time()
    t3 = TfidfRetriever(a, b).retrieve_all()
    print(f"  T3: {len(t3)} ({time.time()-t0:.0f}s)")
    t0 = time.time()
    t5 = PrivateLabelRetriever(a, b, BRIDGE_CSV).retrieve_all()
    print(f"  T5: {len(t5)} ({time.time()-t0:.0f}s)")
    t0 = time.time()
    t7 = SemanticRetriever(a_ids, a_vec, b_store, k=10, cosine_floor=0.7).retrieve_all()
    print(f"  T7: {len(t7)} ({time.time()-t0:.0f}s)")
    t0 = time.time()
    t7 = SemanticRetriever(a_ids, a_vec, b_store, k=20, cosine_floor=0.55).retrieve_all()
    print(f"  T7: {len(t7)} ({time.time()-t0:.0f}s)")

    union = union_candidates(t1, t3, t5, t7)
    print(f"  union: {len(union)} unique pairs")
    return to_dataframe(union)


def stage_score(
    candidates_df: pd.DataFrame, a_index: dict, b_index: dict
) -> pd.DataFrame:
    print("[5/8] Scoring deterministic features ...")
    t0 = time.time()
    scored = _score_candidates(candidates_df, a_index, b_index)
    print(f"  scored {len(scored)} in {time.time()-t0:.0f}s")
    if "route" in scored.columns:
        for r, n in scored["route"].value_counts().items():
            print(f"    {r}: {n}")
    return scored


def stage_judge(
    routed: pd.DataFrame,
    a_index: dict,
    b_index: dict,
    embedder: OpenAIEmbedder,
    knowledge: KnowledgeRetriever,
    bridge_idx: dict,
    workers: int,
    cache_path: Path,
) -> pd.DataFrame:
    print(f"[6/8] LLM judge on {len(routed)} routed candidates ...")
    if routed.empty:
        return pd.DataFrame()

    client, model = get_judge_client()
    cache = JudgmentCache(cache_path)
    builder = RAGContextBuilder(embedder, knowledge, category_bridge=bridge_idx)

    # Phase 1: collect every A-row group + its top-K B candidates.
    K = 5
    a_ids: list[str] = []
    a_rows: list[dict] = []
    b_ids_list: list[list[str]] = []
    b_rows_list: list[list[dict]] = []
    request_subs: list[pd.DataFrame] = []
    for a_id, sub in routed.groupby("item_id_a"):
        sub_top = sub.sort_values("final_score", ascending=False).head(K).reset_index(drop=True)
        a_row = a_index.get(a_id)
        if a_row is None:
            continue
        b_ids = sub_top["item_id_b"].astype(str).tolist()
        b_rows = [b_index.get(bid) or {} for bid in b_ids]
        a_ids.append(str(a_id))
        a_rows.append(a_row)
        b_ids_list.append(b_ids)
        b_rows_list.append(b_rows)
        request_subs.append(sub_top)

    n_groups = len(a_ids)
    print(f"  {n_groups} LLM batches (K=5)")

    # Phase 2: batch-embed ALL per-group RAG context queries up front.
    # This replaces the silent per-call serial embedding bottleneck.
    print(f"  [context] batch-embedding {n_groups} RAG queries ...")
    contexts = builder.build_many(list(zip(a_rows, b_rows_list)))

    # Phase 3: assemble the judge requests with their pre-built contexts.
    requests: list[JudgeRequest] = [
        JudgeRequest(
            a_id=a_ids[i],
            a_row=a_rows[i],
            b_ids=b_ids_list[i],
            b_rows=b_rows_list[i],
            extra_context_text=contexts[i].as_prompt_block(),
            context_signature=contexts[i].cache_signature(),
        )
        for i in range(n_groups)
    ]

    # Phase 4: run the LLM judge with thread-pool concurrency.
    judge = RAGJudge(client, model, cache, version=DEFAULT_CACHE_VERSION)
    results = judge.judge_many(requests, workers=workers)

    accepted_rows: list[dict[str, Any]] = []
    for r, sub in zip(results, request_subs):
        if r is None or not r.is_match or r.confidence < 0.85:
            continue
        idx = r.best_candidate_index
        if idx is None or idx < 0 or idx >= len(sub):
            continue
        row = sub.iloc[idx].to_dict()
        row["llm_confidence"] = r.confidence
        row["llm_match_type"] = r.match_type
        row["llm_evidence_summary"] = r.evidence_summary
        row["accept_source"] = "llm"
        accepted_rows.append(row)
    print(f"  LLM accepted {len(accepted_rows)} / {len(results)} batches (confidence>=0.85)")
    return pd.DataFrame(accepted_rows)


def stage_select_and_write(scored: pd.DataFrame, llm_accepted: pd.DataFrame, label: str):
    print(f"[7/8] Selecting one B per A ({label}) ...")
    auto = scored[scored["route"] == "auto_accept"].copy()
    auto["llm_confidence"] = None
    auto["accept_source"] = "rules"
    if llm_accepted.empty:
        accepted = auto
    else:
        accepted = pd.concat([auto, llm_accepted], ignore_index=True)
    final = _select_one(accepted)
    print(f"  final matches: {len(final)} rows")

    SYSTEM2_OUTPUTS.mkdir(parents=True, exist_ok=True)
    matches_csv = SYSTEM2_OUTPUTS / f"matches_{label}.csv"
    deliverable = final[["item_id_a", "item_id_b"]].copy()
    deliverable.columns = ["item_id_A", "item_id_B"]
    deliverable.to_csv(matches_csv, index=False)
    print(f"  wrote {matches_csv.name} ({len(deliverable)} rows)")
    # Also write the production deliverable to the system root so it's
    # prominent next to the README, not buried under outputs/.
    if label == "full":
        root_copy = SYSTEM2_ROOT / "matches.csv"
        deliverable.to_csv(root_copy, index=False)
        print(f"  wrote {root_copy} (deliverable copy)")
    final.to_csv(SYSTEM2_OUTPUTS / f"matches_with_features_{label}.csv", index=False)
    scored.to_csv(SYSTEM2_OUTPUTS / f"match_candidates_scored_{label}.csv", index=False)
    return matches_csv, final


def stage_validate(matches_csv: Path, label: str):
    print("[8/8] Validating against eval set ...")
    report = _validate(matches_csv, EVAL_CANDIDATES, EVAL_LABELS)
    md_path = SYSTEM2_OUTPUTS / f"validation_report_{label}.md"
    _write_report(report, md_path)
    if "overall" in report:
        o = report["overall"]
        print(
            f"  eval-set: n={o['n']} shipped={o['shipped']} "
            f"P={o['precision']} R={o['recall']} F1={o['f1']}"
        )
    return report


# ----------------------------------------------------------------------------
# Top-level
# ----------------------------------------------------------------------------


def run(mode: str, a_limit: int | None, llm_workers: int, reuse_system1_candidates: bool = False):
    SYSTEM2_OUTPUTS.mkdir(parents=True, exist_ok=True)
    label = mode if not a_limit else f"{mode}_a{a_limit}"

    a, b, a_index, b_index = stage_load_data(a_limit)

    if mode == "eval":
        a, b, _ = filter_to_eval(a, b)
        a_index = _build_lookup_index(a)
        b_index = _build_lookup_index(b)
        print(f"  eval mode: filtered to A={len(a)} B={len(b)} from eval set")

    a_vec, a_ids, b_store = stage_load_vectors(a, b)

    embedder = get_embedder()
    bank = get_bank(embedder.dim)
    knowledge = stage_knowledge(embedder, bank)

    bridge_idx = load_category_bridge(BRIDGE_CSV)

    candidates_df = stage_retrieve(
        a, b, a_vec, a_ids, b_store, reuse_system1_candidates=reuse_system1_candidates
    )
    scored = stage_score(candidates_df, a_index, b_index)

    routed = scored[scored["route"] == "route_to_llm"]
    cache_path = SYSTEM2_OUTPUTS / f"llm_judgments_{label}.jsonl"
    llm_accepted = stage_judge(
        routed, a_index, b_index, embedder, knowledge, bridge_idx,
        workers=llm_workers, cache_path=cache_path,
    )
    matches_csv, _ = stage_select_and_write(scored, llm_accepted, label)
    stage_validate(matches_csv, label)
    print("Done.")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["eval", "full"], default="eval")
    p.add_argument("--a-limit", type=int, default=None)
    p.add_argument("--llm-workers", type=int, default=5)
    p.add_argument(
        "--reuse-system1-candidates",
        action="store_true",
        help="Skip T1/T3/T5 recompute; read System 1's candidates parquet. "
             "Saves ~50 min for the full run; results are identical because both "
             "systems share the same retriever code.",
    )
    args = p.parse_args()
    run(args.mode, args.a_limit, args.llm_workers, reuse_system1_candidates=args.reuse_system1_candidates)


if __name__ == "__main__":
    main()
