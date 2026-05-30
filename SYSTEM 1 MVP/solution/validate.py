"""Eval-set validation: join matches.csv against eval_labels.csv and
report precision/recall/F1 per stratum.

Notes:
- The eval set contains specific (item_id_A, item_id_B) pairs with
  labels. We measure:
    * Precision on the SHIPPED matches that have an eval label
      (positive label rate among shipped matches present in the eval).
    * Recall on the eval positives: how many of them did we ship?
- The candidate sample is intentionally biased to interesting strata
  (mostly likely-matches and likely-non-matches), so absolute numbers
  here are a proxy, not the truth for the whole catalog.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd


def validate_against_eval(
    matches_csv: Path,
    eval_candidates_csv: Path,
    eval_labels_csv: Path,
) -> dict:
    if not matches_csv.exists():
        return {"error": f"missing matches: {matches_csv}"}
    if not eval_candidates_csv.exists() or not eval_labels_csv.exists():
        return {"error": "missing eval files"}

    matches = pd.read_csv(matches_csv, dtype=str)
    cand = pd.read_csv(eval_candidates_csv, dtype=str)
    labels = pd.read_csv(eval_labels_csv)
    eval_df = cand.merge(labels, on="pair_id", how="inner")

    # Coerce types.
    eval_df["label_is_match"] = eval_df["label_is_match"].astype(str).str.lower().isin({"true", "1"})

    # Each eval row is one (A, B) pair with a label. We check whether
    # that exact pair appears in matches.csv.
    matches_pairs = set(zip(matches["item_id_A"].astype(str), matches["item_id_B"].astype(str)))
    eval_df["shipped"] = eval_df.apply(
        lambda r: (str(r["item_id_A"]), str(r["item_id_B"])) in matches_pairs, axis=1
    )

    pos = int(eval_df["label_is_match"].sum())
    neg = int((~eval_df["label_is_match"]).sum())
    shipped = int(eval_df["shipped"].sum())
    tp = int(((eval_df["shipped"]) & (eval_df["label_is_match"])).sum())
    fp = int(((eval_df["shipped"]) & (~eval_df["label_is_match"])).sum())
    fn = int(((~eval_df["shipped"]) & (eval_df["label_is_match"])).sum())
    tn = int(((~eval_df["shipped"]) & (~eval_df["label_is_match"])).sum())

    # Audit fix #16: distinguish "no overlap" from "zero precision".
    precision = round(tp / (tp + fp), 4) if (tp + fp) > 0 else None
    recall = round(tp / (tp + fn), 4) if (tp + fn) > 0 else None
    f1 = (
        round(2 * precision * recall / (precision + recall), 4)
        if precision and recall
        else None
    )

    overall = {
        "n": len(eval_df),
        "positives": pos,
        "negatives": neg,
        "shipped": shipped,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "overlap_note": "no shipped eval pairs" if shipped == 0 else None,
    }

    per_stratum = []
    for stratum, sub in eval_df.groupby("stratum"):
        n = len(sub)
        p = int(sub["label_is_match"].sum())
        s = int(sub["shipped"].sum())
        tp_s = int(((sub["shipped"]) & (sub["label_is_match"])).sum())
        fp_s = int(((sub["shipped"]) & (~sub["label_is_match"])).sum())
        fn_s = int(((~sub["shipped"]) & (sub["label_is_match"])).sum())
        prec_s = round(tp_s / (tp_s + fp_s), 4) if (tp_s + fp_s) > 0 else None
        rec_s = round(tp_s / p, 4) if p > 0 else None
        per_stratum.append(
            {
                "stratum": stratum,
                "n": n,
                "positives": p,
                "shipped": s,
                "tp": tp_s,
                "fp": fp_s,
                "fn": fn_s,
                "precision_on_shipped": prec_s,
                "recall_on_positives": rec_s,
            }
        )

    return {"overall": overall, "per_stratum": per_stratum}


def write_report(report: dict, md_path: Path) -> None:
    md_path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["# Eval-set Validation\n"]
    if "error" in report:
        lines.append(f"ERROR: {report['error']}\n")
        md_path.write_text("\n".join(lines), encoding="utf-8")
        return
    o = report["overall"]
    lines.append("## Overall\n")
    lines.append("| metric | value |")
    lines.append("|---|---|")
    for k, v in o.items():
        lines.append(f"| {k} | {v} |")
    lines.append("")
    lines.append("## Per stratum\n")
    lines.append("| stratum | n | positives | shipped | tp | fp | fn | prec_on_shipped | recall_on_positives |")
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for r in report["per_stratum"]:
        lines.append(
            f"| {r['stratum']} | {r['n']} | {r['positives']} | {r['shipped']} | "
            f"{r['tp']} | {r['fp']} | {r['fn']} | "
            f"{r['precision_on_shipped']} | {r['recall_on_positives']} |"
        )
    md_path.write_text("\n".join(lines), encoding="utf-8")
