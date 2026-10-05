"""
Label equivalent passages on the single-hop eval questions, then re-score the stored Day 5 rankings with
them. Offline: needs only the chunk files and data/eval_set/day5_ablation_results.json (the stored
top-10 lists), not Qdrant or any model.

    python -m scripts.rescore_day5_equivalents            # dry run: prints the table, writes nothing to the eval set
    python -m scripts.rescore_day5_equivalents --write    # also stores equivalent_chunk_ids in eval_questions.json

Writes data/eval_set/day5_rescored_equivalents.json (strict gold vs equivalence-aware, at thresholds
1.0 / 0.9 / 0.8, for every Day 5 stage). See src/eval/equivalence.py for the rule and why it exists.
The original day5_ablation_results.json is NOT modified: it is the record of what was reported.
"""
import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.eval.equivalence import PRIMARY_THRESHOLD, RULE, SENSITIVITY_THRESHOLDS, find_equivalents  # noqa: E402
from src.eval.keyword_baseline import load_chunks  # noqa: E402
from src.eval.scoring import bootstrap_ci, recall_at_k, reciprocal_rank  # noqa: E402

EVAL = REPO_ROOT / "data" / "eval_set" / "eval_questions.json"
DAY5 = REPO_ROOT / "data" / "eval_set" / "day5_ablation_results.json"
OUT = REPO_ROOT / "data" / "eval_set" / "day5_rescored_equivalents.json"
STAGES = ("keyword_baseline", "bm25", "hybrid_rrf", "hybrid_reranked")
KS = (1, 3, 5, 10)


def score_lists(per_question, relevant_by_id):
    rows, out = [], {}
    for pq in per_question:
        rel = relevant_by_id[pq["id"]]
        row = {"id": pq["id"], **{f"recall@{k}": recall_at_k(rel, pq["retrieved"], k) for k in KS},
               "reciprocal_rank": reciprocal_rank(rel, pq["retrieved"])}
        rows.append(row)
    for k in KS:
        vals = [r[f"recall@{k}"] for r in rows]
        out[f"recall@{k}"] = {"mean": sum(vals) / len(vals), "ci_95": list(bootstrap_ci(vals))}
    rr = [r["reciprocal_rank"] for r in rows]
    out["mrr@10"] = {"mean": sum(rr) / len(rr), "ci_95": list(bootstrap_ci(rr))}
    out["per_question"] = rows
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true", help="store equivalent_chunk_ids (primary threshold) in the eval set")
    args = ap.parse_args()

    chunks = {}
    for f in sorted((REPO_ROOT / "data" / "processed" / "chunks").glob("*.jsonl")):
        for c in load_chunks(str(f)):
            chunks[c["chunk_id"]] = c
    questions = json.loads(EVAL.read_text(encoding="utf-8"))
    single = [q for q in questions if q["type"] == "single_hop"]
    day5 = json.loads(DAY5.read_text(encoding="utf-8"))

    tokens = {}
    equivalents = {t: {q["id"]: find_equivalents(q["gold_chunk_ids"], chunks, t, tokens) for q in single}
                   for t in SENSITIVITY_THRESHOLDS}

    result = {"rule": RULE.format(t=PRIMARY_THRESHOLD), "primary_threshold": PRIMARY_THRESHOLD,
              "note": "Rankings are the stored Day 5 top-10 lists; only the relevance labels differ. "
                      "MRR is MRR@10 (lists are cut at 10). 'as_reported' is whatever day5_ablation_results.json "
                      "holds when this runs: after the Oct 5 corpus rebuild it is the re-run on the NEW corpus, "
                      "not the original old-corpus record (that survives only in CHANGES.md / the plan doc).",
              "equivalents_per_question": {t: {qid: list(e) for qid, e in eq.items()} for t, eq in equivalents.items()},
              "stages": {}}
    for stage in STAGES:
        pq = day5[stage]["per_question"]
        strict = {q["id"]: q["gold_chunk_ids"] for q in single}
        rep = day5[stage]
        result["stages"][stage] = {
            "as_reported": {**{f"recall@{k}": rep["recall_at_k"][str(k)] for k in KS}, "mrr@10": rep["mrr"]},
            "strict_gold": score_lists(pq, strict)}
        for t in SENSITIVITY_THRESHOLDS:
            rel = {q["id"]: q["gold_chunk_ids"] + list(equivalents[t][q["id"]]) for q in single}
            result["stages"][stage][f"equivalent_{t}"] = score_lists(pq, rel)
    OUT.write_text(json.dumps(result, indent=2), encoding="utf-8")

    print(f"Equivalents per question at {PRIMARY_THRESHOLD}: "
          f"{ {qid: len(e) for qid, e in equivalents[PRIMARY_THRESHOLD].items() if e} }")
    cols = ["as_reported", "strict_gold"] + [f"equivalent_{t}" for t in SENSITIVITY_THRESHOLDS]
    for metric in ("recall@1", "recall@5", "recall@10", "mrr@10"):
        print(f"\n{metric:10} " + " ".join(f"{c:>16}" for c in cols))
        for stage in STAGES:
            vals = [result["stages"][stage][c][metric]["mean"] for c in cols]
            print(f"{stage:18}"[:18] + " " + " ".join(f"{v:16.3f}" for v in vals))
    print(f"\nWrote {OUT}")

    if args.write:
        for q in single:
            q["equivalent_chunk_ids"] = list(equivalents[PRIMARY_THRESHOLD][q["id"]])
            q["equivalence_rule"] = RULE.format(t=PRIMARY_THRESHOLD)
        EVAL.write_text(json.dumps(questions, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"Stored equivalent_chunk_ids (threshold {PRIMARY_THRESHOLD}) in {EVAL}")


if __name__ == "__main__":
    main()
