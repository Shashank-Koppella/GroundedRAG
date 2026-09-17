"""
Recall@k / MRR scoring, with bootstrapped confidence intervals (same machinery as the
VLM project's bootstrap testing, per Section 6/7 of the plan).

Works against ANY retriever object exposing `.rank(query, ticker=None, k=int) -> List[chunk_id]`
so the same scoring code runs unchanged against the keyword baseline today, and against
BM25/dense/hybrid/reranked retrievers from Day 5 onward — that reuse is the point of
building the harness before the system it evaluates.

Only "chunk"-type questions (single_hop, and any out_of_scope questions you choose to
score for false-positive rate) go through Recall@k/MRR. "sql"-type (multi_hop_numeric)
questions are reported separately — see score_sql_routing_sanity() below — since there's
no gold chunk to recall until the SQL tool exists in Phase B.
"""
import json
import random
from pathlib import Path
from typing import List, Dict, Callable


def recall_at_k(gold_ids: List[str], retrieved_ids: List[str], k: int) -> int:
    """1 if any gold chunk appears in the top-k retrieved, else 0."""
    if not gold_ids:
        return 0
    top_k = set(retrieved_ids[:k])
    return 1 if top_k & set(gold_ids) else 0


def reciprocal_rank(gold_ids: List[str], retrieved_ids: List[str]) -> float:
    """1/rank of the first correct result; 0 if none found."""
    gold_set = set(gold_ids)
    for i, chunk_id in enumerate(retrieved_ids, start=1):
        if chunk_id in gold_set:
            return 1.0 / i
    return 0.0


def bootstrap_ci(values: List[float], n_resamples: int = 2000, ci: float = 0.95, seed: int = 42):
    """Percentile bootstrap CI over a list of per-question metric values."""
    if not values:
        return (float("nan"), float("nan"))
    rng = random.Random(seed)
    n = len(values)
    means = []
    for _ in range(n_resamples):
        resample = [values[rng.randrange(n)] for _ in range(n)]
        means.append(sum(resample) / n)
    means.sort()
    lower_idx = int((1 - ci) / 2 * n_resamples)
    upper_idx = int((1 + ci) / 2 * n_resamples) - 1
    return (means[lower_idx], means[upper_idx])


def score_retrieval(
    questions: List[Dict],
    rank_fn: Callable[[str, str, int], List[str]],
    k_values: List[int] = (1, 3, 5, 10),
) -> Dict:
    """
    questions: eval_questions.json entries with gold_answer_type == "chunk" and a
               non-empty gold_chunk_ids list (unlabeled questions are skipped, with a
               warning, so a partially-labeled set still produces a valid partial report).
    rank_fn:   callable(query, ticker, k) -> List[chunk_id], e.g. KeywordBaseline(...).rank
    """
    scoreable = [q for q in questions if q["gold_answer_type"] == "chunk" and q["gold_chunk_ids"]]
    skipped = [q for q in questions if q["gold_answer_type"] == "chunk" and not q["gold_chunk_ids"]]
    if skipped:
        print(f"[scoring] Skipping {len(skipped)} unlabeled chunk-type questions "
              f"(run scripts/label_eval_set.py to label them): "
              f"{[q['id'] for q in skipped]}")

    max_k = max(k_values)
    recall_by_k = {k: [] for k in k_values}
    rr_values = []
    per_question = []

    for q in scoreable:
        retrieved = rank_fn(q["question"], q.get("ticker"), max_k)
        row = {"id": q["id"], "question": q["question"], "gold": q["gold_chunk_ids"], "retrieved": retrieved}
        for k in k_values:
            row[f"recall@{k}"] = recall_at_k(q["gold_chunk_ids"], retrieved, k)
            recall_by_k[k].append(row[f"recall@{k}"])
        rr = reciprocal_rank(q["gold_chunk_ids"], retrieved)
        row["reciprocal_rank"] = rr
        rr_values.append(rr)
        per_question.append(row)

    report = {
        "n_scored": len(scoreable),
        "n_skipped_unlabeled": len(skipped),
        "recall_at_k": {},
        "mrr": None,
        "per_question": per_question,
    }
    for k in k_values:
        vals = recall_by_k[k]
        mean = sum(vals) / len(vals) if vals else float("nan")
        lo, hi = bootstrap_ci(vals) if vals else (float("nan"), float("nan"))
        report["recall_at_k"][k] = {"mean": mean, "ci_95": [lo, hi]}

    if rr_values:
        mrr_mean = sum(rr_values) / len(rr_values)
        lo, hi = bootstrap_ci(rr_values)
        report["mrr"] = {"mean": mrr_mean, "ci_95": [lo, hi]}

    return report


def score_sql_routing_sanity(
    questions: List[Dict],
    rank_fn: Callable[[str, str, int], List[str]],
    confidence_threshold_terms: int = 2,
) -> Dict:
    """
    Pre-agent sanity check for multi_hop_numeric questions: does the keyword/dense
    retriever return chunks with meaningful term overlap for a question that actually
    needs SQL, not prose? A high hit rate here is a real early warning that the eventual
    agent will need a strong routing signal (Phase B, Opus session #2) rather than just
    "if retrieval returns something, trust it."

    This is NOT Recall@k (there's no gold chunk) — it's a false-confidence probe.
    """
    numeric_qs = [q for q in questions if q["gold_answer_type"] == "sql"]
    results = []
    for q in numeric_qs:
        tickers = q["ticker"].split(",") if q["ticker"] else [None]
        for ticker in tickers:
            retrieved = rank_fn(q["question"], ticker, 5)
            results.append({
                "id": q["id"],
                "ticker": ticker,
                "n_chunks_returned": len(retrieved),
                "false_confidence_flag": len(retrieved) > 0,
            })
    n_flagged = sum(1 for r in results if r["false_confidence_flag"])
    return {
        "n_numeric_questions_checked": len(results),
        "n_false_confidence_flags": n_flagged,
        "note": ("Any non-zero flag here means the retriever will confidently hand prose "
                 "to a numeric question if nothing routes it to SQL first — expected until "
                 "the agent exists (Phase B), tracked here as a baseline to compare against."),
        "details": results,
    }


if __name__ == "__main__":
    import argparse
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from keyword_baseline import KeywordBaseline, load_chunks

    parser = argparse.ArgumentParser()
    parser.add_argument("--questions", default=str(Path(__file__).resolve().parents[2] / "data" / "eval_set" / "eval_questions.json"))
    parser.add_argument("--chunks", required=True)
    args = parser.parse_args()

    questions = json.loads(Path(args.questions).read_text())
    chunks = load_chunks(args.chunks)
    baseline = KeywordBaseline(chunks)

    retrieval_report = score_retrieval(questions, baseline.rank)
    sql_report = score_sql_routing_sanity(questions, baseline.rank)

    print(json.dumps({"retrieval": {k: v for k, v in retrieval_report.items() if k != "per_question"},
                       "sql_routing_sanity": {k: v for k, v in sql_report.items() if k != "details"}},
                      indent=2))
