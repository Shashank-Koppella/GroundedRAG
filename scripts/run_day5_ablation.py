"""
Phase A, Day 5 -- runs the full 4-stage hybrid-retrieval ablation described in
Section 7 of the plan: keyword baseline -> BM25 -> RRF hybrid (BM25+dense, k=60) ->
+ baseline reranker, scoring each stage with the existing eval harness
(src/eval/scoring.py) against the same 20 labeled single-hop questions, plus the
SQL-routing sanity check against the 20 multi-hop-numeric questions.

Needs a running local Qdrant with the Day 3 embeddings already uploaded, and
huggingface.co reachable (to load bge-base-en-v1.5 for query embedding and
cross-encoder/ms-marco-MiniLM-L-6-v2 for reranking) -- i.e. run this from your own
machine/venv, not from inside a sandboxed agent session.

Usage (real, concrete command -- run from the repo root, with your venv active):
    python -m scripts.run_day5_ablation

Writes data/eval_set/day5_ablation_results.json with every stage's full report
(including per-question detail) and prints the summary table to stdout.
"""
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.eval.keyword_baseline import load_chunks, KeywordBaseline
from src.eval.scoring import score_retrieval, score_sql_routing_sanity
from src.retrieval.bm25_retriever import BM25Retriever
from src.retrieval.dense_retriever import DenseRetriever
from src.retrieval.hybrid_retriever import HybridRetriever
from src.retrieval.reranker import RerankerRetriever


def summarize(name, report):
    lines = [f"\n=== {name} ==="]
    lines.append(f"n_scored={report['n_scored']} n_skipped_unlabeled={report['n_skipped_unlabeled']}")
    for k, v in report["recall_at_k"].items():
        lines.append(f"  Recall@{k}: {v['mean']:.4f}  (95% CI [{v['ci_95'][0]:.4f}, {v['ci_95'][1]:.4f}])")
    lines.append(f"  MRR: {report['mrr']['mean']:.4f}  (95% CI [{report['mrr']['ci_95'][0]:.4f}, {report['mrr']['ci_95'][1]:.4f}])")
    text = "\n".join(lines)
    print(text)
    return text


def main():
    t0 = time.time()
    all_chunks = []
    for f in sorted((REPO_ROOT / "data" / "processed" / "chunks").glob("*.jsonl")):
        all_chunks.extend(load_chunks(str(f)))
    chunks_by_id = {c["chunk_id"]: c for c in all_chunks}
    print(f"Loaded {len(all_chunks)} chunks in {time.time() - t0:.1f}s")

    questions = json.loads((REPO_ROOT / "data" / "eval_set" / "eval_questions.json").read_text())

    results = {}

    # Stage 0: keyword baseline (Day 3 floor, re-run here for a same-process comparison)
    kw = KeywordBaseline(all_chunks)
    results["keyword_baseline"] = score_retrieval(questions, kw.rank)

    # Stage 1: BM25
    t0 = time.time()
    bm25 = BM25Retriever(all_chunks)
    print(f"Built BM25 index in {time.time() - t0:.1f}s")
    results["bm25"] = score_retrieval(questions, bm25.rank)
    results["bm25_sql_sanity"] = score_sql_routing_sanity(questions, bm25.rank)

    # Stage 2: RRF hybrid (BM25 + dense, k=60)
    t0 = time.time()
    dense = DenseRetriever()  # loads bge-base-en-v1.5 + connects to localhost:6333
    hybrid = HybridRetriever(bm25, dense, k_rrf=60, candidate_pool_size=100)
    print(f"Built dense retriever + hybrid wrapper in {time.time() - t0:.1f}s")
    results["hybrid_rrf"] = score_retrieval(questions, hybrid.rank)
    results["hybrid_sql_sanity"] = score_sql_routing_sanity(questions, hybrid.rank)

    # Stage 3: hybrid + baseline reranker
    t0 = time.time()
    reranker = RerankerRetriever(hybrid, chunks_by_id, candidate_pool_size=50)
    print(f"Loaded reranker model in {time.time() - t0:.1f}s")
    results["hybrid_reranked"] = score_retrieval(questions, reranker.rank)
    results["hybrid_reranked_sql_sanity"] = score_sql_routing_sanity(questions, reranker.rank)

    print("\n" + "=" * 60)
    print("DAY 5 ABLATION TABLE")
    print("=" * 60)
    summarize("Stage 0: Keyword baseline (Day 3 floor)", results["keyword_baseline"])
    summarize("Stage 1: BM25", results["bm25"])
    summarize("Stage 2: RRF hybrid (BM25 + dense, k=60)", results["hybrid_rrf"])
    summarize("Stage 3: RRF hybrid + baseline reranker", results["hybrid_reranked"])

    out_path = REPO_ROOT / "data" / "eval_set" / "day5_ablation_results.json"
    serializable = {k: v for k, v in results.items()}
    out_path.write_text(json.dumps(serializable, indent=2))
    print(f"\nWrote full results (including per-question detail) to {out_path}")


if __name__ == "__main__":
    main()
