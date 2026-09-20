"""
Phase A, Day 5 follow-up diagnostic -- answers the one question the ablation table
cannot: WHY the numbers are low, and therefore whether Phase C's LoRA reranker
fine-tune is aimed at the real bottleneck.

The Day 5 table shows hybrid R@10 = 0.35 and the reranker LOSING two correct chunks
(sh_006, sh_019) while rescuing zero. But the results JSON only stores each stage's
top-10, so three things stay invisible:

  1. RECALL CEILING. For the 13 questions everything misses, is the gold chunk even in
     the candidate pool the reranker sees (hybrid's top-50)? If yes, the bottleneck is
     the RERANKER and a LoRA fine-tune targets it directly. If no, the bottleneck is
     RETRIEVAL/CHUNKING and no amount of reranker fine-tuning can fix it -- you'd be
     spending Phase C on the wrong component. This is a go/no-go input for the plan's
     Opus session #3.

  2. WHICH RETRIEVER FINDS WHAT. Gold's uncapped rank in BM25 vs dense separately,
     which says whether the two retrievers fail on the SAME questions (fusion can't
     help) or DIFFERENT ones (fusion has headroom the k=60 setting isn't capturing).

  3. ADJACENT-CHUNK NEAR MISSES -- a possible measurement artifact. Chunks are 250
     words with 40-word overlap and each question has exactly ONE hand-labeled gold
     chunk_id. If a retriever returns the chunk immediately before/after gold, that
     chunk very likely contains much of the same passage, but Recall@k scores it as a
     complete miss. If this is common, the true retrieval quality is materially better
     than the headline numbers, and the eval is understating the system.

Also dumps the hard negatives (chunks ranked above gold) for Phase C hard-negative
mining -- the plan's Day 6 task.

Usage (run from repo root, venv active, Qdrant running with all 11,245 points):
    python -m scripts.diagnose_day5_ceiling

Writes data/eval_set/day5_ceiling_diagnostic.json and prints a summary.
"""
import json
import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.eval.keyword_baseline import load_chunks
from src.retrieval.bm25_retriever import BM25Retriever
from src.retrieval.dense_retriever import DenseRetriever
from src.retrieval.hybrid_retriever import HybridRetriever, reciprocal_rank_fusion

DEEP_K = 200  # how far down each retriever's list to look for gold


def parse_chunk_id(chunk_id):
    """AAPL_000032019324000123_item_1a_0010 -> (AAPL, 000032019324000123, item_1a, 10)"""
    parts = chunk_id.rsplit("_", 1)
    prefix, idx = parts[0], int(parts[1])
    return prefix, idx


def rank_in(ranked_ids, gold_ids):
    gold = set(gold_ids)
    for i, cid in enumerate(ranked_ids, 1):
        if cid in gold:
            return i
    return None


def main():
    all_chunks = []
    for f in sorted((REPO_ROOT / "data" / "processed" / "chunks").glob("*.jsonl")):
        all_chunks.extend(load_chunks(str(f)))
    chunks_by_id = {c["chunk_id"]: c for c in all_chunks}
    print(f"Loaded {len(all_chunks)} chunks")

    questions = json.loads((REPO_ROOT / "data" / "eval_set" / "eval_questions.json").read_text())
    single_hop = [q for q in questions if q["gold_answer_type"] == "chunk" and q["gold_chunk_ids"]]

    bm25 = BM25Retriever(all_chunks)
    dense = DenseRetriever()
    hybrid = HybridRetriever(bm25, dense, k_rrf=60, candidate_pool_size=100)

    rows = []
    for q in single_hop:
        ticker, gold_ids = q.get("ticker"), q["gold_chunk_ids"]

        bm25_list = bm25.rank(q["question"], ticker=ticker, k=DEEP_K)
        dense_list = dense.rank(q["question"], ticker=ticker, k=DEEP_K)
        fused_list = reciprocal_rank_fusion([bm25_list, dense_list], k=60)

        gold_prefix, gold_idx = parse_chunk_id(gold_ids[0])
        # Adjacent-chunk near miss: same filing+item, chunk_index within +/-1 of gold.
        adjacent_ids = {f"{gold_prefix}_{gold_idx + d:04d}" for d in (-1, 1)}
        adjacent_ids &= set(chunks_by_id)

        fused_top10 = fused_list[:10]
        adj_rank = rank_in(fused_top10, adjacent_ids) if adjacent_ids else None

        # Hard negatives for Phase C: what the fused retriever put ABOVE gold.
        g_rank = rank_in(fused_list, gold_ids)
        hard_negs = fused_list[:min(g_rank - 1, 10)] if g_rank else fused_list[:10]

        rows.append({
            "id": q["id"], "ticker": ticker, "gold": gold_ids[0],
            "bm25_rank": rank_in(bm25_list, gold_ids),
            "dense_rank": rank_in(dense_list, gold_ids),
            "hybrid_rank": g_rank,
            "in_reranker_pool_top50": bool(g_rank and g_rank <= 50),
            "adjacent_chunk_in_top10_rank": adj_rank,
            "hard_negatives": hard_negs,
        })

    # ---- summary ----
    def pct(n): return f"{n}/{len(rows)} ({100*n/len(rows):.0f}%)"

    print("\n" + "=" * 78)
    print("RECALL CEILING DIAGNOSTIC")
    print("=" * 78)
    print(f"\n{'qid':7} {'tick':6} {'BM25':>6} {'DENSE':>6} {'HYBRID':>7} {'in top50?':>10} {'adj@10':>7}")
    print("-" * 78)
    for r in rows:
        f = lambda x: str(x) if x else ">%d" % DEEP_K
        print(f"{r['id']:7} {str(r['ticker']):6} {f(r['bm25_rank']):>6} {f(r['dense_rank']):>6} "
              f"{f(r['hybrid_rank']):>7} {str(r['in_reranker_pool_top50']):>10} "
              f"{str(r['adjacent_chunk_in_top10_rank'] or '-'):>7}")

    in_pool = sum(1 for r in rows if r["in_reranker_pool_top50"])
    in_top10 = sum(1 for r in rows if r["hybrid_rank"] and r["hybrid_rank"] <= 10)
    adj_hits = sum(1 for r in rows if r["adjacent_chunk_in_top10_rank"])
    bm25_only = sum(1 for r in rows if r["bm25_rank"] and not r["dense_rank"])
    dense_only = sum(1 for r in rows if r["dense_rank"] and not r["bm25_rank"])
    neither = sum(1 for r in rows if not r["bm25_rank"] and not r["dense_rank"])

    print("\n" + "-" * 78)
    print(f"Gold in hybrid top-10 (measured R@10):        {pct(in_top10)}")
    print(f"Gold in hybrid top-50 (RERANKER CEILING):     {pct(in_pool)}")
    print(f"  -> headroom a perfect reranker could reach: {in_pool - in_top10} more questions")
    print(f"\nAdjacent chunk in top-10 but gold not counted: {pct(adj_hits)}")
    print(f"  -> if high, Recall@k UNDERSTATES real quality (labeling granularity artifact)")
    print(f"\nFound by BM25 only (within top {DEEP_K}):   {bm25_only}")
    print(f"Found by dense only (within top {DEEP_K}):  {dense_only}")
    print(f"Found by NEITHER (within top {DEEP_K}):     {neither}  <- retrieval/chunking failures")

    out = REPO_ROOT / "data" / "eval_set" / "day5_ceiling_diagnostic.json"
    out.write_text(json.dumps(rows, indent=2))
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
