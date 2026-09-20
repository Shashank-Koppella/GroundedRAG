"""
Hand-written Reciprocal Rank Fusion (RRF) -- Phase A, Day 5, stage 2 of the hybrid
retrieval ablation (Section 5/6/7 of the plan).

Deliberately hand-written rather than reaching for Qdrant's built-in sparse-vector
fusion (Section 5's stated tradeoff: doing this by hand means being able to explain
the fusion math in an interview, not just "the vector DB handles it").

RRF formula (Section 6):
    score(doc) = sum over retrievers i of  1 / (k + rank_i(doc))
where rank_i(doc) is the doc's 1-indexed rank in retriever i's list (a doc not
returned by a retriever at all contributes 0 for that retriever, not a penalty term --
this is standard RRF and is what lets it combine lists from retrievers with
incomparable/uncalibrated raw scores, which is exactly BM25 vs. cosine similarity).

k=60 is the standard default from the original RRF paper (Cormack et al. 2009) and is
what Section 6/7 of the plan specifies -- it controls how much rank position matters
for low-ranked docs (larger k flattens the curve, giving less weight to being #1 vs #5).

HybridRetriever wraps any two objects exposing `.rank(query, ticker=None, k=int)`, so
it works with BM25Retriever + DenseRetriever today, and would work unchanged with any
other retriever pair implementing the same interface.
"""
import json
from typing import List, Dict, Optional


def reciprocal_rank_fusion(ranked_lists: List[List[str]], k: int = 60) -> List[str]:
    """
    ranked_lists: one ranked list of chunk_ids (best-first) per retriever.
    Returns a single fused list of chunk_ids, best-first, covering the union of all
    input lists (a chunk_id appearing in only one list still gets a (small) score
    from that one list and is included).
    """
    scores: Dict[str, float] = {}
    for ranked in ranked_lists:
        for rank, chunk_id in enumerate(ranked, start=1):
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (k + rank)
    fused = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [chunk_id for chunk_id, _ in fused]


class HybridRetriever:
    """Fuses two retrievers' rankings via RRF. `.rank(query, ticker=None, k=10)`
    returns the fused top-k chunk_ids, matching every other retriever's interface.

    Each underlying retriever is asked for `candidate_pool_size` results (not just
    the final `k`) before fusion, so RRF has enough of each retriever's ranking to
    actually combine -- fusing two already-truncated top-10 lists would silently drop
    a chunk that (say) BM25 ranked #3 but dense ranked #40, even though RRF's whole
    point is to let a strong showing in even one retriever surface a doc the other
    missed.
    """

    def __init__(self, retriever_a, retriever_b, k_rrf: int = 60, candidate_pool_size: int = 100):
        self.retriever_a = retriever_a
        self.retriever_b = retriever_b
        self.k_rrf = k_rrf
        self.candidate_pool_size = candidate_pool_size

    def rank(self, query: str, ticker: Optional[str] = None, k: int = 10) -> List[str]:
        list_a = self.retriever_a.rank(query, ticker=ticker, k=self.candidate_pool_size)
        list_b = self.retriever_b.rank(query, ticker=ticker, k=self.candidate_pool_size)
        fused = reciprocal_rank_fusion([list_a, list_b], k=self.k_rrf)
        return fused[:k]


if __name__ == "__main__":
    import argparse
    from pathlib import Path
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from src.eval.keyword_baseline import load_chunks
    from src.retrieval.bm25_retriever import BM25Retriever
    from src.retrieval.dense_retriever import DenseRetriever

    parser = argparse.ArgumentParser()
    parser.add_argument("--chunks-dir", required=True)
    parser.add_argument("--query", required=True)
    parser.add_argument("--ticker", default=None)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=6333)
    args = parser.parse_args()

    all_chunks = []
    for f in sorted(Path(args.chunks_dir).glob("*.jsonl")):
        all_chunks.extend(load_chunks(str(f)))

    bm25 = BM25Retriever(all_chunks)
    dense = DenseRetriever(host=args.host, port=args.port)
    hybrid = HybridRetriever(bm25, dense)

    results = hybrid.rank(args.query, ticker=args.ticker, k=args.k)
    print(json.dumps(results, indent=2))
