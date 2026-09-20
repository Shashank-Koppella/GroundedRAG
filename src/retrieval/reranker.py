"""
Baseline cross-encoder reranker -- Phase A, Day 5, stage 3 of the hybrid retrieval
ablation (Section 5/7 of the plan): BM25 -> RRF hybrid -> reranker on top.

Uses `cross-encoder/ms-marco-MiniLM-L-6-v2` (Section 5's tool-stack choice: free,
fast, CPU-friendly) to rescore a candidate pool from an upstream retriever, and is
also the model that gets LoRA-fine-tuned in Phase C -- this file is the baseline the
Phase C before/after comparison is measured against, so its interface needs to stay
stable across that later change (the LoRA-adapted version should be a drop-in
CrossEncoder swap, not a rewrite of this wrapper).

Wraps any upstream retriever exposing `.rank(query, ticker=None, k=int) -> List[chunk_id]`
(HybridRetriever today) and re-scores its top `candidate_pool_size` results with the
cross-encoder, which scores a (query, passage_text) pair jointly -- strictly more
expressive than either BM25's term overlap or dense's independently-encoded cosine
similarity, at the cost of being too slow to run over the full corpus (hence "rerank
a shortlist" rather than "replace retrieval").

RerankerRetriever itself exposes the same `.rank(query, ticker=None, k=10)` interface
as every other retriever in this package, so score_retrieval() runs against it
unchanged, and Day 5's 4-stage ablation table (keyword baseline / BM25 / RRF hybrid /
+reranker) is scored identically at every stage.

NOTE: like dense_retriever.py, loading `cross-encoder/ms-marco-MiniLM-L-6-v2` needs a
huggingface.co download, which is not reachable from inside a Claude cloud/agent
sandbox. Run this against your own local venv -- see the Day 5 run commands.
"""
import json
from pathlib import Path
from typing import List, Dict, Optional

MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"


def load_reranker_model():
    from sentence_transformers import CrossEncoder
    return CrossEncoder(MODEL_NAME)


class RerankerRetriever:
    """Reranks an upstream retriever's candidate pool with a cross-encoder.
    `.rank(query, ticker=None, k=10)` returns chunk_ids ordered by cross-encoder
    score, best-first."""

    def __init__(self, upstream_retriever, chunks_by_id: Dict[str, Dict],
                 model=None, candidate_pool_size: int = 50):
        """
        upstream_retriever: anything exposing `.rank(query, ticker, k)`, typically
                             HybridRetriever, whose top candidate_pool_size results
                             get reranked.
        chunks_by_id:        {chunk_id: chunk_dict} for the full corpus, needed to
                              look up each candidate's text for the cross-encoder --
                              upstream retrievers only return chunk_ids, not text.
        model:                an already-loaded sentence_transformers.CrossEncoder,
                              or None to lazy-load load_reranker_model() on first use.
        """
        self.upstream_retriever = upstream_retriever
        self.chunks_by_id = chunks_by_id
        self._model = model
        self.candidate_pool_size = candidate_pool_size

    def _get_model(self):
        if self._model is None:
            self._model = load_reranker_model()
        return self._model

    def rank(self, query: str, ticker: Optional[str] = None, k: int = 10) -> List[str]:
        candidates = self.upstream_retriever.rank(query, ticker=ticker, k=self.candidate_pool_size)
        if not candidates:
            return []

        model = self._get_model()
        pairs = [(query, self.chunks_by_id[cid]["text"]) for cid in candidates if cid in self.chunks_by_id]
        valid_candidates = [cid for cid in candidates if cid in self.chunks_by_id]

        scores = model.predict(pairs)
        scored = list(zip(valid_candidates, scores))
        scored.sort(key=lambda x: x[1], reverse=True)
        return [chunk_id for chunk_id, _ in scored[:k]]


if __name__ == "__main__":
    import argparse
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from src.eval.keyword_baseline import load_chunks
    from src.retrieval.bm25_retriever import BM25Retriever
    from src.retrieval.dense_retriever import DenseRetriever
    from src.retrieval.hybrid_retriever import HybridRetriever

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
    chunks_by_id = {c["chunk_id"]: c for c in all_chunks}

    bm25 = BM25Retriever(all_chunks)
    dense = DenseRetriever(host=args.host, port=args.port)
    hybrid = HybridRetriever(bm25, dense)
    reranker = RerankerRetriever(hybrid, chunks_by_id)

    results = reranker.rank(args.query, ticker=args.ticker, k=args.k)
    print(json.dumps(results, indent=2))
