"""
Tests for src/retrieval/reranker.py (Phase A, Day 5, stage 3: baseline cross-encoder
reranker). Uses a fake upstream retriever and a fake cross-encoder model -- no real
Qdrant/BM25 corpus or huggingface.co access required -- to test the wrapper's own
logic: it reranks by predicted score (not upstream order), looks up text correctly,
and truncates to k. Whether the real ms-marco-MiniLM-L-6-v2 model's scores actually
improve Recall@k/MRR is verified by running this against the real corpus (see the
Day 5 run commands), not here.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.retrieval.reranker import RerankerRetriever


class FakeUpstreamRetriever:
    def __init__(self, candidate_ids):
        self.candidate_ids = candidate_ids
        self.last_call = None

    def rank(self, query, ticker=None, k=10):
        self.last_call = {"query": query, "ticker": ticker, "k": k}
        return self.candidate_ids[:k]


class FakeCrossEncoder:
    """Scores pairs by a fixed lookup table keyed on passage text, so tests can
    control exactly which candidate should end up ranked first."""
    def __init__(self, score_by_text):
        self.score_by_text = score_by_text
        self.last_pairs = None

    def predict(self, pairs):
        self.last_pairs = pairs
        return [self.score_by_text[text] for _, text in pairs]


CHUNKS_BY_ID = {
    "A": {"chunk_id": "A", "text": "weak match text"},
    "B": {"chunk_id": "B", "text": "strong match text"},
    "C": {"chunk_id": "C", "text": "medium match text"},
}


def test_rerank_reorders_by_cross_encoder_score_not_upstream_order():
    # Upstream (BM25/hybrid) ranked A first, but the cross-encoder should think B is best.
    upstream = FakeUpstreamRetriever(["A", "B", "C"])
    model = FakeCrossEncoder({"weak match text": 0.1, "strong match text": 0.9, "medium match text": 0.5})
    reranker = RerankerRetriever(upstream, CHUNKS_BY_ID, model=model, candidate_pool_size=10)

    results = reranker.rank("query", k=10)
    assert results == ["B", "C", "A"]


def test_rerank_respects_k_after_reranking():
    upstream = FakeUpstreamRetriever(["A", "B", "C"])
    model = FakeCrossEncoder({"weak match text": 0.1, "strong match text": 0.9, "medium match text": 0.5})
    reranker = RerankerRetriever(upstream, CHUNKS_BY_ID, model=model, candidate_pool_size=10)

    results = reranker.rank("query", k=2)
    assert results == ["B", "C"]


def test_rerank_requests_candidate_pool_size_from_upstream_not_final_k():
    upstream = FakeUpstreamRetriever(["A", "B", "C"])
    model = FakeCrossEncoder({"weak match text": 0.1, "strong match text": 0.9, "medium match text": 0.5})
    reranker = RerankerRetriever(upstream, CHUNKS_BY_ID, model=model, candidate_pool_size=50)

    reranker.rank("query", k=2)
    assert upstream.last_call["k"] == 50


def test_rerank_passes_ticker_through_to_upstream():
    upstream = FakeUpstreamRetriever(["A"])
    model = FakeCrossEncoder({"weak match text": 0.1})
    reranker = RerankerRetriever(upstream, CHUNKS_BY_ID, model=model)
    reranker.rank("query", ticker="AAPL", k=5)
    assert upstream.last_call["ticker"] == "AAPL"


def test_rerank_builds_query_passage_pairs_correctly():
    upstream = FakeUpstreamRetriever(["A", "B"])
    model = FakeCrossEncoder({"weak match text": 0.1, "strong match text": 0.9})
    reranker = RerankerRetriever(upstream, CHUNKS_BY_ID, model=model)
    reranker.rank("what is the risk", k=5)
    assert model.last_pairs == [("what is the risk", "weak match text"), ("what is the risk", "strong match text")]


def test_rerank_empty_upstream_returns_empty():
    upstream = FakeUpstreamRetriever([])
    model = FakeCrossEncoder({})
    reranker = RerankerRetriever(upstream, CHUNKS_BY_ID, model=model)
    assert reranker.rank("query", k=5) == []


def test_rerank_skips_candidate_ids_missing_from_chunks_by_id():
    # Defensive: an upstream retriever could in principle return a chunk_id not in
    # the local chunks_by_id map (e.g. stale index) -- must not KeyError.
    upstream = FakeUpstreamRetriever(["A", "GHOST_ID"])
    model = FakeCrossEncoder({"weak match text": 0.1})
    reranker = RerankerRetriever(upstream, CHUNKS_BY_ID, model=model)
    results = reranker.rank("query", k=5)
    assert results == ["A"]


def test_interface_matches_other_retrievers_signature():
    import inspect
    from src.retrieval.bm25_retriever import BM25Retriever
    rerank_sig = inspect.signature(RerankerRetriever.rank)
    bm25_sig = inspect.signature(BM25Retriever.rank)
    assert list(rerank_sig.parameters) == list(bm25_sig.parameters)
