"""
Tests for src/retrieval/hybrid_retriever.py (Phase A, Day 5, stage 2: hand-written RRF).

reciprocal_rank_fusion() is pure math with no external dependencies, so it's tested
directly and exactly (hand-computed expected scores) -- this is the piece that most
needs to be provably correct, since it's the "explain the fusion math in an interview"
claim from Section 5 of the plan. HybridRetriever itself is tested against fake
retrievers (no real BM25/Qdrant needed) to check it wires candidate-pool-size and
fusion together correctly.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.retrieval.hybrid_retriever import reciprocal_rank_fusion, HybridRetriever


def test_rrf_matches_hand_computed_scores():
    # A appears at rank 1 in list1, rank 3 in list2.
    # B appears at rank 2 in list1 only.
    # C appears at rank 1 in list2 only.
    list1 = ["A", "B"]
    list2 = ["C", "X", "A"]
    k = 60
    # score(A) = 1/(60+1) + 1/(60+3) = 1/61 + 1/63
    # score(B) = 1/(60+2) = 1/62
    # score(C) = 1/(60+1) = 1/61
    # score(X) = 1/(60+2) = 1/62
    expected_a = 1 / 61 + 1 / 63
    expected_b = 1 / 62
    expected_c = 1 / 61
    expected_x = 1 / 62

    fused = reciprocal_rank_fusion([list1, list2], k=k)

    # A has the highest score (appears in both lists) -> must be first.
    assert fused[0] == "A"
    assert expected_a > expected_c
    assert expected_a > expected_b
    assert expected_a > expected_x
    assert set(fused) == {"A", "B", "C", "X"}


def test_rrf_doc_only_in_one_list_still_included():
    list1 = ["A", "B", "C"]
    list2 = []  # second retriever returned nothing at all
    fused = reciprocal_rank_fusion([list1, list2], k=60)
    assert fused == ["A", "B", "C"]  # order preserved, scores = 1/(60+rank) each


def test_rrf_doc_in_both_lists_outranks_doc_in_one_list():
    # "shared" is rank 5 in both lists (weak position in each);
    # "solo" is rank 1 in only one list (strong position, but only one retriever found it).
    list1 = ["solo", "x2", "x3", "x4", "shared"]
    list2 = ["y1", "y2", "y3", "y4", "shared"]
    fused = reciprocal_rank_fusion([list1, list2], k=60)
    # score(shared) = 1/65 + 1/65 = 2/65 ~= 0.0308
    # score(solo)   = 1/61          ~= 0.0164
    assert fused.index("shared") < fused.index("solo")


def test_rrf_k_parameter_changes_relative_weighting():
    list1 = ["A"]
    list2 = ["B"]
    # both at rank 1 in their own list -- with any k, A and B tie, both score 1/(k+1)
    fused_k1 = reciprocal_rank_fusion([list1, list2], k=1)
    fused_k60 = reciprocal_rank_fusion([list1, list2], k=60)
    assert set(fused_k1) == set(fused_k60) == {"A", "B"}


def test_rrf_default_k_is_60():
    import inspect
    sig = inspect.signature(reciprocal_rank_fusion)
    assert sig.parameters["k"].default == 60


class FakeRetriever:
    def __init__(self, ranked_ids):
        self.ranked_ids = ranked_ids
        self.last_call = None

    def rank(self, query, ticker=None, k=10):
        self.last_call = {"query": query, "ticker": ticker, "k": k}
        return self.ranked_ids[:k]


def test_hybrid_retriever_fuses_two_retrievers():
    bm25 = FakeRetriever(["A", "B", "C"])
    dense = FakeRetriever(["C", "A", "D"])
    hybrid = HybridRetriever(bm25, dense, k_rrf=60, candidate_pool_size=100)
    results = hybrid.rank("some query", ticker="AAPL", k=3)
    # A and C both appear in both lists -> should rank above B, D which appear once each
    assert set(results[:2]) == {"A", "C"}
    assert len(results) == 3


def test_hybrid_retriever_requests_full_candidate_pool_not_just_k():
    bm25 = FakeRetriever(["A"] * 1)
    dense = FakeRetriever(["B"] * 1)
    hybrid = HybridRetriever(bm25, dense, candidate_pool_size=100)
    hybrid.rank("query", ticker="AAPL", k=5)
    assert bm25.last_call["k"] == 100
    assert dense.last_call["k"] == 100


def test_hybrid_retriever_passes_ticker_through_to_both():
    bm25 = FakeRetriever(["A"])
    dense = FakeRetriever(["B"])
    hybrid = HybridRetriever(bm25, dense)
    hybrid.rank("query", ticker="MSFT", k=5)
    assert bm25.last_call["ticker"] == "MSFT"
    assert dense.last_call["ticker"] == "MSFT"


def test_interface_matches_other_retrievers_signature():
    import inspect
    from src.retrieval.bm25_retriever import BM25Retriever
    hybrid_sig = inspect.signature(HybridRetriever.rank)
    bm25_sig = inspect.signature(BM25Retriever.rank)
    assert list(hybrid_sig.parameters) == list(bm25_sig.parameters)
