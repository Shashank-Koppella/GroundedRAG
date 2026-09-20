"""
Tests for src/retrieval/bm25_retriever.py (Phase A, Day 5, stage 1).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.retrieval.bm25_retriever import BM25Retriever


def make_chunk(chunk_id, ticker, text):
    return {"chunk_id": chunk_id, "ticker": ticker, "item": "item_1a", "text": text}


CHUNKS = [
    make_chunk("AAPL_0001", "AAPL", "Apple relies on manufacturing partners concentrated in China and Taiwan for its supply chain."),
    make_chunk("AAPL_0002", "AAPL", "Apple's revenue grew year over year driven by strong iPhone demand in international markets."),
    make_chunk("MSFT_0001", "MSFT", "Microsoft's cloud business Azure faces competition from Amazon Web Services and Google Cloud."),
    make_chunk("MSFT_0002", "MSFT", "Microsoft relies on manufacturing partners for its Surface hardware line, concentrated in China."),
]


def test_rank_returns_best_matching_chunk_first():
    retriever = BM25Retriever(CHUNKS)
    results = retriever.rank("manufacturing partners concentrated in China", k=10)
    assert results[0] in ("AAPL_0001", "MSFT_0002")
    assert "AAPL_0002" not in results[:1]


def test_rank_filters_by_ticker():
    retriever = BM25Retriever(CHUNKS)
    results = retriever.rank("manufacturing partners concentrated in China", ticker="AAPL", k=10)
    assert results[0] == "AAPL_0001"
    assert "MSFT_0002" not in results  # ticker filter excludes it even though it's a strong match


def test_rank_excludes_zero_score_chunks():
    retriever = BM25Retriever(CHUNKS)
    # "quantum blockchain" appears in none of the corpus
    results = retriever.rank("quantum blockchain", k=10)
    assert results == []


def test_rank_respects_k():
    retriever = BM25Retriever(CHUNKS)
    results = retriever.rank("manufacturing partners concentrated in China revenue Azure cloud", k=2)
    assert len(results) <= 2


def test_rank_uses_same_tokenizer_as_keyword_baseline():
    from src.eval.keyword_baseline import tokenize
    assert tokenize("Apple's Q3-2024 revenue!") == ["apple", "s", "q3", "2024", "revenue"]


def test_interface_matches_keyword_baseline_signature():
    import inspect
    from src.eval.keyword_baseline import KeywordBaseline
    bm25_sig = inspect.signature(BM25Retriever.rank)
    kw_sig = inspect.signature(KeywordBaseline.rank)
    assert list(bm25_sig.parameters) == list(kw_sig.parameters)
