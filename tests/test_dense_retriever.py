"""
Tests for src/retrieval/dense_retriever.py (Phase A, Day 5).

Uses a fake Qdrant client and a fake embedding model -- no real Qdrant instance or
network/huggingface access required -- to test the retriever's own logic: that it
builds the ticker filter correctly, calls search with the right args, and maps
Qdrant's response back into the shared `.rank()` chunk_id-list interface. Whether a
*real* Qdrant instance returns semantically correct neighbors is verified by actually
running this against your local Qdrant + Day 3 embeddings (see the Day 5 run
commands), not here.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from src.retrieval.dense_retriever import DenseRetriever


class FakeHit:
    def __init__(self, chunk_id, score):
        self.payload = {"chunk_id": chunk_id, "ticker": "AAPL", "item": "item_1a", "text": "..."}
        self.score = score


class FakeQueryResponse:
    """qdrant-client's query_points() returns a QueryResponse wrapping .points,
    not a bare list of hits -- the fake mirrors that shape so a test passing here
    means the real call site unwraps .points correctly."""
    def __init__(self, points):
        self.points = points


class FakeQdrantClient:
    """Records the last query_points() call so tests can assert on it.

    Deliberately exposes NO .search() attribute: qdrant-client >=1.14 removed it, and
    a fake that still offered it would let a regression back into dense_retriever.py
    without any test failing."""
    def __init__(self, hits):
        self._hits = hits
        self.last_call = None

    def query_points(self, collection_name, query, query_filter, limit):
        self.last_call = {
            "collection_name": collection_name,
            "query": query,
            "query_filter": query_filter,
            "limit": limit,
        }
        return FakeQueryResponse(self._hits[:limit])


class FakeModel:
    """Stands in for the loaded SentenceTransformer -- encode() just needs to return
    something with .tolist(), matching real model.encode()'s numpy-array output."""
    def encode(self, texts, normalize_embeddings=True, show_progress_bar=True):
        return np.array([[0.1, 0.2, 0.3] for _ in texts])


def test_rank_returns_chunk_ids_in_score_order():
    hits = [FakeHit("AAPL_0001", 0.95), FakeHit("AAPL_0002", 0.80)]
    client = FakeQdrantClient(hits)
    retriever = DenseRetriever(client=client, model=FakeModel())
    results = retriever.rank("what are apple's risks", k=5)
    assert results == ["AAPL_0001", "AAPL_0002"]


def test_rank_passes_ticker_as_qdrant_filter():
    from qdrant_client.models import Filter, FieldCondition, MatchValue
    hits = [FakeHit("AAPL_0001", 0.95)]
    client = FakeQdrantClient(hits)
    retriever = DenseRetriever(client=client, model=FakeModel())
    retriever.rank("query", ticker="AAPL", k=5)

    expected_filter = Filter(must=[FieldCondition(key="ticker", match=MatchValue(value="AAPL"))])
    assert client.last_call["query_filter"] == expected_filter


def test_rank_no_ticker_means_no_filter():
    client = FakeQdrantClient([FakeHit("AAPL_0001", 0.9)])
    retriever = DenseRetriever(client=client, model=FakeModel())
    retriever.rank("query", ticker=None, k=5)
    assert client.last_call["query_filter"] is None


def test_rank_uses_collection_name_and_limit():
    client = FakeQdrantClient([FakeHit("AAPL_0001", 0.9)])
    retriever = DenseRetriever(client=client, model=FakeModel(), collection_name="groundedrag_chunks")
    retriever.rank("query", k=7)
    assert client.last_call["collection_name"] == "groundedrag_chunks"
    assert client.last_call["limit"] == 7


def test_interface_matches_bm25_retriever_signature():
    import inspect
    from src.retrieval.bm25_retriever import BM25Retriever
    dense_sig = inspect.signature(DenseRetriever.rank)
    bm25_sig = inspect.signature(BM25Retriever.rank)
    assert list(dense_sig.parameters) == list(bm25_sig.parameters)


def test_constructing_with_explicit_model_does_not_touch_lazy_loader():
    # Passing model=FakeModel() means _get_model() must never call generate_embeddings.load_model()
    # (which would try to download from huggingface.co) -- this is the whole point of the
    # constructor accepting a pre-loaded model.
    client = FakeQdrantClient([FakeHit("AAPL_0001", 0.9)])
    retriever = DenseRetriever(client=client, model=FakeModel())
    assert retriever._get_model() is retriever._model


def test_uses_query_points_not_removed_search_api():
    """Regression test for the qdrant-client 1.19.1 breakage found on Day 5.

    `QdrantClient.search()` was deprecated in 1.10 and REMOVED by 1.14+, so the
    original dense_retriever implementation raised
    `AttributeError: 'QdrantClient' object has no attribute 'search'` the first time it
    was run against a real (1.19.1) client -- invisible until then, because every unit
    test used a fake that happily provided .search(). This test pins the call onto
    .query_points() by asserting against a fake that only implements that method.
    """
    client = FakeQdrantClient([FakeHit("AAPL_0001", 0.9)])
    retriever = DenseRetriever(client=client, model=FakeModel())
    results = retriever.rank("query", ticker="AAPL", k=3)

    assert results == ["AAPL_0001"]
    assert client.last_call is not None, "query_points() was never called"
    assert not hasattr(client, "search"), "fake must not offer the removed .search() API"


def test_unwraps_query_response_points_attribute():
    """The return value of query_points() is a QueryResponse object, not a list --
    iterating it directly (instead of .points) would raise TypeError against a real
    client, so pin the unwrapping explicitly."""
    hits = [FakeHit("AAPL_0001", 0.95), FakeHit("AAPL_0002", 0.80)]
    client = FakeQdrantClient(hits)
    retriever = DenseRetriever(client=client, model=FakeModel())
    assert retriever.rank("query", k=2) == ["AAPL_0001", "AAPL_0002"]
