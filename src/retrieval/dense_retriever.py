"""
Dense (Qdrant) retriever -- Phase A, Day 5, dense side of the RRF hybrid fusion
(Section 5/7 of the plan). Wraps the existing local Qdrant collection stood up by
`src/eval/qdrant_setup.py` in Day 3 (`groundedrag_chunks`, bge-base-en-v1.5, dim 768,
cosine distance) and the query-side embedding function from `generate_embeddings.py`
(the asymmetric bge instruction prefix, so query/passage embeddings share a space).

Exposes the same `.rank(query, ticker=None, k=10)` interface as KeywordBaseline and
BM25Retriever, so score_retrieval()/score_sql_routing_sanity() run unchanged, and so
hybrid_retriever.py can fuse this with BM25Retriever's ranked list without caring which
retriever produced which list.

NOTE: this retriever needs a running local Qdrant instance and a downloaded
bge-base-en-v1.5 model -- neither is reachable from inside a Claude cloud/agent
sandbox (Qdrant is bound to your machine's own localhost; huggingface.co is not on
the sandbox's network allowlist). Run this against your own local Qdrant + venv,
where Docker and the HF download both work -- see the Day 5 run commands in the
session's final summary for the exact commands.
"""
import json
from pathlib import Path
from typing import List, Dict, Optional

COLLECTION_NAME = "groundedrag_chunks"


class DenseRetriever:
    """Qdrant-backed dense retriever. `.rank(query, ticker=None, k=10)` returns
    chunk_ids ordered best-first (highest cosine similarity first)."""

    def __init__(self, client=None, model=None, host: str = "localhost", port: int = 6333,
                 collection_name: str = COLLECTION_NAME):
        """
        client: an existing qdrant_client.QdrantClient, or None to create one from
                host/port (matches src/eval/qdrant_setup.py's get_client()).
        model:  an already-loaded sentence_transformers.SentenceTransformer
                (BAAI/bge-base-en-v1.5), or None to load it lazily on first .rank()
                call, so constructing a DenseRetriever doesn't force a model load
                (and doesn't force a huggingface.co hit) if it's only used for its
                interface shape in a test.
        """
        if client is None:
            from qdrant_client import QdrantClient
            client = QdrantClient(host=host, port=port)
        self.client = client
        self._model = model
        self.collection_name = collection_name

    def _get_model(self):
        if self._model is None:
            import sys
            sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src" / "eval"))
            from generate_embeddings import load_model
            self._model = load_model()
        return self._model

    def rank(self, query: str, ticker: Optional[str] = None, k: int = 10) -> List[str]:
        import sys
        sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src" / "eval"))
        from generate_embeddings import embed_queries
        from qdrant_client.models import Filter, FieldCondition, MatchValue

        model = self._get_model()
        vec = embed_queries(model, [query])[0].tolist()

        query_filter = None
        if ticker:
            query_filter = Filter(must=[FieldCondition(key="ticker", match=MatchValue(value=ticker))])

        # qdrant-client removed `.search()` (deprecated in 1.10, gone by 1.14+) in favour
        # of `.query_points()`. Two differences that matter: the vector argument is named
        # `query`, not `query_vector`, and the return value is a QueryResponse wrapping
        # `.points` rather than a bare list of hits. requirements.txt pins
        # qdrant-client>=1.14 so this can't silently break against an older client.
        response = self.client.query_points(
            collection_name=self.collection_name,
            query=vec,
            query_filter=query_filter,
            limit=k,
        )
        return [p.payload["chunk_id"] for p in response.points]


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", required=True)
    parser.add_argument("--ticker", default=None)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=6333)
    args = parser.parse_args()

    retriever = DenseRetriever(host=args.host, port=args.port)
    results = retriever.rank(args.query, ticker=args.ticker, k=args.k)
    print(json.dumps(results, indent=2))
