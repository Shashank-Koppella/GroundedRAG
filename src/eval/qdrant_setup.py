"""
Stands up a local Qdrant collection and upserts embedded chunks (output of
generate_embeddings.py). Native hybrid search support (dense + sparse fusion) is why
Qdrant was chosen in Section 5 — this script only sets up the dense side for Day 3;
sparse vectors get added when BM25 fusion is built on Day 5, same collection, no migration.

Run a local Qdrant instance first:
    docker run -p 6333:6333 -p 6334:6334 qdrant/qdrant

Payload fields (ticker, item, chunk_id) are stored alongside each vector so retrieval
can filter by ticker before searching — needed for company-scoped questions in the eval
set without a separate index per ticker.
"""
import json
from pathlib import Path
from typing import List, Dict

COLLECTION_NAME = "groundedrag_chunks"
EMBEDDING_DIM = 768  # bge-base-en-v1.5 output dimension


def load_embedded_chunks(path: str) -> List[Dict]:
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def get_client(host: str = "localhost", port: int = 6333):
    from qdrant_client import QdrantClient
    return QdrantClient(host=host, port=port)


def create_collection(client, dim: int = EMBEDDING_DIM, recreate: bool = False):
    from qdrant_client.models import Distance, VectorParams

    exists = client.collection_exists(COLLECTION_NAME)
    if exists and not recreate:
        print(f"Collection '{COLLECTION_NAME}' already exists, leaving as-is (pass --recreate to rebuild).")
        return
    if exists and recreate:
        client.delete_collection(COLLECTION_NAME)

    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
    )
    print(f"Created collection '{COLLECTION_NAME}' (dim={dim}, cosine distance).")


def upsert_chunks(client, embedded_chunks: List[Dict], batch_size: int = 256):
    from qdrant_client.models import PointStruct

    points = []
    for i, row in enumerate(embedded_chunks):
        points.append(PointStruct(
            id=i,  # Qdrant needs int/UUID ids; chunk_id kept in payload for the real key
            vector=row["embedding"],
            payload={
                "chunk_id": row["chunk_id"],
                "ticker": row.get("ticker"),
                "item": row.get("item"),
                "text": row["text"],
            },
        ))

    for start in range(0, len(points), batch_size):
        batch = points[start:start + batch_size]
        client.upsert(collection_name=COLLECTION_NAME, points=batch)
        print(f"Upserted {start + len(batch)}/{len(points)}")


def sanity_search(client, model, query: str, ticker: str = None, k: int = 5):
    """Quick smoke test: embed a query and confirm the index returns something sane."""
    from qdrant_client.models import Filter, FieldCondition, MatchValue
    from generate_embeddings import embed_queries

    vec = embed_queries(model, [query])[0].tolist()
    query_filter = None
    if ticker:
        query_filter = Filter(must=[FieldCondition(key="ticker", match=MatchValue(value=ticker))])

    results = client.search(
        collection_name=COLLECTION_NAME,
        query_vector=vec,
        query_filter=query_filter,
        limit=k,
    )
    return [(r.payload["chunk_id"], r.score, r.payload["text"][:150]) for r in results]


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--embeddings", required=True, help="Path to embeddings JSONL from generate_embeddings.py")
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=6333)
    parser.add_argument("--recreate", action="store_true")
    args = parser.parse_args()

    client = get_client(args.host, args.port)
    embedded = load_embedded_chunks(args.embeddings)
    dim = len(embedded[0]["embedding"]) if embedded else EMBEDDING_DIM
    create_collection(client, dim=dim, recreate=args.recreate)
    upsert_chunks(client, embedded)
    print("Qdrant index stood up and populated.")
