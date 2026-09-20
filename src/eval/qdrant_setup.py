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


def make_point_id(chunk_id: str) -> str:
    """Deterministic, globally-unique Qdrant point id derived from the chunk_id.

    BUG THIS FIXES (found Day 5, silently corrupted the Day 3 index): the original
    implementation used `id=i` from `enumerate()`. Because qdrant_setup.py is invoked
    ONCE PER COMPANY, that counter restarts at 0 on every run -- so MSFT's point 0
    overwrote AAPL's point 0, GOOGL's overwrote MSFT's, and so on. Qdrant's upsert
    treats a colliding id as an update, not an error, so all 8 uploads reported success
    while leaving only the last ~2,192 points (ORCL, plus the tail of AVGO) in a
    collection that should have held 11,245. Six of eight companies had zero vectors,
    which is invisible from the upload logs and only showed up as "dense retrieval
    contributes nothing" in the Day 5 ablation.

    uuid5 is used rather than a running integer because it is (a) globally unique across
    companies, since Day 3's make_chunk_id() already guarantees chunk_id uniqueness
    corpus-wide, and (b) idempotent -- re-uploading one company updates its own points
    in place instead of duplicating them or trampling another company's.
    """
    import uuid
    return str(uuid.uuid5(uuid.NAMESPACE_URL, chunk_id))


def upsert_chunks(client, embedded_chunks: List[Dict], batch_size: int = 256):
    from qdrant_client.models import PointStruct

    points = []
    for row in embedded_chunks:
        points.append(PointStruct(
            id=make_point_id(row["chunk_id"]),
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

    # See the note in src/retrieval/dense_retriever.py: `.search()` no longer exists on
    # qdrant-client >=1.14; `.query_points()` replaces it. This call path was never
    # exercised on Day 3 (the __main__ block below only creates the collection and
    # upserts), which is why the uploads succeeded while this function stayed broken.
    response = client.query_points(
        collection_name=COLLECTION_NAME,
        query=vec,
        query_filter=query_filter,
        limit=k,
    )
    return [(p.payload["chunk_id"], p.score, p.payload["text"][:150]) for p in response.points]


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

    # Post-upload verification. The Day 5 bug above was invisible precisely because the
    # upload reported success per-company and nobody ever asked the collection how many
    # points it actually held. Always print the running total, and warn loudly if this
    # company's chunks did not increase it by the expected amount.
    total = client.count(collection_name=COLLECTION_NAME, exact=True).count
    print(f"Collection '{COLLECTION_NAME}' now holds {total} points "
          f"(this run uploaded {len(embedded)}).")
    if total < len(embedded):
        print(f"  WARNING: collection holds fewer points ({total}) than this single "
              f"upload contained ({len(embedded)}) -- ids are colliding and overwriting.")
    print("Qdrant index stood up and populated.")
