"""
Generates dense embeddings for all chunks using BAAI/bge-base-en-v1.5, per Section 5's
tool-stack choice.

NOTE ON WHERE TO RUN THIS: this sandbox's network allowlist doesn't include
huggingface.co, so the model weights can't actually download from inside this chat's
container — this script is correct and ready to run, but run it in your own
environment (local machine or Colab) where the download can happen. Installing the
`sentence-transformers` package itself (from PyPI) works fine in either place.

bge models expect a query-side instruction prefix for asymmetric retrieval
("represent this sentence for searching relevant passages: ") — passages themselves are
embedded plain. Getting this wrong is a common, silent quality hit: queries and passages
end up in slightly different embedding spaces and Recall@k looks mysteriously mediocre
with no error thrown. Baked into embed_queries() vs embed_passages() below so it can't be
forgotten later during Day 5's hybrid build.
"""
import json
from pathlib import Path
from typing import List, Dict

QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "
MODEL_NAME = "BAAI/bge-base-en-v1.5"


def load_model():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(MODEL_NAME)


def embed_passages(model, texts: List[str]):
    # Passages: no instruction prefix.
    return model.encode(texts, normalize_embeddings=True, show_progress_bar=True)


def embed_queries(model, texts: List[str]):
    # Queries: bge's asymmetric-retrieval instruction prefix.
    prefixed = [QUERY_INSTRUCTION + t for t in texts]
    return model.encode(prefixed, normalize_embeddings=True, show_progress_bar=True)


def load_chunks(chunks_path: str) -> List[Dict]:
    chunks = []
    with open(chunks_path) as f:
        for line in f:
            line = line.strip()
            if line:
                chunks.append(json.loads(line))
    return chunks


def main(chunks_path: str, out_path: str):
    chunks = load_chunks(chunks_path)
    print(f"Loaded {len(chunks)} chunks from {chunks_path}")

    model = load_model()
    texts = [c["text"] for c in chunks]
    embeddings = embed_passages(model, texts)

    out = []
    for chunk, vec in zip(chunks, embeddings):
        out.append({
            "chunk_id": chunk["chunk_id"],
            "ticker": chunk.get("ticker"),
            "item": chunk.get("item"),
            "text": chunk["text"],
            "embedding": vec.tolist(),
        })

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        for row in out:
            f.write(json.dumps(row) + "\n")
    print(f"Wrote {len(out)} embedded chunks to {out_path}")
    print(f"Embedding dim: {len(out[0]['embedding']) if out else 'n/a'}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--chunks", required=True, help="Path to processed chunks JSONL")
    parser.add_argument("--out", default="data/processed/embeddings.jsonl")
    args = parser.parse_args()
    main(args.chunks, args.out)
