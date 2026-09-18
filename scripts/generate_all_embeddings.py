"""
Runs generate_embeddings.py's logic across all 8 companies in one call, loading the
bge-base-en-v1.5 model ONCE instead of once per ticker (calling main() eight times
separately would reload ~440MB of model weights eight times — this loads it once and
reuses it, which matters since model load is the slow part of each individual run).

Usage (run from repo root):

    python scripts/generate_all_embeddings.py

Skips any ticker whose chunks file doesn't exist yet (prints a warning, keeps going)
rather than failing the whole run — useful if you're re-running after fixing one
company's ingestion (e.g. re-running just ORCL after the section_splitter fix) without
wanting to wait for all 8 to regenerate.

Writes data/processed/embeddings/{TICKER}.jsonl for each company found, same format
generate_embeddings.py's main() writes — src/eval/qdrant_setup.py reads that format
directly, no changes needed there.
"""
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
# Must come before the src.ingestion import below. Running this as `python
# scripts/generate_all_embeddings.py` (a direct file path) does NOT put the repo root on
# sys.path automatically — only `python -m scripts.generate_all_embeddings` does that.
# Inserting it explicitly here makes the script work either way, rather than silently
# depending on how the person happens to invoke it.
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src" / "eval"))

from generate_embeddings import load_model, embed_passages, load_chunks  # noqa: E402
from src.ingestion.config import COMPANIES  # noqa: E402

CHUNKS_DIR = REPO_ROOT / "data" / "processed" / "chunks"
EMBEDDINGS_DIR = REPO_ROOT / "data" / "processed" / "embeddings"


def embed_one_ticker(model, ticker: str) -> dict:
    chunks_path = CHUNKS_DIR / f"{ticker}.jsonl"
    if not chunks_path.exists():
        print(f"[{ticker}] SKIPPED — no chunks file at {chunks_path}")
        return {"ticker": ticker, "status": "skipped", "n_chunks": 0}

    start = time.monotonic()
    chunks = load_chunks(str(chunks_path))
    if not chunks:
        print(f"[{ticker}] SKIPPED — chunks file exists but is empty")
        return {"ticker": ticker, "status": "empty", "n_chunks": 0}

    texts = [c["text"] for c in chunks]
    embeddings = embed_passages(model, texts)

    out_rows = []
    for chunk, vec in zip(chunks, embeddings):
        out_rows.append({
            "chunk_id": chunk["chunk_id"],
            "ticker": chunk.get("ticker"),
            "item": chunk.get("item"),
            "text": chunk["text"],
            "embedding": vec.tolist(),
        })

    EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = EMBEDDINGS_DIR / f"{ticker}.jsonl"
    with open(out_path, "w") as f:
        for row in out_rows:
            f.write(json.dumps(row) + "\n")

    elapsed = time.monotonic() - start
    print(f"[{ticker}] wrote {len(out_rows)} embedded chunks -> {out_path} ({elapsed:.1f}s)")
    return {"ticker": ticker, "status": "ok", "n_chunks": len(out_rows), "elapsed_s": elapsed}


def main():
    print(f"Loading {'-'.join(['bge', 'base', 'en', 'v1.5'])} model (one-time cost for this whole run)...")
    model_start = time.monotonic()
    model = load_model()
    print(f"Model loaded in {time.monotonic() - model_start:.1f}s\n")

    results = []
    for ticker in COMPANIES:
        results.append(embed_one_ticker(model, ticker))

    print("\n=== Summary ===")
    ok = [r for r in results if r["status"] == "ok"]
    skipped = [r for r in results if r["status"] != "ok"]
    total_chunks = sum(r["n_chunks"] for r in ok)
    total_time = sum(r.get("elapsed_s", 0) for r in ok)
    for r in results:
        print(f"  {r['ticker']:6s} {r['status']:8s} {r['n_chunks']:6d} chunks")
    print(f"\n{len(ok)}/8 companies embedded, {total_chunks} total chunks, {total_time:.1f}s total")
    if skipped:
        print(f"Skipped: {[r['ticker'] for r in skipped]} — run src.ingestion.pipeline first if these are missing")


if __name__ == "__main__":
    main()
