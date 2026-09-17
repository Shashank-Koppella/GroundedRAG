"""
Trivial keyword-search baseline. This exists purely as a floor to compare later
retrievers (BM25, dense, hybrid+RRF, reranked) against — Section 6/7 of the plan calls
for exactly this. It is deliberately dumb: raw term-overlap count, no IDF weighting,
no stemming. If a real retriever can't beat this, something is wrong with it.
"""
import json
import re
from pathlib import Path
from typing import List, Dict

TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> List[str]:
    return TOKEN_RE.findall(text.lower())


def load_chunks(chunks_path: str) -> List[Dict]:
    """
    Expects JSONL matching src/ingestion/pipeline.py's real output schema:
    {"chunk_id": ..., "ticker": ..., "item": ..., "text": ..., "start_word": ...,
     "end_word": ..., "form": ..., "filing_date": ..., "accession_number": ..., "cik": ...}
    """
    chunks = []
    with open(chunks_path) as f:
        for line in f:
            line = line.strip()
            if line:
                chunks.append(json.loads(line))
    return chunks


class KeywordBaseline:
    """Raw term-overlap ranking. `.rank(query, ticker=None, k=10)` returns chunk_ids
    ordered best-first."""

    def __init__(self, chunks: List[Dict]):
        self.chunks = chunks
        self._tokenized = [
            (c["chunk_id"], c.get("ticker"), set(tokenize(c.get("text", ""))))
            for c in chunks
        ]

    def rank(self, query: str, ticker: str = None, k: int = 10) -> List[str]:
        q_terms = set(tokenize(query))
        pool = self._tokenized
        if ticker:
            pool = [t for t in pool if t[1] == ticker]
        scored = [(chunk_id, len(q_terms & terms)) for chunk_id, _, terms in pool]
        scored.sort(key=lambda x: x[1], reverse=True)
        return [chunk_id for chunk_id, score in scored[:k] if score > 0]


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--chunks", required=True, help="Path to chunks JSONL")
    parser.add_argument("--query", required=True)
    parser.add_argument("--ticker", default=None)
    parser.add_argument("--k", type=int, default=10)
    args = parser.parse_args()

    chunks = load_chunks(args.chunks)
    baseline = KeywordBaseline(chunks)
    results = baseline.rank(args.query, ticker=args.ticker, k=args.k)
    print(json.dumps(results, indent=2))
