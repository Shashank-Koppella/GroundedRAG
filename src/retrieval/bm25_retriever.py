"""
BM25 retriever — Phase A, Day 5, stage 1 of the hybrid-retrieval ablation
(Section 5/7 of the plan: BM25 baseline -> RRF fusion with dense -> reranker).

Uses `rank_bm25.BM25Okapi` (already in requirements.txt, unused until now) over the
same chunk JSONL schema as `keyword_baseline.py` (pipeline.py's real output:
chunk_id/ticker/item/text/...). Exposes the same `.rank(query, ticker=None, k=10)`
interface as KeywordBaseline so `src/eval/scoring.py`'s `score_retrieval()` and
`score_sql_routing_sanity()` run against it completely unchanged.

Tokenization matches keyword_baseline.py's `tokenize()` exactly (lowercase,
`[a-z0-9]+`) so a Recall@k delta between the keyword baseline and BM25 reflects
BM25's actual ranking function (TF saturation + length normalization + IDF), not a
difference in what counts as a token.

BM25 is corpus-dependent: fitting one BM25Okapi index over ALL companies' chunks
means IDF is computed across the full 8-company corpus. This matters for ticker-scoped
queries (most of the eval set) — a term that's rare company-wide but common within one
company's filings still gets a global IDF weight. This is the standard/expected way to
use BM25 with post-hoc filtering (matches how the dense retriever will also filter
after a global similarity search, so the two stages are comparable in Day 5's ablation)
rather than building a separate index per ticker, which would fragment IDF statistics
across already-small per-company corpora instead.
"""
import json
from pathlib import Path
from typing import List, Dict

from rank_bm25 import BM25Okapi

# Reuse keyword_baseline's tokenizer definition directly (not just the same regex
# hand-copied) so the two retrievers can never silently drift out of sync.
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src" / "eval"))
from keyword_baseline import tokenize, load_chunks  # noqa: E402


class BM25Retriever:
    """BM25Okapi over the full chunk corpus. `.rank(query, ticker=None, k=10)`
    returns chunk_ids ordered best-first, matching KeywordBaseline's interface."""

    def __init__(self, chunks: List[Dict]):
        self.chunks = chunks
        self._chunk_ids = [c["chunk_id"] for c in chunks]
        self._tickers = [c.get("ticker") for c in chunks]
        self._tokenized_corpus = [tokenize(c.get("text", "")) for c in chunks]
        self._bm25 = BM25Okapi(self._tokenized_corpus)

    def rank(self, query: str, ticker: str = None, k: int = 10) -> List[str]:
        q_tokens = tokenize(query)
        scores = self._bm25.get_scores(q_tokens)

        indices = range(len(self._chunk_ids))
        if ticker:
            indices = [i for i in indices if self._tickers[i] == ticker]

        scored = [(self._chunk_ids[i], scores[i]) for i in indices]
        # BM25 scores are non-negative but a chunk with zero query-term overlap
        # still gets score 0.0, not a negative/undefined score — drop those explicitly
        # (matches KeywordBaseline's `score > 0` filter) so "not retrieved" behaves the
        # same way across both retrievers for Recall@k purposes.
        scored = [(cid, s) for cid, s in scored if s > 0]
        scored.sort(key=lambda x: x[1], reverse=True)
        return [chunk_id for chunk_id, _ in scored[:k]]


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--chunks-dir", required=True,
                         help="Directory of per-ticker chunk JSONL files, e.g. data/processed/chunks/")
    parser.add_argument("--query", required=True)
    parser.add_argument("--ticker", default=None)
    parser.add_argument("--k", type=int, default=10)
    args = parser.parse_args()

    all_chunks = []
    for f in sorted(Path(args.chunks_dir).glob("*.jsonl")):
        all_chunks.extend(load_chunks(str(f)))

    retriever = BM25Retriever(all_chunks)
    results = retriever.rank(args.query, ticker=args.ticker, k=args.k)
    print(json.dumps(results, indent=2))
