"""
Equivalent passages for retrieval relevance (added Oct 5 after an audit).

The single-hop questions name a company and a topic but NOT a filing ("What does Apple identify as a
risk related to ..."). Companies repeat risk-factor and MD&A text from filing to filing, so the passage
that answers a question usually exists, word for word, in several filings. The Day 3/5 labels name one
of them; a retriever that returns another copy was scored as a miss. That understated Recall@k / MRR
and distorted the stage comparison (the reranker's apparent R@1 gain came from which copy it ranked).

Rule (fixed before any rescoring was looked at): a chunk is an EQUIVALENT of a question's gold chunk if
it belongs to the same company and its token-set Jaccard similarity with that gold chunk is >= 0.9.
Thresholds 1.0 and 0.8 are reported alongside as a sensitivity check; 0.9 is the primary number.
Equivalents are stored separately from `gold_chunk_ids` (which stay as labeled), so strict-gold and
equivalence-aware scores can both be reported.

Known limit: a passage whose answer sentence sits in the 40-word overlap between two chunks makes the
neighbouring chunk answer-bearing too, without making it lexically similar. That is only detectable by
reading (sh_011 was found this way) and is handled as a label correction, not by this rule.
"""
import re
from typing import Dict, Iterable

PRIMARY_THRESHOLD = 0.9
SENSITIVITY_THRESHOLDS = (1.0, 0.9, 0.8)
RULE = ("same company, token-set Jaccard >= {t} with a gold chunk: the same passage repeated in another "
        "filing; the question names no filing, so it answers the question equally")

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def token_set(text: str) -> frozenset:
    return frozenset(_TOKEN_RE.findall(text.lower()))


def jaccard(a: frozenset, b: frozenset) -> float:
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)


def find_equivalents(gold_ids: Iterable[str], chunks_by_id: Dict[str, Dict], threshold: float = PRIMARY_THRESHOLD,
                     _tokens: Dict[str, frozenset] = None) -> Dict[str, float]:
    """chunk_id -> best Jaccard with any gold chunk, for every non-gold chunk of the same company at or
    above `threshold`. Gold ids missing from `chunks_by_id` raise (a stale label is an error, not a skip)."""
    gold = list(dict.fromkeys(gold_ids))
    missing = [g for g in gold if g not in chunks_by_id]
    if missing:
        raise KeyError(f"gold chunk ids not in the corpus: {missing}")
    tok = _tokens if _tokens is not None else {}

    def toks(cid):
        if cid not in tok:
            tok[cid] = token_set(chunks_by_id[cid]["text"])
        return tok[cid]

    tickers = {chunks_by_id[g]["ticker"] for g in gold}
    out: Dict[str, float] = {}
    for cid, c in chunks_by_id.items():
        if cid in gold or c["ticker"] not in tickers:
            continue
        best = max(jaccard(toks(g), toks(cid)) for g in gold if chunks_by_id[g]["ticker"] == c["ticker"])
        if best >= threshold:
            out[cid] = round(best, 4)
    return dict(sorted(out.items(), key=lambda kv: (-kv[1], kv[0])))


def relevant_ids(question: Dict) -> list:
    """Gold plus stored equivalents (the equivalence-aware relevance set)."""
    return list(dict.fromkeys(list(question.get("gold_chunk_ids") or []) +
                              list(question.get("equivalent_chunk_ids") or [])))
