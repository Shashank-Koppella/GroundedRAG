"""
Phase A, Day 6 -- mine and categorize hard negatives from Day 5's recall-ceiling
diagnostic for Phase C's LoRA reranker fine-tune (Day 10).

Source: data/eval_set/day5_ceiling_diagnostic.json's `hard_negatives` field per
single-hop question -- chunks the RRF-fused retriever (pre-reranker) ranked ABOVE
the gold chunk. These are the retriever's actual, real mistakes, not synthetic
negatives, which is what makes them worth training on.

Day 5's plan-flagged hypothesis (from sh_011, the AMZN AWS operating-income
question): the retriever sometimes finds the right NEIGHBORHOOD -- the chunk
immediately before/after the gold chunk, same filing, same item section -- but
not the specific sentence-bearing chunk. That's a chunking-granularity failure,
distinct from "finds nothing relevant at all" (a genuine retrieval failure), and
the standing instruction for Day 6 was to check whether other questions show the
same pattern before assuming sh_011 was a one-off, and to distinguish failure
types explicitly rather than treating every hard negative as one undifferentiated
bucket.

This script answers that by categorizing every mined hard negative, per question,
into one of four types (checked in this priority order, since a hard negative can
match more than one):

  1. neighborhood_miss      -- same ticker+filing(accession)+item section as a gold
                                chunk, chunk_index within +/-1. This IS the sh_011
                                pattern: the retriever found the right neighborhood.
  2. cross_filing_confusion -- same ticker+item section as a gold chunk, but a
                                DIFFERENT filing (different accession number, i.e.
                                the right topic from a different fiscal year/quarter).
  3. topic_drift            -- same ticker as a gold chunk, but a different item
                                section entirely (right company, wrong topic).
  4. off_topic / same_doc_distant -- everything else: a different ticker, or the
                                same filing+item as gold but more than 1 chunk away
                                (same document, but not a real near-miss).

A question's own pattern label is genuine_retrieval_failure if the fused retriever
never found gold at all within the Day 5 diagnostic's DEEP_K=200 search (hybrid_rank
is null) -- distinct from any hard-negative categorization, since there IS no
"above gold" list to mine in that case; neighborhood_miss if ANY of its hard
negatives are a true +/-1 adjacency (the sh_011-type mistake); otherwise whichever
of cross_filing_confusion / topic_drift accounts for at least half its hard
negatives, or mixed if neither does.

Output: data/eval_set/day6_hard_negatives.json -- per-question categorization plus
a flat list of (query, positive, hard_negative, category, sample_weight) training
triples for Phase C, with neighborhood_miss triples given a higher sample_weight
(deliberately oversampled per the Day 6 standing instruction, since it is the
hardest and most valuable failure mode to train against, even though it is rare
in this eval set) and genuine_retrieval_failure questions excluded from the
triples entirely (there is no valid "above gold" hard negative to mine when gold
was never found -- including one would mean training against the wrong signal).

Usage (run from repo root; no Qdrant/network needed -- pure JSON transform over
already-computed Day 5 output plus the chunk files):
    python -m scripts.mine_hard_negatives
Writes data/eval_set/day6_hard_negatives_v2.json (the original day6_hard_negatives.json is kept as the
record of what Day 6 reported).

OCT 5 AUDIT CHANGES (v2)
  * False negatives are removed. A "hard negative" that is itself relevant -- a gold id, a stored
    equivalent (same passage in another filing, src/eval/equivalence.py), or token-set Jaccard >=
    FALSE_NEGATIVE_JACCARD with a gold chunk -- is recorded as `excluded_false_negative` and never
    becomes a training triple. v1 trained against answer-bearing text, including sh_011's _0025 (which
    contains the answer sentence) at 3x weight.
  * Every relevant id is listed as a positive, not just gold_ids[0].
  * THE OUTPUT IS DIAGNOSTIC ONLY, NOT TRAINING DATA FOR AN EVAL ON THIS SET. Every triple's query is
    an eval question, so a reranker fine-tuned on these and then scored on eval_questions.json would be
    scored on its own training data. Phase C must mine from questions disjoint from the eval set; see
    `assert_disjoint_from_eval`.
"""
import json
import re
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

NEIGHBOR_WINDOW = 1  # strict +/-1 chunk adjacency = the sh_011 pattern
FALSE_NEGATIVE_JACCARD = 0.8  # a "negative" this similar to a gold chunk is very likely answer-bearing
NEIGHBORHOOD_MISS_WEIGHT = 3.0  # deliberate oversampling for Phase C training
DEFAULT_WEIGHT = 1.0

CHUNK_ID_RE = re.compile(r"^([A-Z]+)_(\d+)_(item_[0-9a-z]+)_(\d+)$")

CATEGORY_DEFINITIONS = {
    "neighborhood_miss": (
        "Same ticker+filing+item section as a gold chunk, chunk_index within +/-1. "
        "The retriever found the right neighborhood but not the sentence-bearing "
        "chunk -- the sh_011 pattern, a chunking-granularity failure."
    ),
    "cross_filing_confusion": (
        "Same ticker+item section as a gold chunk, different filing (accession) -- "
        "right topic, wrong fiscal year/quarter."
    ),
    "topic_drift": (
        "Same ticker as a gold chunk, different item section -- right company, "
        "wrong topic."
    ),
    "same_doc_distant": (
        "Same ticker+filing+item section as a gold chunk, but more than 1 chunk "
        "away -- same document, not a real near-miss."
    ),
    "off_topic": "Different ticker entirely from every gold chunk for this question.",
}


def parse_chunk_id(chunk_id: str):
    """AAPL_000032019324000123_item_1a_0010 -> (AAPL, '000032019324000123', 'item_1a', 10)"""
    m = CHUNK_ID_RE.match(chunk_id)
    if not m:
        return None
    ticker, accession, item, idx = m.groups()
    return ticker, accession, item, int(idx)


def categorize_negative(neg_id: str, gold_parsed: list) -> str:
    p = parse_chunk_id(neg_id)
    if p is None:
        return "off_topic"
    ticker, accession, item, idx = p

    for gt, gacc, gitem, gidx in gold_parsed:
        if ticker == gt and accession == gacc and item == gitem:
            if abs(idx - gidx) <= NEIGHBOR_WINDOW:
                return "neighborhood_miss"

    for gt, gacc, gitem, gidx in gold_parsed:
        if ticker == gt and item == gitem and accession != gacc:
            return "cross_filing_confusion"

    for gt, gacc, gitem, gidx in gold_parsed:
        if ticker == gt and item != gitem:
            return "topic_drift"

    for gt, gacc, gitem, gidx in gold_parsed:
        if ticker == gt and accession == gacc and item == gitem:
            return "same_doc_distant"

    return "off_topic"


def classify_question(hybrid_rank, per_negative_categories: list) -> str:
    if hybrid_rank is None:
        return "genuine_retrieval_failure"
    if not per_negative_categories:
        return "no_hard_negatives"
    counts = Counter(per_negative_categories)
    if counts["neighborhood_miss"] > 0:
        return "neighborhood_miss"
    total = len(per_negative_categories)
    if counts["cross_filing_confusion"] >= total * 0.5:
        return "cross_filing_confusion"
    if counts["topic_drift"] >= total * 0.5:
        return "topic_drift"
    return "mixed"


def assert_disjoint_from_eval(train_question_ids, eval_question_ids) -> None:
    """Phase C guard: training questions must not be evaluation questions."""
    overlap = sorted(set(train_question_ids) & set(eval_question_ids))
    if overlap:
        raise ValueError(f"{len(overlap)} training questions are eval questions (train/test contamination): "
                         f"{overlap[:5]}...")


def is_false_negative(neg_id: str, relevant: list, chunks_by_id: dict, tokens: dict) -> bool:
    if neg_id in relevant:
        return True
    if not chunks_by_id or neg_id not in chunks_by_id:
        return False
    from src.eval.equivalence import jaccard, token_set
    for cid in (neg_id, *relevant):
        if cid in chunks_by_id and cid not in tokens:
            tokens[cid] = token_set(chunks_by_id[cid]["text"])
    return any(jaccard(tokens[neg_id], tokens[g]) >= FALSE_NEGATIVE_JACCARD for g in relevant if g in tokens)


def mine(diagnostic_rows: list, questions_by_id: dict, chunks_by_id: dict = None) -> dict:
    per_question = []
    triples = []
    tokens = {}

    for row in diagnostic_rows:
        qid = row["id"]
        q = questions_by_id[qid]
        gold_ids = q["gold_chunk_ids"]
        relevant = list(dict.fromkeys(gold_ids + list(q.get("equivalent_chunk_ids") or [])))
        gold_parsed = [parse_chunk_id(g) for g in gold_ids]
        all_negs = row["hard_negatives"]
        excluded = [n for n in all_negs if is_false_negative(n, relevant, chunks_by_id, tokens)]
        hard_negs = [n for n in all_negs if n not in excluded]

        neg_categories = [categorize_negative(n, gold_parsed) for n in hard_negs]
        pattern = classify_question(row["hybrid_rank"], neg_categories)

        per_question.append({
            "id": qid,
            "ticker": row["ticker"],
            "hybrid_rank": row["hybrid_rank"],
            "in_reranker_pool_top50": row["in_reranker_pool_top50"],
            "n_hard_negatives": len(hard_negs),
            "excluded_false_negatives": excluded,
            "category_counts": dict(Counter(neg_categories)),
            "pattern": pattern,
        })

        if pattern == "genuine_retrieval_failure":
            # No valid "above gold" hard negative exists when gold was never found
            # in the fused retriever's DEEP_K=200 search -- nothing to mine here.
            continue

        for neg_id, category in zip(hard_negs, neg_categories):
            weight = NEIGHBORHOOD_MISS_WEIGHT if category == "neighborhood_miss" else DEFAULT_WEIGHT
            triples.append({
                "question_id": qid,
                "query": q["question"],
                "ticker": row["ticker"],
                "positive_chunk_id": gold_ids[0],
                "all_positive_chunk_ids": relevant,
                "hard_negative_chunk_id": neg_id,
                "category": category,
                "sample_weight": weight,
            })

    question_pattern_counts = Counter(r["pattern"] for r in per_question)
    all_neg_categories = Counter()
    for r in per_question:
        all_neg_categories.update(r["category_counts"])

    summary = {
        "n_questions": len(per_question),
        "question_level_pattern_counts": dict(question_pattern_counts),
        "hard_negative_level_category_counts": dict(all_neg_categories),
        "n_training_triples": len(triples),
        "n_excluded_false_negatives": sum(len(r["excluded_false_negatives"]) for r in per_question),
        "n_neighborhood_miss_triples": sum(1 for t in triples if t["category"] == "neighborhood_miss"),
        "n_genuine_retrieval_failure_questions_excluded": question_pattern_counts.get("genuine_retrieval_failure", 0),
    }

    return {
        "usage": ("DIAGNOSTIC ONLY. Every query is an eval question: do not train a model on these triples and "
                  "then evaluate it on eval_questions.json (train/test contamination). Phase C must mine from a "
                  "question set disjoint from the eval set (assert_disjoint_from_eval)."),
        "eval_question_ids": sorted({t["question_id"] for t in triples}),
        "category_definitions": CATEGORY_DEFINITIONS,
        "summary": summary,
        "per_question": per_question,
        "training_triples": triples,
    }


def main():
    diagnostic_path = REPO_ROOT / "data" / "eval_set" / "day5_ceiling_diagnostic.json"
    questions_path = REPO_ROOT / "data" / "eval_set" / "eval_questions.json"

    diagnostic_rows = json.loads(diagnostic_path.read_text())
    questions = json.loads(questions_path.read_text())
    questions_by_id = {q["id"]: q for q in questions}
    import sys
    sys.path.insert(0, str(REPO_ROOT))
    from src.eval.keyword_baseline import load_chunks
    chunks_by_id = {}
    for f in sorted((REPO_ROOT / "data" / "processed" / "chunks").glob("*.jsonl")):
        for c in load_chunks(str(f)):
            chunks_by_id[c["chunk_id"]] = c

    result = mine(diagnostic_rows, questions_by_id, chunks_by_id)

    print("=" * 78)
    print("DAY 6 HARD-NEGATIVE MINING")
    print("=" * 78)
    print(f"\n{'qid':7} {'tick':6} {'rank':>6} {'n_negs':>7}  category_counts  -> pattern")
    print("-" * 78)
    for r in result["per_question"]:
        rank = r["hybrid_rank"] if r["hybrid_rank"] is not None else ">200"
        print(f"{r['id']:7} {r['ticker']:6} {str(rank):>6} {r['n_hard_negatives']:>7}  "
              f"{r['category_counts']}  -> {r['pattern']}")

    s = result["summary"]
    print("\n" + "-" * 78)
    print(f"Question-level pattern counts:       {s['question_level_pattern_counts']}")
    print(f"Hard-negative-level category counts: {s['hard_negative_level_category_counts']}")
    print(f"Training triples mined:              {s['n_training_triples']}")
    print(f"  of which neighborhood_miss (oversampled {NEIGHBORHOOD_MISS_WEIGHT}x): "
          f"{s['n_neighborhood_miss_triples']}")
    print(f"Questions excluded (genuine_retrieval_failure, no valid hard negative): "
          f"{s['n_genuine_retrieval_failure_questions_excluded']}")

    print(f"Excluded as false negatives (answer-bearing): {s['n_excluded_false_negatives']}")
    out_path = REPO_ROOT / "data" / "eval_set" / "day6_hard_negatives_v2.json"
    out_path.write_text(json.dumps(result, indent=2))
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
