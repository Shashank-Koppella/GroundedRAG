import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src" / "eval"))

from scoring import recall_at_k, reciprocal_rank, bootstrap_ci, score_retrieval, score_sql_routing_sanity


def test_recall_at_k_hit():
    assert recall_at_k(["a"], ["x", "a", "y"], k=5) == 1


def test_recall_at_k_miss_outside_k():
    assert recall_at_k(["a"], ["x", "y", "a"], k=2) == 0


def test_recall_at_k_no_gold():
    assert recall_at_k([], ["x", "y"], k=5) == 0


def test_reciprocal_rank_first_position():
    assert reciprocal_rank(["a"], ["a", "b", "c"]) == 1.0


def test_reciprocal_rank_third_position():
    assert abs(reciprocal_rank(["a"], ["x", "y", "a"]) - (1 / 3)) < 1e-9


def test_reciprocal_rank_not_found():
    assert reciprocal_rank(["a"], ["x", "y", "z"]) == 0.0


def test_bootstrap_ci_bounds_contain_mean():
    values = [1, 1, 0, 1, 0, 1, 1, 0, 1, 1]
    lo, hi = bootstrap_ci(values, n_resamples=500)
    mean = sum(values) / len(values)
    assert lo <= mean <= hi


def test_bootstrap_ci_empty_list():
    lo, hi = bootstrap_ci([])
    assert lo != lo and hi != hi  # NaN != NaN


def test_score_retrieval_skips_unlabeled_questions():
    questions = [
        {"id": "q1", "type": "single_hop", "ticker": "AAPL", "question": "q",
         "gold_answer_type": "chunk", "gold_chunk_ids": []},
    ]
    def rank_fn(query, ticker, k):
        return ["c1", "c2"]
    report = score_retrieval(questions, rank_fn)
    assert report["n_scored"] == 0
    assert report["n_skipped_unlabeled"] == 1


def test_score_retrieval_perfect_match():
    questions = [
        {"id": "q1", "type": "single_hop", "ticker": "AAPL", "question": "q",
         "gold_answer_type": "chunk", "gold_chunk_ids": ["c1"]},
    ]
    def rank_fn(query, ticker, k):
        return ["c1", "c2", "c3"]
    report = score_retrieval(questions, rank_fn)
    assert report["n_scored"] == 1
    assert report["recall_at_k"][1]["mean"] == 1.0
    assert report["mrr"]["mean"] == 1.0


def test_score_sql_routing_sanity_flags_returned_chunks():
    questions = [
        {"id": "m1", "type": "multi_hop_numeric", "ticker": "AAPL",
         "gold_answer_type": "sql", "question": "revenue growth"},
    ]
    def rank_fn(query, ticker, k):
        return ["c1"]  # retriever wrongly returns something for a numeric question
    report = score_sql_routing_sanity(questions, rank_fn)
    assert report["n_false_confidence_flags"] == 1


def test_score_sql_routing_sanity_handles_multi_company():
    questions = [
        {"id": "m1", "type": "multi_hop_numeric", "ticker": "AAPL,MSFT",
         "gold_answer_type": "sql", "question": "compare growth"},
    ]
    def rank_fn(query, ticker, k):
        return []
    report = score_sql_routing_sanity(questions, rank_fn)
    assert report["n_numeric_questions_checked"] == 2  # split across both companies
    assert report["n_false_confidence_flags"] == 0
