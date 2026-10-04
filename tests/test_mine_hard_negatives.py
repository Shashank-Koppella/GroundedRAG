"""Regression tests for scripts/mine_hard_negatives.py (Phase A, Day 6)."""
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from scripts.mine_hard_negatives import (
    parse_chunk_id,
    categorize_negative,
    classify_question,
    mine,
    NEIGHBORHOOD_MISS_WEIGHT,
    DEFAULT_WEIGHT,
)


def test_parse_chunk_id_extracts_all_four_fields():
    assert parse_chunk_id("AAPL_000032019324000123_item_1a_0010") == (
        "AAPL", "000032019324000123", "item_1a", 10
    )


def test_parse_chunk_id_returns_none_for_malformed_id():
    assert parse_chunk_id("not_a_real_chunk_id") is None


def test_categorize_negative_flags_true_plus_one_adjacency_as_neighborhood_miss():
    gold_parsed = [("AMZN", "000101872426000004", "item_7", 26)]
    # one chunk index below gold, same filing+item
    assert categorize_negative("AMZN_000101872426000004_item_7_0025", gold_parsed) == "neighborhood_miss"
    # one chunk index above gold, same filing+item
    assert categorize_negative("AMZN_000101872426000004_item_7_0027", gold_parsed) == "neighborhood_miss"


def test_categorize_negative_same_doc_two_away_is_not_neighborhood_miss():
    gold_parsed = [("AMZN", "000101872426000004", "item_7", 26)]
    assert categorize_negative("AMZN_000101872426000004_item_7_0024", gold_parsed) == "same_doc_distant"


def test_categorize_negative_same_ticker_item_different_filing_is_cross_filing_confusion():
    gold_parsed = [("AMZN", "000101872426000004", "item_7", 26)]
    assert categorize_negative("AMZN_000101872425000004_item_7_0003", gold_parsed) == "cross_filing_confusion"


def test_categorize_negative_same_ticker_different_item_is_topic_drift():
    gold_parsed = [("AMZN", "000101872426000004", "item_7", 26)]
    assert categorize_negative("AMZN_000101872426000004_item_1a_0005", gold_parsed) == "topic_drift"


def test_categorize_negative_different_ticker_is_off_topic():
    gold_parsed = [("AMZN", "000101872426000004", "item_7", 26)]
    assert categorize_negative("MSFT_000101872423000004_item_7_0001", gold_parsed) == "off_topic"


def test_categorize_negative_checks_every_gold_id_not_just_the_first():
    # gold spans two filings (e.g. sh_003-style widened label); a negative adjacent
    # to the SECOND gold id must still be caught as neighborhood_miss.
    gold_parsed = [("AAPL", "000101872426000004", "item_1a", 42), ("AAPL", "000101872425000004", "item_1a", 42)]
    assert categorize_negative("AAPL_000101872425000004_item_1a_0041", gold_parsed) == "neighborhood_miss"


def test_classify_question_is_genuine_retrieval_failure_when_rank_is_none():
    assert classify_question(None, ["topic_drift", "topic_drift"]) == "genuine_retrieval_failure"


def test_classify_question_is_genuine_retrieval_failure_even_with_no_hard_negatives_listed():
    # rank=None takes priority over the empty-list case -- there's still no gold
    # to categorize negatives against, so this must not fall through to
    # "no_hard_negatives".
    assert classify_question(None, []) == "genuine_retrieval_failure"


def test_classify_question_is_neighborhood_miss_if_any_negative_is_that_category():
    # even a single neighborhood_miss among many other-category negatives should
    # surface the pattern -- this is the rare, high-value case Day 6 is checking for.
    cats = ["cross_filing_confusion"] * 9 + ["neighborhood_miss"]
    assert classify_question(18, cats) == "neighborhood_miss"


def test_classify_question_majority_cross_filing_confusion():
    cats = ["cross_filing_confusion"] * 6 + ["topic_drift"] * 4
    assert classify_question(5, cats) == "cross_filing_confusion"


def test_classify_question_majority_topic_drift():
    cats = ["topic_drift"] * 6 + ["cross_filing_confusion"] * 4
    assert classify_question(5, cats) == "topic_drift"


def test_classify_question_mixed_when_neither_category_reaches_half():
    cats = ["cross_filing_confusion"] * 4 + ["topic_drift"] * 4 + ["off_topic"] * 2
    assert classify_question(5, cats) == "mixed"


def test_mine_excludes_genuine_retrieval_failure_questions_from_training_triples():
    diagnostic_rows = [
        {"id": "q1", "ticker": "NVDA", "hybrid_rank": None, "in_reranker_pool_top50": False,
         "hard_negatives": ["NVDA_000101872426000004_item_7_0001", "NVDA_000101872426000004_item_1a_0002"]},
    ]
    questions_by_id = {"q1": {"id": "q1", "question": "test question",
                               "gold_chunk_ids": ["NVDA_000101872426000004_item_7_0009"]}}
    result = mine(diagnostic_rows, questions_by_id)
    assert result["summary"]["n_training_triples"] == 0
    assert result["per_question"][0]["pattern"] == "genuine_retrieval_failure"
    assert result["summary"]["n_genuine_retrieval_failure_questions_excluded"] == 1


def test_mine_produces_one_triple_per_hard_negative_for_a_found_question():
    diagnostic_rows = [
        {"id": "q2", "ticker": "AMZN", "hybrid_rank": 18, "in_reranker_pool_top50": True,
         "hard_negatives": ["AMZN_000101872426000004_item_7_0025", "AMZN_000101872425000004_item_7_0003"]},
    ]
    questions_by_id = {"q2": {"id": "q2", "question": "AWS driver question",
                               "gold_chunk_ids": ["AMZN_000101872426000004_item_7_0026"]}}
    result = mine(diagnostic_rows, questions_by_id)
    triples = result["training_triples"]
    assert len(triples) == 2
    assert {t["hard_negative_chunk_id"] for t in triples} == {
        "AMZN_000101872426000004_item_7_0025", "AMZN_000101872425000004_item_7_0003"
    }
    assert all(t["positive_chunk_id"] == "AMZN_000101872426000004_item_7_0026" for t in triples)


def test_mine_oversamples_neighborhood_miss_triples_by_configured_weight():
    diagnostic_rows = [
        {"id": "q3", "ticker": "AMZN", "hybrid_rank": 18, "in_reranker_pool_top50": True,
         "hard_negatives": ["AMZN_000101872426000004_item_7_0025", "AMZN_000101872423000004_item_1a_0001"]},
    ]
    questions_by_id = {"q3": {"id": "q3", "question": "test",
                               "gold_chunk_ids": ["AMZN_000101872426000004_item_7_0026"]}}
    result = mine(diagnostic_rows, questions_by_id)
    triples = {t["hard_negative_chunk_id"]: t for t in result["training_triples"]}
    assert triples["AMZN_000101872426000004_item_7_0025"]["category"] == "neighborhood_miss"
    assert triples["AMZN_000101872426000004_item_7_0025"]["sample_weight"] == NEIGHBORHOOD_MISS_WEIGHT
    assert triples["AMZN_000101872423000004_item_1a_0001"]["sample_weight"] == DEFAULT_WEIGHT
