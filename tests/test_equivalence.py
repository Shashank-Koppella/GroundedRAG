"""Tests for src/eval/equivalence.py, the equivalence-aware scoring in src/eval/scoring.py, and the
Day 6 miner's false-negative exclusion and contamination guard (Oct 5 audit fixes)."""
import json
from pathlib import Path

import pytest

from scripts.mine_hard_negatives import assert_disjoint_from_eval, mine
from src.eval.equivalence import find_equivalents, jaccard, relevant_ids, token_set
from src.eval.scoring import score_retrieval

REPO = Path(__file__).resolve().parent.parent

BASE = "Apple depends on manufacturing partners located primarily in China and other Asian countries"
CHUNKS = {
    "AAPL_1_item_1a_0001": {"ticker": "AAPL", "text": BASE + "."},
    "AAPL_2_item_1a_0001": {"ticker": "AAPL", "text": BASE + "!"},                 # same tokens, other filing
    "AAPL_3_item_1a_0001": {"ticker": "AAPL", "text": BASE + " and India today."},  # near, below 0.9
    "AAPL_4_item_7_0003": {"ticker": "AAPL", "text": "Services revenue grew on advertising."},
    "MSFT_1_item_1a_0001": {"ticker": "MSFT", "text": BASE + "."},                 # same text, other company
}


def test_jaccard_and_token_set_basics():
    assert jaccard(token_set("A b, c."), token_set("c b a")) == 1.0
    assert jaccard(token_set("a b"), token_set("c d")) == 0.0


def test_equivalents_are_same_company_above_threshold_only():
    eq = find_equivalents(["AAPL_1_item_1a_0001"], CHUNKS, 0.9)
    assert list(eq) == ["AAPL_2_item_1a_0001"]                          # MSFT copy excluded, near copy below 0.9
    loose = find_equivalents(["AAPL_1_item_1a_0001"], CHUNKS, 0.8)
    assert set(loose) == {"AAPL_2_item_1a_0001", "AAPL_3_item_1a_0001"}


def test_a_stale_gold_id_is_an_error():
    with pytest.raises(KeyError):
        find_equivalents(["AAPL_9_item_1a_0001"], CHUNKS)


def test_relevant_ids_is_gold_plus_equivalents_deduplicated():
    q = {"gold_chunk_ids": ["a", "b"], "equivalent_chunk_ids": ["b", "c"]}
    assert relevant_ids(q) == ["a", "b", "c"]
    assert relevant_ids({"gold_chunk_ids": ["a"]}) == ["a"]


def test_scoring_reports_strict_and_equivalence_aware_side_by_side():
    qs = [{"id": "q1", "question": "x", "ticker": "AAPL", "gold_answer_type": "chunk",
           "gold_chunk_ids": ["g"], "equivalent_chunk_ids": ["e"]}]
    rep = score_retrieval(qs, lambda q, t, k: ["n1", "e", "g"])
    assert rep["recall_at_k"][1]["mean"] == 0 and rep["recall_at_k_with_equivalents"][1]["mean"] == 0
    assert rep["mrr"]["mean"] == pytest.approx(1 / 3) and rep["mrr_with_equivalents"]["mean"] == pytest.approx(1 / 2)
    assert rep["mrr_cutoff"] == 10 and rep["small_n"] is True


def test_miner_excludes_answer_bearing_negatives_and_lists_all_positives():
    q = {"id": "sh_x", "question": "q", "gold_chunk_ids": ["AAPL_1_item_1a_0001"],
         "equivalent_chunk_ids": ["AAPL_2_item_1a_0001"]}
    row = {"id": "sh_x", "ticker": "AAPL", "hybrid_rank": 5, "in_reranker_pool_top50": True,
           "hard_negatives": ["AAPL_2_item_1a_0001", "AAPL_3_item_1a_0001", "AAPL_4_item_7_0003"]}
    out = mine([row], {"sh_x": q}, CHUNKS)
    pq = out["per_question"][0]
    assert pq["excluded_false_negatives"] == ["AAPL_2_item_1a_0001", "AAPL_3_item_1a_0001"]   # equivalent + Jaccard>=0.8
    negs = [t["hard_negative_chunk_id"] for t in out["training_triples"]]
    assert negs == ["AAPL_4_item_7_0003"]
    assert out["training_triples"][0]["all_positive_chunk_ids"] == ["AAPL_1_item_1a_0001", "AAPL_2_item_1a_0001"]
    assert "DIAGNOSTIC ONLY" in out["usage"] and out["eval_question_ids"] == ["sh_x"]


def test_contamination_guard():
    assert_disjoint_from_eval(["t1", "t2"], ["sh_001"])
    with pytest.raises(ValueError, match="contamination"):
        assert_disjoint_from_eval(["t1", "sh_001"], ["sh_001", "sh_002"])


@pytest.mark.skipif(not (REPO / "data" / "eval_set" / "eval_questions.json").exists(), reason="no eval set")
def test_stored_equivalents_are_disjoint_from_gold_and_same_company():
    qs = json.loads((REPO / "data" / "eval_set" / "eval_questions.json").read_text(encoding="utf-8"))
    for q in qs:
        if q["type"] != "single_hop":
            assert not q.get("equivalent_chunk_ids")
            continue
        eq = q.get("equivalent_chunk_ids", [])
        assert not set(eq) & set(q["gold_chunk_ids"]), q["id"]
        assert all(e.split("_")[0] == q["ticker"] for e in eq), q["id"]
