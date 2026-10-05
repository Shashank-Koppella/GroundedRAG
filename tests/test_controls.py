"""Tests for src/eval/controls.py (faithfulness validity controls)."""
import pytest

from src.eval.controls import HEADER_ONLY_FILLER, control_suffix, header_only, pair_contexts

CHUNKS = {f"{t}_{i}": {"chunk_id": f"{t}_{i}", "ticker": t, "text": f"{t} text {i}", "form": "10-K",
                       "filing_date": "2025-01-01", "item": "item_1a"}
          for t in ("AAPL", "MSFT", "NVDA") for i in range(6)}


def ans(qid, ticker, ctx, typ="single_hop", refused=False, gold=()):
    return {"id": qid, "ticker": ticker, "type": typ, "is_refusal": refused,
            "context_chunk_ids": list(ctx), "gold_chunk_ids": list(gold)}


RESULTS = [
    ans("a1", "AAPL", ["AAPL_0", "AAPL_1"], gold=["AAPL_0"]),
    ans("a2", "AAPL", ["AAPL_2", "AAPL_3"], gold=["AAPL_2"]),
    ans("a3", "AAPL", ["AAPL_0", "AAPL_4"], gold=["AAPL_4"]),        # shares AAPL_0 with a1
    ans("m1", "MSFT", ["MSFT_0", "MSFT_1"]),
    ans("o1", None, ["AAPL_5"], typ="out_of_scope"),                  # None ticker, AAPL context
    ans("r1", "NVDA", ["NVDA_0"], refused=True),
]


def test_other_company_never_pairs_onto_the_same_companys_chunks():
    pairs = pair_contexts(RESULTS, CHUNKS, "other_company")
    for qid, ctx in pairs.items():
        own = next(r for r in RESULTS if r["id"] == qid)["ticker"]
        assert all(CHUNKS[c]["ticker"] != own for c in ctx)
    assert "o1" not in pairs and "r1" not in pairs         # out-of-scope and refusals are not used
    assert pairs["a1"] == ["MSFT_0", "MSFT_1"]


def test_same_company_requires_same_ticker_no_shared_chunks_and_no_gold():
    pairs = pair_contexts(RESULTS, CHUNKS, "same_company")
    assert pairs["a1"] == ["AAPL_2", "AAPL_3"]             # a3 shares AAPL_0, so it is skipped
    assert pairs["a2"] == ["AAPL_0", "AAPL_4"]
    assert pairs["a3"] == ["AAPL_2", "AAPL_3"]             # a1 shares AAPL_0
    assert "m1" not in pairs                               # no other MSFT answer: left out, not mis-paired


def test_unknown_mode_raises():
    with pytest.raises(ValueError):
        pair_contexts(RESULTS, CHUNKS, "random")


def test_header_only_keeps_provenance_and_drops_content():
    c = header_only(CHUNKS["AAPL_0"])
    assert c["text"] == HEADER_ONLY_FILLER and c["ticker"] == "AAPL" and c["filing_date"] == "2025-01-01"
    assert CHUNKS["AAPL_0"]["text"] == "AAPL text 0"       # the original is not mutated


def test_output_suffixes_keep_the_day7_name_for_the_original_control():
    assert control_suffix("other_company", False) == "_negctl"
    assert control_suffix("same_company", False) == "_negctl_same"
    assert control_suffix(None, True) == "_hdronly"
    assert control_suffix(None, False) == ""


def test_same_company_partner_may_not_contain_an_equivalent_of_the_answer():
    # a2's context holds AAPL_2; if AAPL_2 is an equivalent of a1's answer, a1 must not be paired with it
    pairs = pair_contexts(RESULTS, CHUNKS, "same_company", {"a1": ["AAPL_0", "AAPL_2"]})
    assert pairs.get("a1") != ["AAPL_2", "AAPL_3"]
