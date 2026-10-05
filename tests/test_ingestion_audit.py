"""Regression tests for the Oct 5 ingestion audit fixes: deterministic tag preference, the net-income
fallback tag, a fixed window start, paginated filing lists, and the 10-Q Item 1 TOC fragment."""
import json
from datetime import date
from pathlib import Path

import pytest

from src.ingestion.config import NET_INCOME_TAG_CANDIDATES, WINDOW_START
from src.ingestion.edgar_client import get_company_filings
from src.ingestion.section_splitter import split_into_items
from src.ingestion.xbrl_extractor import extract_key_metrics


def _fact(val, end, start=None, form="10-K", fp="FY", fy=2025, filed="2025-11-01"):
    return {"val": val, "end": end, "start": start, "form": form, "fp": fp, "fy": fy, "filed": filed}


def _facts(**tags):
    return {"facts": {"us-gaap": {t: {"units": {"USD": f}} for t, f in tags.items()}}}


# ---------------------------------------------------------------- XBRL

def test_same_filing_tie_always_goes_to_the_earlier_listed_tag():
    periods = [(f"2025-{m:02d}-01", f"2025-{m:02d}-28") for m in range(1, 13)] * 1
    pref = [_fact(100 + i, e, s, form="10-Q", fp="Q1") for i, (s, e) in enumerate(periods)]
    other = [_fact(100 + i, e, s, form="10-Q", fp="Q1") for i, (s, e) in enumerate(periods)]
    df = extract_key_metrics(_facts(RevenueFromContractWithCustomerExcludingAssessedTax=pref, Revenues=other),
                             "X", window_start=date(2020, 1, 1))
    rev = df[df.metric == "revenue"]
    assert len(rev) == 12 and set(rev.tag_used) == {"RevenueFromContractWithCustomerExcludingAssessedTax"}


def test_earliest_filing_still_wins_over_tag_preference():
    early = [_fact(5, "2023-12-31", "2023-01-01", filed="2024-02-01")]
    late = [_fact(5, "2023-12-31", "2023-01-01", filed="2025-02-01")]
    df = extract_key_metrics(_facts(RevenueFromContractWithCustomerExcludingAssessedTax=late, Revenues=early),
                             "X", window_start=date(2020, 1, 1))
    assert list(df.tag_used) == ["Revenues"]


def test_profitloss_fills_net_income_periods_netincomeloss_lacks():
    assert NET_INCOME_TAG_CANDIDATES[0] == "NetIncomeLoss" and "ProfitLoss" in NET_INCOME_TAG_CANDIDATES
    assert not any("AvailableToCommon" in t for t in NET_INCOME_TAG_CANDIDATES)   # a different number
    nil = [_fact(10, "2024-11-03", "2023-10-30", filed="2024-12-20")]                 # only one filing uses it
    pl = [_fact(10, "2024-11-03", "2023-10-30", filed="2024-12-20"),
          _fact(12, "2025-11-02", "2024-11-04", filed="2025-12-18"),
          _fact(3, "2025-08-03", "2025-05-05", form="10-Q", fp="Q3", filed="2025-09-10")]
    df = extract_key_metrics(_facts(NetIncomeLoss=nil, ProfitLoss=pl), "AVGO", window_start=date(2020, 1, 1))
    ni = df[df.metric == "net_income"].sort_values("end")
    assert list(ni.end) == ["2024-11-03", "2025-08-03", "2025-11-02"]
    assert list(ni.tag_used) == ["NetIncomeLoss", "ProfitLoss", "ProfitLoss"]   # same filing: preferred tag


def test_fixed_window_start_is_used_instead_of_today():
    f = [_fact(1, "2022-09-24", "2021-09-26"), _fact(2, "2022-06-30", "2021-07-01")]
    df = extract_key_metrics(_facts(Revenues=f), "X", window_start=WINDOW_START)
    assert list(df.end) == ["2022-09-24"]            # 2022-06-30 is before 2022-09-17


# ---------------------------------------------------------------- filing list pagination

def _block(rows):
    return {"form": [r[0] for r in rows], "accessionNumber": [r[1] for r in rows],
            "filingDate": [r[2] for r in rows], "primaryDocument": [r[3] for r in rows]}


def test_filings_are_read_from_paged_history_not_only_recent():
    pages = {
        "https://data.sec.gov/submissions/CIK0000000001.json": {"filings": {
            "recent": _block([("10-Q", "A3", "2024-08-01", "q.htm"), ("4", "F1", "2024-07-01", "f.xml")]),
            "files": [{"name": "CIK0000000001-submissions-001.json", "filingFrom": "2022-01-01", "filingTo": "2024-06-30"},
                      {"name": "CIK0000000001-submissions-002.json", "filingFrom": "2015-01-01", "filingTo": "2021-12-31"}]}},
        "https://data.sec.gov/submissions/CIK0000000001-submissions-001.json":
            _block([("10-K", "A2", "2024-02-01", "k.htm"), ("10-Q", "A1", "2022-10-25", "q1.htm"),
                    ("10-Q", "A0", "2022-08-01", "old.htm"), ("10-Q", "A3", "2024-08-01", "q.htm")]),
    }
    fetched = []

    def fetch(url):
        fetched.append(url)
        return pages[url]

    out = get_company_filings("0000000001", ["10-K", "10-Q"], window_start=WINDOW_START, fetch=fetch)
    assert [r["accession_number"] for r in out] == ["A3", "A2", "A1"]     # newest first, deduped, in window
    assert not any("002" in u for u in fetched)                            # a page wholly before the window is skipped


# ---------------------------------------------------------------- 10-Q Item 1 TOC fragment

TOC_SHAPE = "\n".join([
    "PART I. FINANCIAL INFORMATION",
    "Item 1. Financial Statements",
    "a) Income Statements for the Three and Nine Months Ended March 31, 2026 and 2025 3",
    "b) Comprehensive Income Statements for the Three and Nine Months Ended March 31, 2026 and 2025 4",
    "c) Balance Sheets as of March 31, 2026 and June 30, 2025 5",
    "d) Cash Flows Statements for the Three and Nine Months Ended March 31, 2026 and 2025 6",
    "e) Stockholders' Equity Statements for the Three and Nine Months Ended March 31, 2026 and 2025 7",
    "f) Notes to Financial Statements 8",
    "Item 2. Management's Discussion and Analysis of Financial Condition and Results of Operations 31",
    "Item 3. Quantitative and Qualitative Disclosures about Market Risk 45",
    "Item 4. Controls and Procedures 46",
    "Item 1. Financial Statements",
    " ".join(["statements"] * 600),
    "Item 2. Management's Discussion and Analysis of Financial Condition and Results of Operations",
    " ".join(["mda"] * 500),
    "Item 3. Quantitative and Qualitative Disclosures about Market Risk",
    " ".join(["marketrisk"] * 100),
    "PART II. OTHER INFORMATION",
    "Item 1. Legal Proceedings",
    " ".join(["legal"] * 80),
    "Item 1A. Risk Factors",
    " ".join(["risk"] * 300),
    "Item 2. Unregistered Sales of Equity Securities and Use of Proceeds",
    " ".join(["equity"] * 60),
])
ITEMS = {"item_1": "Financial Statements", "item_1a": "Risk Factors", "item_2": "MD&A"}


def test_without_the_fix_item_1_is_the_toc_fragment():
    s = split_into_items(TOC_SHAPE, ITEMS)
    assert "statements statements" not in s["item_1"] and len(s["item_1"].split()) < 100


def test_with_min_words_item_1_is_the_real_financial_statements():
    s = split_into_items(TOC_SHAPE, ITEMS, {"item_1": 300})
    assert s["item_1"].count("statements") >= 600
    assert "legal" not in s["item_1"] and "mda" not in s["item_1"]


def test_min_words_does_not_change_other_items():
    a, b = split_into_items(TOC_SHAPE, ITEMS), split_into_items(TOC_SHAPE, ITEMS, {"item_1": 300})
    assert a["item_1a"] == b["item_1a"] and a["item_2"] == b["item_2"]


def test_min_words_keeps_the_original_choice_when_no_candidate_is_long_enough():
    s = split_into_items(TOC_SHAPE, ITEMS, {"item_1": 10_000})
    assert s["item_1"] == split_into_items(TOC_SHAPE, ITEMS)["item_1"]


# ---------------------------------------------------------------- corpus comparison guard

def test_compare_corpus_fails_when_a_gold_chunk_changes(tmp_path, monkeypatch):
    import scripts.compare_corpus as cc
    q = [{"id": "sh_1", "gold_chunk_ids": ["A_1_item_1a_0000"], "equivalent_chunk_ids": []}]
    ev = tmp_path / "eval.json"; ev.write_text(json.dumps(q))
    monkeypatch.setattr(cc, "EVAL", ev)
    rec = lambda text: {"chunk_id": "A_1_item_1a_0000", "text": text, "ticker": "A", "form": "10-K",
                        "item": "item_1a", "accession_number": "1", "filing_date": "2025-01-01", "chunk_index": 0}
    assert cc.compare_text({"A_1_item_1a_0000": rec("x")}, {"A_1_item_1a_0000": rec("x")}) == 0
    assert cc.compare_text({"A_1_item_1a_0000": rec("x")}, {"A_1_item_1a_0000": rec("y")}) == 1
    assert cc.compare_text({"A_1_item_1a_0000": rec("x")}, {}) == 1


# ---------------------------------------------------------------- verifier follow-ups (Oct 5)

def test_fallback_tag_never_beats_a_primary_tag_even_when_filed_earlier():
    nil = [_fact(10, "2024-11-03", "2023-10-30", filed="2025-12-18")]     # primary, filed later
    pl = [_fact(11, "2024-11-03", "2023-10-30", filed="2024-12-20")]      # fallback, filed earlier
    df = extract_key_metrics(_facts(NetIncomeLoss=nil, ProfitLoss=pl), "X", window_start=date(2020, 1, 1))
    assert list(df.tag_used) == ["NetIncomeLoss"] and list(df.val) == [10]


def test_facts_filed_after_the_window_end_are_excluded():
    f = [_fact(1, "2026-06-30", "2025-07-01", filed="2026-07-29"), _fact(2, "2026-09-30", "2026-07-01",
         form="10-Q", fp="Q1", filed="2026-10-28")]
    df = extract_key_metrics(_facts(Revenues=f), "X", window_start=WINDOW_START, window_end=date(2026, 9, 17))
    assert list(df.end) == ["2026-06-30"]


def test_filings_after_the_window_end_are_excluded():
    page = {"filings": {"recent": _block([("10-Q", "B2", "2026-10-28", "q.htm"), ("10-K", "B1", "2026-07-29", "k.htm")])}}
    out = get_company_filings("1", ["10-K", "10-Q"], window_start=WINDOW_START, window_end=date(2026, 9, 17),
                              fetch=lambda u: page)
    assert [r["accession_number"] for r in out] == ["B1"]


def test_window_end_reproduces_the_current_corpus_dates():
    from src.ingestion.config import WINDOW_END
    root = Path(__file__).resolve().parent.parent / "data" / "processed" / "chunks"
    dates = [json.loads(l)["filing_date"] for f in root.glob("*.jsonl") for l in open(f, encoding="utf-8")] if root.exists() else []
    if not dates:
        pytest.skip("no corpus")
    assert max(dates) <= WINDOW_END.isoformat() and min(dates) >= WINDOW_START.isoformat()


# ---------------------------------------------------------------- after the corpus rebuild (Oct 5)

def test_conflict_warning_ignores_differences_between_tags_but_flags_same_tag(caplog):
    # ORCL printed 4 'DIFFERENT reported values' warnings that were really NetIncomeLoss vs ProfitLoss
    # (noncontrolling interests). Different tags must stay quiet; a same-tag difference must still warn.
    nil = [_fact(10, "2024-05-31", "2023-06-01", filed="2024-06-20")]
    pl = [_fact(11, "2024-05-31", "2023-06-01", filed="2024-06-20")]
    with caplog.at_level("WARNING"):
        extract_key_metrics(_facts(NetIncomeLoss=nil, ProfitLoss=pl), "ORCL", window_start=date(2020, 1, 1))
    assert not [r for r in caplog.records if "DIFFERENT" in r.getMessage()]

    caplog.clear()
    restated = [_fact(10, "2024-05-31", "2023-06-01", filed="2024-06-20"),
                _fact(12, "2024-05-31", "2023-06-01", filed="2025-06-20")]
    with caplog.at_level("WARNING"):
        df = extract_key_metrics(_facts(NetIncomeLoss=restated), "ORCL", window_start=date(2020, 1, 1))
    msgs = [r.getMessage() for r in caplog.records if "DIFFERENT" in r.getMessage()]
    assert len(msgs) == 1 and "NetIncomeLoss" in msgs[0]
    assert list(df.val) == [10]          # the earliest filing still wins
