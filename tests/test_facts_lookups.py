"""Tests for the status-bearing lookups on FactsDB (Day 9 step 1): latest_fiscal_year, fiscal_year_pair,
same_year_pair, quarter_window. The contract under test is that every lookup says ok / caveat /
unanswerable and never hands the agent a number whose limits it cannot see."""
import csv
from pathlib import Path

import pytest

from src.agent.facts_db import FactsDB, Lookup, build_db

HEADER = ["ticker", "metric", "tag_used", "fy", "fp", "form", "start", "end", "val"]


def fy(t, m, year, val, end=None):
    return [t, m, "Tag", str(year), "FY", "10-K", f"{year}-01-01", end or f"{year}-12-31", str(val)]


def q(t, m, start, end, val, form="10-Q", fp="Q"):
    return [t, m, "Tag", "2024", fp, form, start, end, str(val)]


@pytest.fixture
def db(tmp_path):
    rows = {
        "AAPL": [fy("AAPL", "revenue", 2023, 1000), fy("AAPL", "revenue", 2024, 1100),
                 fy("AAPL", "net_income", 2023, 100), fy("AAPL", "net_income", 2024, 150)],
        # net income stops a year early (the AVGO pattern)
        "MSFT": [fy("MSFT", "revenue", 2023, 500), fy("MSFT", "revenue", 2024, 600), fy("MSFT", "revenue", 2025, 800),
                 fy("MSFT", "net_income", 2023, 50), fy("MSFT", "net_income", 2024, 60)],
        # a missing year in the middle: 2022 and 2024 only
        "AMZN": [fy("AMZN", "revenue", 2022, 10), fy("AMZN", "revenue", 2024, 30)],
        "NVDA": [fy("NVDA", "revenue", 2024, 10)],
        # fiscal years that end on different dates for the two metrics
        "ORCL": [fy("ORCL", "revenue", 2024, 70), fy("ORCL", "net_income", 2024, 7, end="2024-11-30")],
        # four complete quarters needing one derived Q4, plus a 9-month YTD and the annual figure
        "GOOGL": [
            q("GOOGL", "revenue", "2024-01-01", "2024-03-31", 10), q("GOOGL", "revenue", "2024-04-01", "2024-06-30", 11),
            q("GOOGL", "revenue", "2024-07-01", "2024-09-30", 12), q("GOOGL", "revenue", "2025-01-01", "2025-03-31", 14),
            q("GOOGL", "revenue", "2024-01-01", "2024-09-30", 33, fp="Q3"),
            fy("GOOGL", "revenue", 2024, 46),
        ],
        # a real hole: Q1 and Q3 only
        "META": [q("META", "revenue", "2024-01-01", "2024-03-31", 5), q("META", "revenue", "2024-07-01", "2024-09-30", 6)],
        # four reported quarters, no derived Q4 needed
        "AVGO": [q("AVGO", "revenue", "2024-01-01", "2024-03-31", 1), q("AVGO", "revenue", "2024-04-01", "2024-06-30", 2),
                 q("AVGO", "revenue", "2024-07-01", "2024-09-30", 3), q("AVGO", "revenue", "2024-10-01", "2024-12-31", 4)],
    }
    for t, r in rows.items():
        with open(Path(tmp_path) / f"{t}_facts.csv", "w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh); w.writerow(HEADER); w.writerows(r)
    build_db(str(tmp_path), str(tmp_path / "t.sqlite"))
    d = FactsDB(str(tmp_path / "t.sqlite"))
    yield d
    d.close()


# ---------------------------------------------------------------- latest_fiscal_year

def test_latest_fiscal_year_ok(db):
    lk = db.latest_fiscal_year("AAPL", "revenue")
    assert lk.status == "ok" and [f.value for f in lk.facts] == [1100] and lk.caveats == ()


def test_latest_fiscal_year_missing_is_unanswerable_with_a_reason(db):
    lk = db.latest_fiscal_year("META", "revenue")
    assert lk.status == "unanswerable" and lk.facts == () and "META" in lk.reason


def test_latest_fiscal_year_of_a_stalled_series_is_a_caveat_naming_both_dates(db):
    lk = db.latest_fiscal_year("MSFT", "net_income")
    assert lk.status == "caveat" and lk.facts[0].end_date == "2024-12-31"
    assert "2024-12-31" in lk.caveats[0] and "2025-12-31" in lk.caveats[0] and "latest fiscal year" in lk.caveats[0]
    assert db.latest_fiscal_year("MSFT", "revenue").status == "ok"   # the longer series is fine


# ---------------------------------------------------------------- fiscal_year_pair

def test_pair_is_newest_first_and_ok(db):
    lk = db.fiscal_year_pair("AAPL", "revenue")
    assert lk.status == "ok" and [f.end_date for f in lk.facts] == ["2024-12-31", "2023-12-31"]


def test_pair_with_one_year_is_unanswerable_but_shows_what_was_found(db):
    lk = db.fiscal_year_pair("NVDA", "revenue")
    assert lk.status == "unanswerable" and len(lk.facts) == 1 and "only 1" in lk.reason


def test_pair_that_skips_a_year_is_a_caveat_not_a_silent_growth_rate(db):
    lk = db.fiscal_year_pair("AMZN", "revenue")
    assert lk.status == "caveat" and "not consecutive" in lk.caveats[0] and "2.00 years" in lk.caveats[0]


def test_pair_on_a_stalled_series_carries_the_staleness_caveat(db):
    lk = db.fiscal_year_pair("MSFT", "net_income")
    assert lk.status == "caveat" and len(lk.caveats) == 1
    assert "stale" in lk.caveats[0] and "year-over-year change" in lk.caveats[0]


# ---------------------------------------------------------------- same_year_pair

def test_same_year_pair_ok_returns_first_then_second_metric(db):
    lk = db.same_year_pair("AAPL", "net_income", "revenue")
    assert lk.status == "ok" and [f.metric for f in lk.facts] == ["net_income", "revenue"]


def test_same_year_pair_refuses_to_mix_fiscal_years(db):
    lk = db.same_year_pair("ORCL", "net_income", "revenue")
    assert lk.status == "unanswerable" and "2024-11-30" in lk.reason and "2024-12-31" in lk.reason
    assert len(lk.facts) == 2     # what was found is still shown


def test_same_year_pair_with_a_missing_metric_names_it(db):
    lk = db.same_year_pair("AMZN", "net_income", "revenue")
    assert lk.status == "unanswerable" and "net_income" in lk.reason


# ---------------------------------------------------------------- quarter_window

def test_quarter_window_with_a_derived_q4_is_a_caveat_and_flags_it(db):
    lk = db.quarter_window("GOOGL", "revenue", 4)
    assert lk.status == "caveat" and [f.value for f in lk.facts] == [14, 13, 12, 11]   # Q4 = 46 - 33
    assert [f.derived for f in lk.facts] == [False, True, False, False]
    assert "1 of 4 quarters are derived Q4" in lk.caveats[0]


def test_quarter_window_of_reported_quarters_is_ok(db):
    lk = db.quarter_window("AVGO", "revenue", 4)
    assert lk.status == "ok" and [f.value for f in lk.facts] == [4, 3, 2, 1]


def test_quarter_window_over_a_hole_is_unanswerable_not_a_shorter_average(db):
    lk = db.quarter_window("META", "revenue", 4)
    assert lk.status == "unanswerable" and "only 1 of 4" in lk.reason


# ---------------------------------------------------------------- contract shape

def test_lookup_as_dict_shape_and_status_vocabulary(db):
    d = db.fiscal_year_pair("MSFT", "net_income").as_dict()
    assert set(d) == {"status", "facts", "caveats", "reason"} and d["status"] == "caveat"
    assert d["facts"][0]["citation"].startswith("XBRL MSFT net income")
    for lk in (db.latest_fiscal_year("AAPL", "revenue"), db.fiscal_year_pair("NVDA", "revenue"),
               db.quarter_window("GOOGL", "revenue", 4), db.same_year_pair("ORCL", "net_income", "revenue")):
        assert isinstance(lk, Lookup) and lk.status in {"ok", "caveat", "unanswerable"}
        assert (lk.status == "unanswerable") == bool(lk.reason)     # a reason exists exactly when it refuses


def test_lookups_validate_their_inputs(db):
    with pytest.raises(ValueError):
        db.fiscal_year_pair("AAPL", "ebitda")
    with pytest.raises(ValueError):
        db.latest_fiscal_year("AA;PL", "revenue")

# ---------------------------------------------------------------- audit fixes (Oct 5)

@pytest.mark.parametrize("n", [1, 2, 3, 4])
def test_recent_quarters_returns_exactly_n(db, n):
    r = db.recent_quarters("GOOGL", "revenue", n)
    assert len(r["facts"]) == n and r["complete"]


def test_quarter_window_of_one_is_the_latest_quarter(db):
    lk = db.quarter_window("GOOGL", "revenue", 1)
    assert lk.status == "ok" and [f.value for f in lk.facts] == [14]


@pytest.mark.parametrize("bad", [0, -1, 1.5, True, "4"])
def test_recent_quarters_rejects_a_non_positive_or_non_integer_n(db, bad):
    with pytest.raises(ValueError):
        db.recent_quarters("GOOGL", "revenue", bad)


@pytest.mark.parametrize("sql", ["select randomblob(100000000)", "select zeroblob(2000000000)",
                                 "select printf('%.*c', 300000000, 'x')",
                                 "select replace(replace('x','x','xxxxxxxxxx'),'x','xxxxxxxxxx')",
                                 "select hex(1)"])
def test_run_select_denies_memory_unbounded_functions(db, sql):
    with pytest.raises(ValueError, match="not authorized"):
        db.run_select(sql)


def test_run_select_caps_query_length_and_denies_recursive_ctes(db):
    with pytest.raises(ValueError, match="too long"):
        db.run_select("select 1 " + "+ 1 " * 600)
    with pytest.raises(ValueError):
        db.run_select("with recursive c(x) as (select 1 union all select x+1 from c) select count(*) from c")


def test_run_select_still_allows_ordinary_functions(db):
    r = db.run_select("select upper(ticker), count(*), max(val) from facts group by ticker order by 1")
    assert r["rows"][0][0] == "AAPL"
