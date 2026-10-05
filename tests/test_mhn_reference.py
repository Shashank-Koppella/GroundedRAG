"""Tests for scripts/build_mhn_reference.py on a synthetic facts table."""
import csv
from pathlib import Path

import pytest

from scripts.build_mhn_reference import build_reference, growth
from src.agent.facts_db import FactsDB, build_db

HEADER = ["ticker", "metric", "tag_used", "fy", "fp", "form", "start", "end", "val"]


def fy(t, m, year, val):
    return [t, m, "Tag", str(year), "FY", "10-K", f"{year}-01-01", f"{year}-12-31", str(val)]


@pytest.fixture
def db(tmp_path):
    rows = {
        "AAPL": [fy("AAPL", "revenue", 2023, 1000), fy("AAPL", "revenue", 2024, 1100),
                 fy("AAPL", "net_income", 2023, 100), fy("AAPL", "net_income", 2024, 150)],
        # MSFT net income stops a year early (the AVGO pattern)
        "MSFT": [fy("MSFT", "revenue", 2023, 500), fy("MSFT", "revenue", 2024, 600), fy("MSFT", "revenue", 2025, 800),
                 fy("MSFT", "net_income", 2023, 50), fy("MSFT", "net_income", 2024, 60)],
        # a negative base
        "AMZN": [fy("AMZN", "net_income", 2023, -20), fy("AMZN", "net_income", 2024, 30),
                 fy("AMZN", "revenue", 2023, 900), fy("AMZN", "revenue", 2024, 950)],
        "NVDA": [fy("NVDA", "revenue", 2024, 10)],   # only one fiscal year
    }
    for t, r in rows.items():
        with open(Path(tmp_path) / f"{t}_facts.csv", "w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh); w.writerow(HEADER); w.writerows(r)
    build_db(str(tmp_path), str(tmp_path / "t.sqlite"))
    d = FactsDB(str(tmp_path / "t.sqlite"))
    yield d
    d.close()


def test_growth_is_computed_from_the_two_latest_fiscal_years_newest_first(db):
    g = growth(db, "AAPL", "revenue")
    assert g["status"] == "ok" and g["answer_percent"] == pytest.approx(10.0)
    assert [i["end_date"] for i in g["inputs"]] == ["2024-12-31", "2023-12-31"]
    assert g["expression"] == "pct_change(1000, 1100)"


def test_a_series_that_stops_early_is_flagged_not_reported_as_ok(db):
    g = growth(db, "MSFT", "net_income")
    assert g["status"] == "caveat" and g["answer_percent"] == pytest.approx(20.0)
    assert "stale" in g["caveats"][0] and "2025-12-31" in g["caveats"][0]
    assert growth(db, "MSFT", "revenue")["status"] == "ok"   # the longer series is fine


def test_negative_base_is_unanswerable_with_the_calculators_reason(db):
    g = growth(db, "AMZN", "net_income")
    assert g["status"] == "unanswerable" and "negative" in g["reason"]


def test_one_fiscal_year_is_unanswerable(db):
    assert growth(db, "NVDA", "revenue")["status"] == "unanswerable"


def test_comparison_inherits_the_worst_part_status_and_caveats(db):
    ref = build_reference(db)
    assert ref["mhn_001"]["status"] == "ok" and ref["mhn_001"]["answer_percent"] == pytest.approx(10.0)
    # mhn_017 compares AAPL vs MSFT revenue: both ok; MSFT 800/600 = 33.3% beats AAPL 10%
    assert ref["mhn_017"]["status"] == "ok" and ref["mhn_017"]["winner"] == "MSFT"
    # mhn_004 is MSFT net income -> caveat; unrelated tickers with no rows are unanswerable, never a crash
    assert ref["mhn_004"]["status"] == "caveat"
    assert ref["mhn_013"]["status"] == "unanswerable"


def test_margin_requires_both_metrics_in_the_same_fiscal_year(db):
    ref = build_reference(db)
    assert ref["mhn_008"]["kind"] == "margin"   # AMZN: NI 30 / revenue 950 for 2024
    assert ref["mhn_008"]["status"] == "ok" and ref["mhn_008"]["answer_percent"] == pytest.approx(30 / 950 * 100, abs=1e-3)

def test_compare_growth_tie_names_no_winner(tmp_path, monkeypatch):
    import scripts.build_mhn_reference as bmr
    fake = {"A": {"status": "ok", "caveats": [], "answer_percent": 10.0},
            "B": {"status": "ok", "caveats": [], "answer_percent": 10.0}}
    monkeypatch.setattr(bmr, "SPECS", {"x": ("compare_growth", ("A", "B"), "revenue", "higher")})
    monkeypatch.setattr(bmr, "growth", lambda db, t, m: fake[t])
    out = bmr.build_reference(db=None)["x"]
    assert out["winner"] is None and out["status"] == "caveat" and "tie" in out["caveats"][0]
