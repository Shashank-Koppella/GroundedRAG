"""Tests for src/agent/facts_db.py. Synthetic CSVs reproduce each real-data trap found by profiling;
the final integration test checks the tool against an independent plain-Python read of the real CSVs
and is skipped when data/processed/xbrl is not present."""
import csv
import glob
import sqlite3
from pathlib import Path

import pytest

from src.agent.facts_db import FactsDB, build_db, classify_basis

HEADER = ["ticker", "metric", "tag_used", "fy", "fp", "form", "start", "end", "val"]


def write_csv(dirpath, ticker, rows):
    p = Path(dirpath) / f"{ticker}_facts.csv"
    with open(p, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(HEADER)
        for r in rows:
            w.writerow(r)


def row(t, m, fy, fp, form, start, end, val):
    return [t, m, "SomeTag", fy, fp, form, start, end, val]


@pytest.fixture
def db(tmp_path):
    # AAA: clean series. FY ends Dec 31. Fiscal years 2023, 2024; quarters through Q3 2024 + FY2024.
    aaa = [
        row("AAA", "revenue", "2023", "FY", "10-K", "2023-01-01", "2023-12-31", 1000),
        row("AAA", "revenue", "2024", "FY", "10-K", "2024-01-01", "2024-12-31", 1200),
        row("AAA", "revenue", "2023", "Q1", "10-Q", "2023-01-01", "2023-03-31", 220),
        row("AAA", "revenue", "2023", "Q2", "10-Q", "2023-04-01", "2023-06-30", 240),
        row("AAA", "revenue", "2023", "Q2", "10-Q", "2023-01-01", "2023-06-30", 460),   # six-month YTD
        row("AAA", "revenue", "2023", "Q3", "10-Q", "2023-07-01", "2023-09-30", 260),
        row("AAA", "revenue", "2023", "Q3", "10-Q", "2023-01-01", "2023-09-30", 720),   # nine-month YTD
        row("AAA", "revenue", "2024", "Q1", "10-Q", "2024-01-01", "2024-03-31", 270),
        row("AAA", "revenue", "2024", "Q2", "10-Q", "2024-04-01", "2024-06-30", 290),
        row("AAA", "revenue", "2024", "Q3", "10-Q", "2024-07-01", "2024-09-30", 310),
        row("AAA", "revenue", "2024", "Q3", "10-Q", "2024-01-01", "2024-09-30", 870),
    ]
    # BBB: AVGO-style raw-label mess: three annual rows all labelled fy=2024; plus a TTM row from a 10-Q.
    bbb = [
        row("BBB", "net_income", "2024", "FY", "10-K", "2021-11-01", "2022-10-30", 11),
        row("BBB", "net_income", "2024", "FY", "10-K", "2022-10-31", "2023-10-29", 14),
        row("BBB", "net_income", "2024", "FY", "10-K", "2023-10-30", "2024-11-03", 6),
        row("BBB", "net_income", "2025", "Q1", "10-Q", "2024-02-01", "2025-01-31", 999),   # ~365d from a 10-Q = TTM
        row("BBB", "revenue", "2024", "FY", "10-K", "2023-10-30", "2024-11-03", -5),
    ]
    write_csv(tmp_path, "AAA", aaa)
    write_csv(tmp_path, "BBB", bbb)
    path = tmp_path / "facts.sqlite"
    build_db(str(tmp_path), str(path))
    d = FactsDB(str(path))
    yield d
    d.close()


# ---------------------------------------------------------------- classification

@pytest.mark.parametrize("start,end,form,basis", [
    ("2024-01-01", "2024-03-31", "10-Q", "quarter"),
    ("2023-12-31", "2024-03-30", "10-Q", "quarter"),       # 13 weeks
    ("2024-01-01", "2024-06-30", "10-Q", "ytd"),
    ("2024-01-01", "2024-09-30", "10-Q", "ytd"),
    ("2023-01-01", "2023-12-31", "10-K", "fiscal_year"),
    ("2023-10-30", "2024-11-03", "10-K", "fiscal_year"),   # 53-week year
    ("2024-02-01", "2025-01-31", "10-Q", "ttm"),
])
def test_basis_comes_from_period_length_and_form(start, end, form, basis):
    assert classify_basis(start, end, form)[1] == basis


# ---------------------------------------------------------------- fiscal years

def test_fiscal_years_are_identified_by_end_date_not_by_the_raw_fy_label(db):
    fys = db.fiscal_years("BBB", "net_income", n=5)
    assert [f.end_date for f in fys] == ["2024-11-03", "2023-10-29", "2022-10-30"]  # newest first
    assert [f.value for f in fys] == [6, 14, 11]
    assert {f.fy_label for f in fys} == {"2024"}  # the raw label is identical on all three: unusable as a key


def test_ttm_rows_from_10qs_are_never_returned_as_fiscal_years(db):
    vals = [f.value for f in db.fiscal_years("BBB", "net_income", n=10)]
    assert 999 not in vals


def test_fiscal_years_returns_fewer_rather_than_padding(db):
    assert len(db.fiscal_years("AAA", "revenue", n=5)) == 2
    assert db.fiscal_years("AAA", "net_income", n=2) == []   # metric exists in the schema, no rows for this ticker


def test_lookups_are_parameterised_and_validated(db):
    with pytest.raises(ValueError):
        db.fiscal_years("AAA' OR '1'='1", "revenue")
    with pytest.raises(ValueError):
        db.fiscal_years("AAA", "ebitda")
    assert db.fiscal_years("aaa", "revenue", 1)[0].ticker == "AAA"  # case-insensitive ticker


# ---------------------------------------------------------------- quarters, YTD, derived Q4

def test_quarters_exclude_ytd_rows(db):
    q = db.quarters("AAA", "revenue", n=10)["facts"]
    assert all(f.basis == "quarter" for f in q)
    assert 460 not in [f.value for f in q] and 720 not in [f.value for f in q]
    assert [f.value for f in q][:3] == [310, 290, 270]  # newest first


def test_quarters_report_a_gap_where_q4_is_missing(db):
    res = db.quarters("AAA", "revenue", n=7)
    assert res["consecutive"] is False
    assert res["gaps"] and res["gaps"][0][0] == "2023-09-30" and res["gaps"][0][1] == "2024-03-31"


def test_derive_q4_is_annual_minus_nine_month_ytd_and_flagged(db):
    q4 = db.derive_q4("AAA", "revenue", "2023-12-31")
    assert q4.value == 1000 - 720 and q4.derived and q4.basis == "quarter_derived"
    assert q4.start_date == "2023-10-01" and q4.end_date == "2023-12-31"
    assert "derived" in q4.citation()


def test_derive_q4_returns_none_when_either_side_is_missing(db):
    assert db.derive_q4("AAA", "revenue", "2024-12-31") is not None            # 1200 - 870
    assert db.derive_q4("AAA", "revenue", "2022-12-31") is None                # no such fiscal year
    assert db.derive_q4("BBB", "net_income", "2024-11-03") is None             # annual exists, no YTD


def test_recent_quarters_fills_the_fiscal_year_gap_with_a_flagged_derived_q4(db):
    res = db.recent_quarters("AAA", "revenue", n=4)
    assert res["complete"] and res["n_derived"] == 1 and res["requested"] == 4
    vals = [(f.end_date, f.value, f.derived) for f in res["facts"]]
    # newest first: derived Q4'24 (1200 - 870), then Q3'24, Q2'24, Q1'24
    assert vals == [("2024-12-31", 330, True), ("2024-09-30", 310, False),
                    ("2024-06-30", 290, False), ("2024-03-31", 270, False)]


def test_recent_quarters_uses_all_contiguous_history_and_says_incomplete_when_short(db):
    res = db.recent_quarters("AAA", "revenue", n=12)
    ends = [f.end_date for f in res["facts"]]
    assert res["complete"] is False and len(ends) == 8          # Q1'23 .. derived Q4'24, nothing older exists
    assert ends == sorted(ends, reverse=True)


def test_recent_quarters_stops_at_a_real_hole_instead_of_skipping_it(tmp_path):
    rows = [
        row("HHH", "revenue", "2024", "Q1", "10-Q", "2024-01-01", "2024-03-31", 10),
        row("HHH", "revenue", "2024", "Q2", "10-Q", "2024-04-01", "2024-06-30", 20),
        # Q3 2024 (ends 2024-09-30) is missing, and so is the FY row needed to derive Q4
        row("HHH", "revenue", "2025", "Q1", "10-Q", "2025-01-01", "2025-03-31", 40),
        row("HHH", "revenue", "2025", "Q2", "10-Q", "2025-04-01", "2025-06-30", 50),
    ]
    write_csv(tmp_path, "HHH", rows)
    build_db(str(tmp_path), str(tmp_path / "h.sqlite"))
    d = FactsDB(str(tmp_path / "h.sqlite"))
    try:
        res = d.recent_quarters("HHH", "revenue", n=4)
        assert res["complete"] is False
        assert [f.value for f in res["facts"]] == [50, 40]   # stops at the hole; never averages across it
    finally:
        d.close()


# ---------------------------------------------------------------- build-time validation

def test_conflicting_duplicates_fail_the_build_but_identical_ones_collapse(tmp_path):
    good = [row("CCC", "revenue", "2024", "FY", "10-K", "2024-01-01", "2024-12-31", 5)] * 2
    write_csv(tmp_path, "CCC", good)
    assert build_db(str(tmp_path), str(tmp_path / "a.sqlite"))["rows"] == 1
    bad = [row("CCC", "revenue", "2024", "FY", "10-K", "2024-01-01", "2024-12-31", 5),
           row("CCC", "revenue", "2024", "FY", "10-K", "2024-01-01", "2024-12-31", 6)]
    write_csv(tmp_path, "CCC", bad)
    with pytest.raises(ValueError, match="conflicting"):
        build_db(str(tmp_path), str(tmp_path / "b.sqlite"))


def test_build_rejects_unknown_metrics_non_integers_and_missing_files(tmp_path):
    write_csv(tmp_path, "DDD", [row("DDD", "ebitda", "2024", "FY", "10-K", "2024-01-01", "2024-12-31", 5)])
    with pytest.raises(ValueError, match="unexpected metric"):
        build_db(str(tmp_path), str(tmp_path / "x.sqlite"))
    write_csv(tmp_path, "DDD", [row("DDD", "revenue", "2024", "FY", "10-K", "2024-01-01", "2024-12-31", "5.5")])
    with pytest.raises(ValueError):
        build_db(str(tmp_path), str(tmp_path / "y.sqlite"))
    with pytest.raises(FileNotFoundError):
        build_db(str(tmp_path / "empty_nonexistent"), str(tmp_path / "z.sqlite"))


def test_rebuild_is_idempotent(tmp_path):
    write_csv(tmp_path, "EEE", [row("EEE", "revenue", "2024", "FY", "10-K", "2024-01-01", "2024-12-31", 7)])
    p = str(tmp_path / "e.sqlite")
    assert build_db(str(tmp_path), p)["rows"] == build_db(str(tmp_path), p)["rows"] == 1


# ---------------------------------------------------------------- read-only SQL

def test_run_select_returns_columns_and_rows(db):
    r = db.run_select("SELECT ticker, val FROM facts WHERE ticker='AAA' AND basis='fiscal_year' ORDER BY end_date")
    assert r["columns"] == ["ticker", "val"] and r["rows"] == [["AAA", 1000], ["AAA", 1200]] and not r["truncated"]


def test_run_select_truncates_at_max_rows(db):
    r = db.run_select("SELECT * FROM facts", max_rows=3)
    assert len(r["rows"]) == 3 and r["truncated"]


@pytest.mark.parametrize("sql", [
    "DELETE FROM facts",
    "DROP TABLE facts",
    "INSERT INTO facts VALUES ('X','revenue','t','10-K','2024-01-01','2024-12-31',365,'fiscal_year',1,'2024','FY')",
    "UPDATE facts SET val = 0",
    "ATTACH DATABASE '/tmp/x.db' AS x",
    "PRAGMA writable_schema = ON",
    "SELECT 1; DROP TABLE facts",
    "SELECT 1; SELECT 2",
    "",
    "   ;  ",
    "WITH t AS (SELECT 1) DELETE FROM facts",
    "CREATE TABLE y (a)",
    "VACUUM",
])
def test_run_select_refuses_everything_that_is_not_one_select(db, sql):
    with pytest.raises(ValueError):
        db.run_select(sql)
    n = db.run_select("SELECT COUNT(*) FROM facts")["rows"][0][0]
    assert n == 16  # nothing was written


def test_run_select_cannot_read_other_files_or_load_extensions(db):
    for sql in ["SELECT load_extension('x')", "SELECT readfile('/etc/passwd')"]:
        with pytest.raises(ValueError):
            db.run_select(sql)


def test_run_select_step_limit_stops_a_runaway_query(db):
    with pytest.raises(ValueError):
        db.run_select("SELECT COUNT(*) FROM facts a, facts b, facts c, facts d, facts e", max_steps=5000)


def test_the_database_file_is_unchanged_after_all_of_the_above(db):
    con = sqlite3.connect(db.db_path)
    assert con.execute("SELECT COUNT(*) FROM facts").fetchone()[0] == 16
    con.close()


# ---------------------------------------------------------------- integration with the real CSVs

REAL_DIR = Path(__file__).resolve().parent.parent / "data" / "processed" / "xbrl"


@pytest.mark.skipif(not list(REAL_DIR.glob("*_facts.csv")), reason="real XBRL CSVs not present")
def test_tool_matches_an_independent_plain_python_read_of_the_real_csvs(tmp_path):
    import datetime as dt
    p = str(tmp_path / "real.sqlite")
    build_db(str(REAL_DIR), p)
    d = FactsDB(p)
    try:
        rows = []
        for f in sorted(glob.glob(str(REAL_DIR / "*_facts.csv"))):
            with open(f, encoding="utf-8") as fh:
                rows += list(csv.DictReader(fh))
        for t in ("AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "ORCL", "AVGO"):
            for m in ("revenue", "net_income"):
                annual = sorted(
                    [r for r in rows if r["ticker"] == t and r["metric"] == m and r["form"] == "10-K"
                     and (dt.date.fromisoformat(r["end"]) - dt.date.fromisoformat(r["start"])).days >= 300],
                    key=lambda r: r["end"], reverse=True)[:2]
                got = d.fiscal_years(t, m, n=2)
                assert [(g.end_date, g.value) for g in got] == [(r["end"], int(r["val"])) for r in annual], (t, m)
        # AVGO net income: three annual rows with distinct end dates despite identical raw fy labels
        avgo = d.fiscal_years("AVGO", "net_income", n=10)
        assert len({f.end_date for f in avgo}) == len(avgo) >= 2
        # every derived Q4 in the real data must be positive for revenue (a negative one would mean a bad subtraction)
        for t in ("AAPL", "MSFT", "GOOGL", "META", "NVDA", "ORCL", "AMZN", "AVGO"):
            rq = d.recent_quarters(t, "revenue", n=4)
            assert all(f.value > 0 for f in rq["facts"]), t
            assert [f.end_date for f in rq["facts"]] == sorted([f.end_date for f in rq["facts"]], reverse=True)
    finally:
        d.close()


def test_rebuild_removes_stale_sqlite_sidecar_files(tmp_path):
    write_csv(tmp_path, "FFF", [row("FFF", "revenue", "2024", "FY", "10-K", "2024-01-01", "2024-12-31", 7)])
    p = tmp_path / "f.sqlite"
    for suffix in ("-journal", "-wal", "-shm"):
        Path(str(p) + suffix).write_bytes(b"stale")
    build_db(str(tmp_path), str(p))
    assert not Path(str(p) + "-journal").exists() and not Path(str(p) + "-wal").exists()
    assert FactsDB(str(p)).fiscal_years("FFF", "revenue", 1)[0].value == 7
