"""
Build the SQLite facts table from the XBRL CSVs and print a coverage report.

    python -m scripts.build_facts_db

Rebuilds data/processed/facts.sqlite from data/processed/xbrl/*_facts.csv (idempotent). The report
exists because coverage is uneven and silent gaps are the failure mode: read it before trusting a
numeric answer for a ticker.
"""
import argparse
import collections
import sqlite3
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.agent.facts_db import build_db  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv-dir", default=str(REPO_ROOT / "data" / "processed" / "xbrl"))
    ap.add_argument("--db", default=str(REPO_ROOT / "data" / "processed" / "facts.sqlite"))
    args = ap.parse_args()

    info = build_db(args.csv_dir, args.db)
    print(f"Built {info['db_path']}: {info['rows']} rows from {info['files']} files; by basis {info['by_basis']}")

    con = sqlite3.connect(args.db)
    rows = con.execute("SELECT ticker, metric, basis, COUNT(*), MIN(end_date), MAX(end_date) "
                       "FROM facts GROUP BY ticker, metric, basis ORDER BY ticker, metric, basis").fetchall()
    print("\nCoverage (rows per ticker / metric / basis, and the date range):")
    print(f"  {'ticker':6} {'metric':11} {'basis':12} {'rows':>4}  first end -> last end")
    for t, m, b, n, lo, hi in rows:
        print(f"  {t:6} {m:11} {b:12} {n:4d}  {lo} -> {hi}")

    print("\nThings to know before trusting an answer:")
    fy = collections.defaultdict(int)
    for t, m, n in con.execute("SELECT ticker, metric, COUNT(*) FROM facts WHERE basis='fiscal_year' GROUP BY ticker, metric"):
        fy[(t, m)] = n
    thin = sorted(k for k, v in fy.items() if v < 3)
    print(f"  - fiscal-year rows per ticker/metric (10-K annual): fewest = {min(fy.values()) if fy else 0}; "
          f"series with < 3: {thin or 'none'}")
    ttm = con.execute("SELECT COUNT(*) FROM facts WHERE basis='ttm'").fetchone()[0]
    print(f"  - {ttm} rows are trailing-twelve-month figures from 10-Qs; they are NOT fiscal years and are excluded from fiscal_years()")
    mislabeled = con.execute(
        "SELECT COUNT(*) FROM (SELECT ticker, metric, fy_label, fp_label, basis FROM facts "
        "GROUP BY ticker, metric, fy_label, fp_label, basis HAVING COUNT(*) > 1)").fetchone()[0]
    print(f"  - {mislabeled} (ticker, metric, fy label, fp label, basis) groups hold more than one row: the raw fy label is not unique, "
          "so periods are always identified by end date")
    miss = [t for t in ("AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "ORCL", "AVGO")
            if (t, "net_income") not in fy or (t, "revenue") not in fy]
    print(f"  - tickers missing a fiscal-year series entirely: {miss or 'none'}")
    print("  - Q4 never appears in a 10-Q; recent_quarters() derives it (annual minus nine-month YTD) and flags it")
    con.close()


if __name__ == "__main__":
    main()
