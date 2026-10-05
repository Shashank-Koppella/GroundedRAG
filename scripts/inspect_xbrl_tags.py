"""
List which XBRL concepts a company actually reports for revenue and net income, and which periods each
covers. Needs network access to data.sec.gov (run on your own machine).

    python -m scripts.inspect_xbrl_tags --tickers AVGO GOOGL ORCL

Why: AVGO's net income stopped at FY2024 in the facts table because its NetIncomeLoss facts come only
from one filing. The extractor now falls back to ProfitLoss; this prints the evidence for whether that is
the right concept (it should cover the missing quarters and FY2025, with values matching the filings'
"Net income" line), and shows every other net-income-like concept so nothing is guessed.
"""
import argparse
import collections
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.ingestion.config import COMPANIES, WINDOW_START  # noqa: E402
from src.ingestion.edgar_client import fetch_company_facts  # noqa: E402

PATTERN = re.compile(r"(NetIncome|ProfitLoss|Revenue|SalesRevenue)", re.I)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tickers", nargs="+", default=["AVGO"])
    args = ap.parse_args()
    for t in args.tickers:
        cik = COMPANIES[t][0]
        facts = fetch_company_facts(cik) or {}
        gaap = facts.get("facts", {}).get("us-gaap", {})
        print(f"\n===== {t} (CIK {cik}): concepts matching {PATTERN.pattern} with 10-K/10-Q facts ending on or after {WINDOW_START} =====")
        for concept in sorted(c for c in gaap if PATTERN.search(c)):
            usd = gaap[concept].get("units", {}).get("USD", [])
            rows = [f for f in usd if f.get("form") in ("10-K", "10-Q") and f.get("end", "") >= WINDOW_START.isoformat()]
            if not rows:
                continue
            periods = {(f.get("start"), f["end"]) for f in rows}
            annual = sorted({f["end"] for f in rows if f.get("form") == "10-K" and f.get("fp") == "FY"})
            filings = collections.Counter(f.get("accn") for f in rows)
            latest = max(rows, key=lambda f: f["end"])
            print(f"  {concept:60} periods={len(periods):3d} filings={len(filings):2d} "
                  f"10-K FY ends={annual[-4:]} latest end={latest['end']} val={latest['val']:,}")


if __name__ == "__main__":
    main()
