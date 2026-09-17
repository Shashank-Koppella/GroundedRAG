"""
Dump every header-line match (item_number, char position, and the line
itself) for ONE real filing, in document order. Use this to see exactly
what the matcher is finding on real EDGAR HTML instead of guessing from
symptoms.

Usage (run from project root, with venv active):

    python -m scripts.inspect_matches MSFT --form 10-Q --index 0

--index 0 = most recent matching filing for that ticker (default).
"""

import argparse

from src.ingestion.config import COMPANIES, FORM_TYPES, YEARS_BACK
from src.ingestion.edgar_client import get_company_filings, fetch_filing_document
from src.ingestion.section_splitter import html_to_text, _find_all_item_matches


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("ticker")
    parser.add_argument("--form", default="10-Q", choices=FORM_TYPES)
    parser.add_argument("--index", type=int, default=0, help="0 = most recent")
    args = parser.parse_args()

    ticker = args.ticker.upper()
    if ticker not in COMPANIES:
        raise SystemExit(f"Unknown ticker {ticker}. Options: {list(COMPANIES)}")
    cik, _name = COMPANIES[ticker]

    filings = get_company_filings(cik, [args.form], YEARS_BACK)
    if not filings:
        raise SystemExit(f"No {args.form} filings found for {ticker}")
    filing = filings[args.index]
    print(
        f"=== {ticker} {filing['form']} filed {filing['filing_date']} "
        f"(accession {filing['accession_number']}) ===\n"
    )

    html = fetch_filing_document(cik, filing["accession_number"], filing["primary_document"])
    text = html_to_text(html)
    matches = _find_all_item_matches(text)

    print(f"{len(matches)} header-line matches, in document order:\n")
    for i, m in enumerate(matches):
        end_of_line = text.find("\n", m.start)
        if end_of_line == -1:
            end_of_line = len(text)
        line_text = text[m.start:end_of_line].strip()

        gap_to_next = (
            (matches[i + 1].start - m.start) if i + 1 < len(matches) else len(text) - m.start
        )

        print(
            f"[{i:3d}] pos={m.start:8d}  item={m.item_number:4s}  "
            f"gap_to_next={gap_to_next:6d}  line={line_text!r}"
        )


if __name__ == "__main__":
    main()
