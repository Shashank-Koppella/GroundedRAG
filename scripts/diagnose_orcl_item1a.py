"""
scripts/inspect_matches.py showed ORCL's 10-K has NO titled match for Item 1A anywhere in
the document body — only its (correctly-rejected) TOC entry. The gap from Item 1 (Business)
to Item 1B is 158,622 characters, ~5-10x every other company's Item 1 span, strongly
suggesting Risk Factors content is inside that span with an undetected header.

This script searches directly for "RISK FACTORS" (case-insensitive) in the flattened text
and prints repr() of the raw characters immediately before/after each hit — repr() is the
point: it reveals tabs, non-breaking spaces, zero-width characters, or other formatting
that a plain print() would hide and that could be exactly why ITEM_HEADER_PATTERN or the
title check in section_splitter.py isn't matching it.

Usage:
    python -m scripts.diagnose_orcl_item1a --form 10-K --index 0
"""
import argparse

from src.ingestion.config import COMPANIES, FORM_TYPES
from src.ingestion.edgar_client import get_company_filings, fetch_filing_document
from src.ingestion.section_splitter import html_to_text


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--form", default="10-K", choices=FORM_TYPES)
    parser.add_argument("--index", type=int, default=0)
    args = parser.parse_args()

    cik, _name = COMPANIES["ORCL"]
    filings = get_company_filings(cik, [args.form], 4)
    filing = filings[args.index]
    print(f"=== ORCL {filing['form']} filed {filing['filing_date']} "
          f"(accession {filing['accession_number']}) ===\n")

    html = fetch_filing_document(cik, filing["accession_number"], filing["primary_document"])
    text = html_to_text(html)

    needle = "risk factors"
    lower = text.lower()
    start = 0
    hit_num = 0
    while True:
        idx = lower.find(needle, start)
        if idx == -1:
            break
        hit_num += 1
        before = text[max(0, idx - 80): idx]
        match_text = text[idx: idx + len(needle)]
        after = text[idx + len(needle): idx + len(needle) + 120]

        print(f"--- hit {hit_num} at char {idx} ---")
        print("BEFORE (repr): ", repr(before))
        print("MATCH  (repr): ", repr(match_text))
        print("AFTER  (repr): ", repr(after))

        # Also show the whole line(s) this sits on, the way the real header matcher sees it
        line_start = text.rfind("\n", 0, idx) + 1
        line_end = text.find("\n", idx)
        if line_end == -1:
            line_end = len(text)
        print("FULL LINE (repr):", repr(text[line_start:line_end]))
        print()

        start = idx + len(needle)

    print(f"Total 'risk factors' occurrences (case-insensitive): {hit_num}")
    print("\nAlso checking for the all-caps SEC convention 'ITEM 1A' specifically:")
    lower_item = text.lower()
    idx = lower_item.find("item 1a")
    count = 0
    start = 0
    while True:
        idx = lower_item.find("item 1a", start)
        if idx == -1:
            break
        count += 1
        line_start = text.rfind("\n", 0, idx) + 1
        line_end = text.find("\n", idx)
        if line_end == -1:
            line_end = len(text)
        print(f"  'item 1a' hit {count} at char {idx}, full line (repr): "
              f"{repr(text[line_start:line_end])}")
        start = idx + 7
    print(f"Total 'item 1a' occurrences (case-insensitive): {count}")


if __name__ == "__main__":
    main()
