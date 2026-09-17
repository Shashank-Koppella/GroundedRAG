"""
Diagnostic for the Day 1 ingestion output. Run locally after pipeline.py:

    python -m scripts.inspect_chunks

For each ticker's chunks.jsonl, groups chunks by (filing_date, form, item) and
prints word counts. Flags anything suspiciously short (< 500 words for a
target Item section is a red flag for real 10-K/10-Q filings — genuine
Item 1A / MD&A sections are almost always longer than that).
"""

import json
from collections import defaultdict
from pathlib import Path

CHUNKS_DIR = Path("data/processed/chunks")
SHORT_SECTION_WORD_THRESHOLD = 500


def main():
    for path in sorted(CHUNKS_DIR.glob("*.jsonl")):
        ticker = path.stem
        # (filing_date, form, item) -> total word count, chunk count
        sections = defaultdict(lambda: {"words": 0, "chunks": 0})

        with open(path, encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line)
                key = (rec["filing_date"], rec["form"], rec["item"])
                sections[key]["words"] += len(rec["text"].split())
                sections[key]["chunks"] += 1

        print(f"\n=== {ticker} ({len(sections)} filing/item combos) ===")
        flagged = 0
        for (filing_date, form, item), stats in sorted(sections.items()):
            flag = ""
            if stats["words"] < SHORT_SECTION_WORD_THRESHOLD:
                flag = "  <-- SUSPICIOUSLY SHORT"
                flagged += 1
            print(
                f"  {filing_date}  {form:5s}  {item:8s}  "
                f"{stats['words']:6d} words  {stats['chunks']:3d} chunks{flag}"
            )
        if flagged:
            print(f"  ** {flagged}/{len(sections)} section(s) under "
                  f"{SHORT_SECTION_WORD_THRESHOLD} words — likely truncated **")


if __name__ == "__main__":
    main()
