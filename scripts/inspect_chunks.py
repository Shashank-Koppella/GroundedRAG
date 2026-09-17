"""
Diagnostic for the Day 1 ingestion output. Run locally after pipeline.py:

    python -m scripts.inspect_chunks

For each ticker's chunks.jsonl, groups chunks by (filing_date, form, item) and
prints word counts. Flags anything suspiciously short (< 500 words for a
target Item section is a red flag for real 10-K/10-Q filings — genuine
Item 1A / MD&A sections are almost always longer than that).

Word count uses max(end_word) across a section's chunks, not a sum of each
chunk's word count — chunks overlap by design (overlap_words=40, ~15% of
chunk_size_words=250), so summing per-chunk counts double-counts the
overlapping words and inflates the reported length. max(end_word) gives the
real, non-duplicated word count of the original section text. This only works
because pipeline.py now writes start_word/end_word into each chunk record —
if you're running this against chunks written before that fix, word counts
will silently fall back to the (inflated) sum instead.
"""

import json
from collections import defaultdict
from pathlib import Path

CHUNKS_DIR = Path("data/processed/chunks")
SHORT_SECTION_WORD_THRESHOLD = 500


def main():
    for path in sorted(CHUNKS_DIR.glob("*.jsonl")):
        ticker = path.stem
        # (filing_date, form, item) -> max end_word seen, chunk count, has_word_bounds
        sections = defaultdict(lambda: {"max_end_word": 0, "chunks": 0, "has_bounds": True, "fallback_sum": 0})

        with open(path, encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line)
                key = (rec["filing_date"], rec["form"], rec["item"])
                s = sections[key]
                s["chunks"] += 1
                s["fallback_sum"] += len(rec["text"].split())
                if "end_word" in rec:
                    s["max_end_word"] = max(s["max_end_word"], rec["end_word"])
                else:
                    s["has_bounds"] = False

        print(f"\n=== {ticker} ({len(sections)} filing/item combos) ===")
        flagged = 0
        for (filing_date, form, item), stats in sorted(sections.items()):
            words = stats["max_end_word"] if stats["has_bounds"] else stats["fallback_sum"]
            note = "" if stats["has_bounds"] else "  [no start/end_word in record — inflated, re-run pipeline]"
            flag = ""
            if words < SHORT_SECTION_WORD_THRESHOLD:
                flag = "  <-- SUSPICIOUSLY SHORT"
                flagged += 1
            print(
                f"  {filing_date}  {form:5s}  {item:8s}  "
                f"{words:6d} words  {stats['chunks']:3d} chunks{flag}{note}"
            )
        if flagged:
            print(f"  ** {flagged}/{len(sections)} section(s) under "
                  f"{SHORT_SECTION_WORD_THRESHOLD} words — likely truncated **")


if __name__ == "__main__":
    main()
