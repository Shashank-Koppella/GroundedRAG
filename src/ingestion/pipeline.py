"""
Day 1 orchestrator: for each of the 8 companies, pull recent 10-K/10-Qs,
split into target Item sections, chunk the text, and separately pull XBRL
structured facts. Writes:

  data/processed/chunks/{ticker}.jsonl       one line per chunk, with metadata
  data/processed/xbrl/{ticker}_facts.csv     long-format structured facts
  data/processed/manifest.json               what was pulled, when, counts

Run locally (this needs network access to sec.gov, which the environment
that generated this code doesn't have):

    python -m src.ingestion.pipeline

Before running: fill in a real SEC_USER_AGENT in config.py.
"""

import json
import logging
import time
from pathlib import Path

from src.ingestion.config import (
    COMPANIES,
    FORM_TYPES,
    YEARS_BACK,
    TARGET_ITEMS_10K,
    TARGET_ITEMS_10Q,
)
from src.ingestion.edgar_client import (
    get_company_filings,
    fetch_filing_document,
    fetch_company_facts,
)
from src.ingestion.section_splitter import html_to_text, split_into_items
from src.ingestion.chunker import chunk_text
from src.ingestion.xbrl_extractor import extract_key_metrics

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

OUT_DIR = Path("data/processed")
CHUNKS_DIR = OUT_DIR / "chunks"
XBRL_DIR = OUT_DIR / "xbrl"


def process_company(ticker: str, cik: str) -> dict:
    log.info("=== %s (CIK %s) ===", ticker, cik)
    summary = {"ticker": ticker, "cik": cik, "filings_processed": 0, "chunks_written": 0}

    filings = get_company_filings(cik, FORM_TYPES, YEARS_BACK)
    log.info("%s: %d filings in window", ticker, len(filings))

    chunk_path = CHUNKS_DIR / f"{ticker}.jsonl"
    CHUNKS_DIR.mkdir(parents=True, exist_ok=True)

    with open(chunk_path, "w", encoding="utf-8") as out_f:
        for filing in filings:
            try:
                html = fetch_filing_document(
                    cik, filing["accession_number"], filing["primary_document"]
                )
            except Exception as e:
                log.warning("Failed to fetch %s %s: %s", ticker, filing["accession_number"], e)
                continue

            text = html_to_text(html)
            target_items = TARGET_ITEMS_10K if filing["form"] == "10-K" else TARGET_ITEMS_10Q
            sections = split_into_items(text, target_items)

            if not sections:
                log.warning(
                    "%s %s (%s): no target sections found — splitter heuristic may need "
                    "tuning for this filing's HTML structure",
                    ticker,
                    filing["form"],
                    filing["filing_date"],
                )

            for item_key, section_text in sections.items():
                base_metadata = {
                    "ticker": ticker,
                    "cik": cik,
                    "form": filing["form"],
                    "filing_date": filing["filing_date"],
                    "accession_number": filing["accession_number"],
                    "item": item_key,
                }
                chunks = chunk_text(section_text, metadata=base_metadata)
                for c in chunks:
                    record = {
                        "text": c.text,
                        "chunk_index": c.chunk_index,
                        **c.metadata,
                    }
                    out_f.write(json.dumps(record) + "\n")
                    summary["chunks_written"] += 1

            summary["filings_processed"] += 1

    log.info(
        "%s: wrote %d chunks from %d filings -> %s",
        ticker,
        summary["chunks_written"],
        summary["filings_processed"],
        chunk_path,
    )

    # --- XBRL structured facts (separate from the text pipeline above) ---
    XBRL_DIR.mkdir(parents=True, exist_ok=True)
    facts = fetch_company_facts(cik)
    if facts:
        df = extract_key_metrics(facts, ticker, YEARS_BACK)
        facts_path = XBRL_DIR / f"{ticker}_facts.csv"
        df.to_csv(facts_path, index=False)
        summary["xbrl_rows"] = len(df)
        log.info("%s: wrote %d XBRL fact rows -> %s", ticker, len(df), facts_path)
    else:
        summary["xbrl_rows"] = 0

    return summary


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    manifest = {"run_started": time.strftime("%Y-%m-%dT%H:%M:%S"), "companies": []}

    for ticker, (cik, name) in COMPANIES.items():
        try:
            summary = process_company(ticker, cik)
        except Exception:
            log.exception("Unrecoverable error processing %s — skipping", ticker)
            summary = {"ticker": ticker, "cik": cik, "error": "failed"}
        manifest["companies"].append(summary)

    manifest["run_finished"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    with open(OUT_DIR / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    log.info("Done. Manifest written to %s", OUT_DIR / "manifest.json")


if __name__ == "__main__":
    main()
