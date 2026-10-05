"""
Day 1 orchestrator: for each of the 8 companies, pull recent 10-K/10-Qs,
split into target Item sections, chunk the text, and separately pull XBRL
structured facts. Writes:

  data/processed/chunks/{ticker}.jsonl       one line per chunk, with metadata
  data/processed/xbrl/{ticker}_facts.csv     long-format structured facts
  data/processed/manifest.json               what was pulled, when, counts

Run locally (this needs network access to sec.gov, which the environment
that generated this code doesn't have):

    python -m src.ingestion.pipeline                                   # everything -> data/processed
    python -m src.ingestion.pipeline --out-dir data/processed_v2       # rebuild beside the current corpus
    python -m src.ingestion.pipeline --xbrl-only --out-dir data/processed_v2   # facts only (fast)
    python -m src.ingestion.pipeline --tickers AVGO META               # a subset

Rebuild into a separate --out-dir and compare with scripts/compare_corpus.py before replacing
data/processed: chunk ids are positional, so any change to the text pipeline can move gold labels.

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
    WINDOW_START,
    WINDOW_END,
    MIN_WORDS_10Q_ITEM_1,
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


def make_chunk_id(ticker: str, accession_number: str, item_key: str, chunk_index: int) -> str:
    """
    Globally unique, stable chunk identifier. `chunk_index` alone is NOT unique
    across a ticker's file — chunk_text() restarts it at 0 for every section, so
    two different filings' item_1a chunk 0 would otherwise collide. Accession
    number is per-filing-unique (SEC-assigned), so
    ticker + accession_number + item_key + chunk_index is unique across the
    entire corpus. Pulled out as its own function so it's unit-testable without
    network access (see tests/test_pipeline.py).
    """
    accession_nodash = accession_number.replace("-", "")
    return f"{ticker}_{accession_nodash}_{item_key}_{chunk_index:04d}"


def process_company(ticker: str, cik: str, out_dir: Path = OUT_DIR, xbrl_only: bool = False) -> dict:
    log.info("=== %s (CIK %s) ===", ticker, cik)
    summary = {"ticker": ticker, "cik": cik, "filings_processed": 0, "chunks_written": 0}
    chunks_dir, xbrl_dir = Path(out_dir) / "chunks", Path(out_dir) / "xbrl"
    if not xbrl_only:
        _process_text(ticker, cik, chunks_dir, summary)
    _process_xbrl(ticker, cik, xbrl_dir, summary)
    return summary


def _process_text(ticker: str, cik: str, CHUNKS_DIR: Path, summary: dict) -> None:

    filings = get_company_filings(cik, FORM_TYPES, YEARS_BACK, window_start=WINDOW_START, window_end=WINDOW_END)
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
            min_words = {"item_1": MIN_WORDS_10Q_ITEM_1} if filing["form"] == "10-Q" else None
            sections = split_into_items(text, target_items, min_words)

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
                        "chunk_id": make_chunk_id(
                            ticker, filing["accession_number"], item_key, c.chunk_index
                        ),
                        "text": c.text,
                        "chunk_index": c.chunk_index,
                        "start_word": c.start_word,
                        "end_word": c.end_word,
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



def _process_xbrl(ticker: str, cik: str, XBRL_DIR: Path, summary: dict) -> None:
    # --- XBRL structured facts (separate from the text pipeline above) ---
    XBRL_DIR.mkdir(parents=True, exist_ok=True)
    facts = fetch_company_facts(cik)
    if facts:
        df = extract_key_metrics(facts, ticker, YEARS_BACK, window_start=WINDOW_START, window_end=WINDOW_END)
        facts_path = XBRL_DIR / f"{ticker}_facts.csv"
        df.to_csv(facts_path, index=False)
        summary["xbrl_rows"] = len(df)
        log.info("%s: wrote %d XBRL fact rows -> %s", ticker, len(df), facts_path)
    else:
        summary["xbrl_rows"] = 0


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=str(OUT_DIR))
    ap.add_argument("--xbrl-only", action="store_true", help="re-extract XBRL facts only (no filing downloads)")
    ap.add_argument("--tickers", nargs="*", default=None, help=f"subset of {list(COMPANIES)}")
    args = ap.parse_args(argv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tickers = args.tickers or list(COMPANIES)
    unknown = [t for t in tickers if t not in COMPANIES]
    if unknown:
        raise SystemExit(f"unknown tickers {unknown}; choose from {list(COMPANIES)}")
    manifest = {"run_started": time.strftime("%Y-%m-%dT%H:%M:%S"), "window_start": WINDOW_START.isoformat(),
                "window_end": WINDOW_END.isoformat(), "tickers": tickers,
                "xbrl_only": args.xbrl_only, "companies": []}

    for ticker in tickers:
        cik, name = COMPANIES[ticker]
        try:
            summary = process_company(ticker, cik, out_dir, args.xbrl_only)
        except Exception:
            log.exception("Unrecoverable error processing %s — skipping", ticker)
            summary = {"ticker": ticker, "cik": cik, "error": "failed"}
        manifest["companies"].append(summary)

    manifest["run_finished"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    name = "manifest_xbrl.json" if args.xbrl_only else "manifest.json"
    if set(tickers) != set(COMPANIES):   # a partial run must not overwrite the full-corpus manifest
        name = name.replace(".json", "_" + "_".join(tickers) + ".json")
    with open(out_dir / name, "w") as f:
        json.dump(manifest, f, indent=2)

    log.info("Done. Manifest written to %s", out_dir / name)


if __name__ == "__main__":
    main()
