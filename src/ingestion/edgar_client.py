"""
Thin client for two SEC EDGAR endpoints:
  - data.sec.gov/submissions/       -> list of a company's filings
  - www.sec.gov/Archives/edgar/data -> the actual filing document (HTML)

Rate-limited to stay well under SEC's fair-access guidance, and sends a real
identifying User-Agent as SEC's automated-access rules require. Both are
non-negotiable — SEC will block a client that doesn't do this, and it's a
one-line fix so there's no reason to skip it.

NOTE: this needs network access to data.sec.gov / www.sec.gov, which the
Claude sandbox that generated this code cannot reach. Run and debug this
locally. If a request 403s, the first thing to check is SEC_USER_AGENT in
config.py — a placeholder or missing email will get you blocked.
"""

import time
import logging
from datetime import date, timedelta
from typing import Optional

import requests

from src.ingestion.config import (
    SEC_USER_AGENT,
    MIN_REQUEST_INTERVAL_SECONDS,
    SUBMISSIONS_URL,
    ARCHIVES_URL,
)

log = logging.getLogger(__name__)

_last_request_time = 0.0


def _rate_limited_get(url: str) -> requests.Response:
    """GET with a shared User-Agent and a minimum spacing between requests."""
    global _last_request_time
    elapsed = time.monotonic() - _last_request_time
    if elapsed < MIN_REQUEST_INTERVAL_SECONDS:
        time.sleep(MIN_REQUEST_INTERVAL_SECONDS - elapsed)

    resp = requests.get(url, headers={"User-Agent": SEC_USER_AGENT}, timeout=30)
    _last_request_time = time.monotonic()

    if resp.status_code == 429:
        log.warning("Rate limited by SEC on %s — backing off 5s and retrying once", url)
        time.sleep(5)
        resp = requests.get(url, headers={"User-Agent": SEC_USER_AGENT}, timeout=30)
        _last_request_time = time.monotonic()

    resp.raise_for_status()
    return resp


SUBMISSIONS_PAGE_URL = "https://data.sec.gov/submissions/{name}"


def _columns_to_rows(block: dict):
    return zip(block.get("form", []), block.get("accessionNumber", []), block.get("filingDate", []),
               block.get("primaryDocument", []))


def get_company_filings(cik: str, form_types: list[str], years_back: int = 4,
                        window_start: "date | None" = None, fetch=None,
                        window_end: "date | None" = None) -> list[dict]:
    """
    Return filings of the given form types for a company within the window, newest first.

    Each item: {accession_number, filing_date, form, primary_document}

    Oct 5 audit fix: the submissions JSON inlines only the ~1,000 most RECENT filings of ANY form type
    in `filings.recent`; older ones live in extra pages listed in `filings.files`. Heavy Form 4 filers
    (META, GOOGL) pushed their 2022-2024 10-Ks/10-Qs out of `recent`, so META's corpus started in
    2024-08 and GOOGL's in 2023-07 while the other six started in late 2022. Every page whose date
    range overlaps the window is now read and merged (deduplicated by accession number).
    `fetch` is injectable for tests (url -> parsed JSON).
    """
    fetch = fetch or (lambda u: _rate_limited_get(u).json())
    data = fetch(SUBMISSIONS_URL.format(cik=cik))
    cutoff = window_start or (date.today() - timedelta(days=365 * years_back))

    filings = data.get("filings", {})
    blocks = [filings.get("recent", {})]
    for page in filings.get("files", []) or []:
        try:
            page_to = date.fromisoformat(page.get("filingTo", "9999-12-31"))
        except ValueError:
            page_to = date.max
        if page_to >= cutoff and page.get("name"):
            blocks.append(fetch(SUBMISSIONS_PAGE_URL.format(name=page["name"])))

    results, seen = [], set()
    for form, accession, filing_date, primary_doc in (row for b in blocks for row in _columns_to_rows(b)):
        if accession in seen:
            continue
        if form not in form_types:
            continue
        try:
            f_date = date.fromisoformat(filing_date)
        except ValueError:
            continue
        if f_date < cutoff or (window_end is not None and f_date > window_end):
            continue
        seen.add(accession)
        results.append(
            {
                "accession_number": accession,
                "filing_date": filing_date,
                "form": form,
                "primary_document": primary_doc,
            }
        )

    results.sort(key=lambda r: r["filing_date"], reverse=True)
    return results


def fetch_filing_document(cik: str, accession_number: str, primary_document: str) -> str:
    """Download the raw HTML of a filing's primary document."""
    accession_nodash = accession_number.replace("-", "")
    cik_int = int(cik)  # archives path wants the CIK without leading zeros
    url = ARCHIVES_URL.format(
        cik_int=cik_int, accession_nodash=accession_nodash, primary_doc=primary_document
    )
    return _rate_limited_get(url).text


def fetch_company_facts(cik: str) -> Optional[dict]:
    """Download the full XBRL company-facts JSON for a company. None if unavailable."""
    from src.ingestion.config import COMPANY_FACTS_URL

    url = COMPANY_FACTS_URL.format(cik=cik)
    try:
        return _rate_limited_get(url).json()
    except requests.HTTPError as e:
        log.warning("No XBRL company facts for CIK %s: %s", cik, e)
        return None
