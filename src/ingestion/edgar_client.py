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


def get_company_filings(cik: str, form_types: list[str], years_back: int) -> list[dict]:
    """
    Return recent filings of the given form types for a company, newest first.

    Each item: {accession_number, filing_date, form, primary_document}

    The submissions endpoint returns "recent" filings inline (fine for our
    4-year window at these companies' filing cadence — a handful of 10-Ks +
    ~12-16 10-Qs each); it does NOT paginate into older filings the way the
    full submissions history does, which is a known limitation acceptable
    for this project's scope.
    """
    url = SUBMISSIONS_URL.format(cik=cik)
    data = _rate_limited_get(url).json()

    recent = data.get("filings", {}).get("recent", {})
    forms = recent.get("form", [])
    accessions = recent.get("accessionNumber", [])
    dates = recent.get("filingDate", [])
    primary_docs = recent.get("primaryDocument", [])

    cutoff = date.today() - timedelta(days=365 * years_back)

    results = []
    for form, accession, filing_date, primary_doc in zip(forms, accessions, dates, primary_docs):
        if form not in form_types:
            continue
        try:
            f_date = date.fromisoformat(filing_date)
        except ValueError:
            continue
        if f_date < cutoff:
            continue
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
