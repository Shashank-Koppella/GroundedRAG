"""
Pull structured financial facts (revenue, net income, etc.) from a company's
XBRL company-facts JSON and normalize into a long-format table: one row per
(tag, fiscal period, value). This is the raw material for the SQLite table
that Phase B, Day 8 builds — Day 1 just extracts and saves it, doesn't build
the DB yet (see plan Section 7).
"""

from datetime import date, timedelta

import pandas as pd

from src.ingestion.config import REVENUE_TAG_CANDIDATES, NET_INCOME_TAG_CANDIDATES


def _facts_for_tag(company_facts: dict, tag: str) -> list[dict]:
    """Pull the USD-denominated facts for one us-gaap tag, if present."""
    try:
        units = company_facts["facts"]["us-gaap"][tag]["units"]
    except KeyError:
        return []
    return units.get("USD", [])


def _first_available_tag(company_facts: dict, candidates: list[str]) -> tuple[str, list[dict]]:
    for tag in candidates:
        facts = _facts_for_tag(company_facts, tag)
        if facts:
            return tag, facts
    return "", []


def extract_key_metrics(company_facts: dict, ticker: str, years_back: int = 4) -> pd.DataFrame:
    """
    Returns a long-format DataFrame:
    ticker | metric | tag_used | fy | fp | form | start | end | val

    Only 10-K/10-Q sourced facts, only within the lookback window. `metric`
    is our normalized name ("revenue", "net_income"); `tag_used` records
    which underlying GAAP tag actually had data, since it varies by company
    and era — worth keeping for debugging/interview questions about this.
    """
    cutoff = date.today() - timedelta(days=365 * years_back)
    rows = []

    metric_sources = {
        "revenue": REVENUE_TAG_CANDIDATES,
        "net_income": NET_INCOME_TAG_CANDIDATES,
    }

    for metric_name, candidates in metric_sources.items():
        tag_used, facts = _first_available_tag(company_facts, candidates)
        if not facts:
            continue
        for fact in facts:
            if fact.get("form") not in ("10-K", "10-Q"):
                continue
            try:
                end_date = date.fromisoformat(fact["end"])
            except (KeyError, ValueError):
                continue
            if end_date < cutoff:
                continue
            rows.append(
                {
                    "ticker": ticker,
                    "metric": metric_name,
                    "tag_used": tag_used,
                    "fy": fact.get("fy"),
                    "fp": fact.get("fp"),
                    "form": fact.get("form"),
                    "start": fact.get("start"),
                    "end": fact.get("end"),
                    "val": fact.get("val"),
                }
            )

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.drop_duplicates(subset=["ticker", "metric", "fy", "fp", "form", "end"])
    return df
