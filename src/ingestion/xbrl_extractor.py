"""
Pull structured financial facts (revenue, net income, etc.) from a company's
XBRL company-facts JSON and normalize into a long-format table: one row per
(tag, fiscal period, value). This is the raw material for the SQLite table
that Phase B, Day 8 builds — Day 1 just extracts and saves it, doesn't build
the DB yet (see plan Section 7).
"""

import logging
from datetime import date, timedelta

import pandas as pd

from src.ingestion.config import REVENUE_TAG_CANDIDATES, NET_INCOME_TAG_CANDIDATES

log = logging.getLogger(__name__)


def _facts_for_tag(company_facts: dict, tag: str) -> list[dict]:
    """Pull the USD-denominated facts for one us-gaap tag, if present."""
    try:
        units = company_facts["facts"]["us-gaap"][tag]["units"]
    except KeyError:
        return []
    return units.get("USD", [])


def _all_available_tag_facts(company_facts: dict, candidates: list[str]) -> list[tuple[str, dict]]:
    """
    Returns (tag_used, fact) for every fact found across ALL candidate tags that
    have data — not just the first tag that has any data.

    Bug this fixes: the original version picked the single first candidate tag
    with any facts and used ONLY that tag for the whole company, silently
    dropping any period reported under a different candidate tag. That's a real
    risk here, not a hypothetical one: revenue recognition tags changed
    industry-wide around ASC 606 adoption (~2018), and some companies use a
    different tag in different fiscal years even afterward. If a company's
    4-year lookback window straddles a tag change, the old version would
    silently return data for only part of the window. Collecting from every
    candidate tag and de-duplicating downstream (extract_key_metrics, on
    ticker/metric/fy/fp/form/end) fixes this while still preferring the
    earlier-listed (more current) tag name when two tags report the same
    period, since candidates are processed in preference order and
    pandas.drop_duplicates keeps the first occurrence.
    """
    results = []
    for tag in candidates:
        for fact in _facts_for_tag(company_facts, tag):
            results.append((tag, fact))
    return results


def extract_key_metrics(company_facts: dict, ticker: str, years_back: int = 4) -> pd.DataFrame:
    """
    Returns a long-format DataFrame:
    ticker | metric | tag_used | fy | fp | form | start | end | val

    Only 10-K/10-Q sourced facts, only within the lookback window. `metric`
    is our normalized name ("revenue", "net_income"); `tag_used` records
    which underlying GAAP tag actually had data, since it varies by company
    and era — worth keeping for debugging/interview questions about this.

    REAL BUG FOUND AND FIXED (via live AVGO data, Sep 17 — same discipline as
    the Day 1 section-splitter bugs: real symptom, real diagnostic, real fix):

    SEC's company-facts JSON reports each fact multiple times across filings,
    not once. A 10-K doesn't just report its own fiscal year — it also
    reports the prior year's (and often the year before that's) figures as
    comparatives, and each of those comparative appearances gets tagged with
    THAT FILING's "fy", not the value's own true fiscal year. Concretely:
    Broadcom's period ending 2022-10-30 showed up FOUR times in the raw
    facts — tagged fy=2022 (its own original 10-K), and again as fy=2023,
    fy=2024 (x2) from later 10-Ks reporting it as a comparative. The original
    dedup key (ticker, metric, fy, fp, form, end) treated these as four
    DIFFERENT rows because fy differed, so a single real period silently
    became several. This directly threatens the multi-hop-numeric eval
    questions ("two most recent fiscal years") — duplicate periods make
    "most recent two" ambiguous or wrong.

    Fix: `end` (the period's actual end date) is the reliable identifier of
    a real fiscal period; `fy` is not, because it reflects which filing
    reported the value rather than the value's own period.

    SECOND REAL BUG, found one layer deeper (via the actual pipeline run
    against all 8 companies, not anticipated up front): dropping `fy` from
    the dedup key was necessary but not sufficient. Duration-type XBRL facts
    (revenue, net income) are defined by a (start, end) PAIR, not by `end`
    alone — a 10-Q routinely reports the SAME metric under the SAME tag with
    the SAME end date but TWO different start dates: once for the
    standalone quarter ("three months ended") and once for the fiscal-
    year-to-date cumulative ("six/nine months ended"). Both are correct,
    both are real, and they are NOT duplicates of each other. Keying dedup
    on (ticker, metric, form, end) without `start` collapsed these into
    false "conflicting value" warnings on nearly every 10-Q period, for
    every company, for both metrics when run against real data — that
    uniformity across the whole corpus (not sporadic, not company-specific)
    was the tell that something structural was wrong, not that 8 companies
    had independently restated the same quarters.

    Fix: dedup key is now (ticker, metric, form, start, end) — a real
    period is defined by its full (start, end) span, so quarter-only and
    year-to-date-cumulative facts correctly survive as separate, distinct
    rows instead of being forced into a false conflict. Genuine same-period
    duplicates (identical start AND end, reported again later as a
    comparative) still collapse correctly, keeping the earliest-filed
    occurrence and logging a real warning only when start, end, form, AND
    metric all match but val still differs.

    KNOWN FOLLOW-UP, not fixed here (deliberately — matches the plan's
    Day 1 scope: extract and save, don't build the SQL table yet; that's
    Phase B, Day 8): this table now legitimately contains BOTH the
    quarter-only and year-to-date figures for many 10-Q periods,
    distinguishable only by `start`. A question like "quarterly revenue for
    Q2" needs the row whose (end - start) is ~1 quarter, not the YTD row —
    Phase B's SQL table build needs to select on duration, not just pull
    "the" row for a period. Left as an explicit TODO for that phase rather
    than silently guessed at now.
    """
    cutoff = date.today() - timedelta(days=365 * years_back)
    rows = []

    metric_sources = {
        "revenue": REVENUE_TAG_CANDIDATES,
        "net_income": NET_INCOME_TAG_CANDIDATES,
    }

    for metric_name, candidates in metric_sources.items():
        tag_facts = _all_available_tag_facts(company_facts, candidates)
        for tag_used, fact in tag_facts:
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
                    "filed": fact.get("filed"),  # used only to pick the original filing on dedup, not exposed as a "real" column below
                }
            )

    all_columns = ["ticker", "metric", "tag_used", "fy", "fp", "form", "start", "end", "val", "filed"]
    df = pd.DataFrame(rows, columns=all_columns)

    if not df.empty:
        dup_key = ["ticker", "metric", "form", "start", "end"]
        # Flag genuine restatements (same real period, different reported value)
        # before resolving duplicates — these are data-quality signals worth a
        # human look, not something to silently paper over. dropna=False is
        # required: pandas.groupby() drops rows with a NaN/None group key by
        # default, and `start` is None for plenty of real facts (10-K annuals
        # in particular) — without dropna=False those periods silently never
        # get checked for conflicts at all.
        conflicting = (
            df.groupby(dup_key, dropna=False)["val"].nunique().reset_index(name="n_distinct_vals")
        )
        conflicting = conflicting[conflicting["n_distinct_vals"] > 1]
        for _, row in conflicting.iterrows():
            log.warning(
                "%s: %s %s period %s to %s has %d DIFFERENT reported values across "
                "filings (possible restatement, not just a duplicate) — verify manually",
                row["ticker"], row["metric"], row["form"], row["start"], row["end"], row["n_distinct_vals"],
            )

        # Prefer the earliest-filed occurrence per real period (the original
        # filing, not a later comparative restatement) before dropping.
        df = df.sort_values("filed", na_position="last")
        df = df.drop_duplicates(subset=dup_key, keep="first")
        df = df.sort_values("end").reset_index(drop=True)

    return df[["ticker", "metric", "tag_used", "fy", "fp", "form", "start", "end", "val"]]
