"""
Shared config for the ingestion pipeline: the 8-company universe, form types,
lookback window, and SEC's required request etiquette.

CIKs are hardcoded rather than resolved dynamically from company_tickers.json
on every run — one less network call, and one less way for a ticker rename
or JSON schema change on SEC's end to silently break ingestion. Verified
against SEC EDGAR company search directly (see project notes) as of
2026-09-16.
"""

# ticker -> (CIK as 10-digit zero-padded string, display name)
COMPANIES = {
    "AAPL":  ("0000320193", "Apple Inc."),
    "MSFT":  ("0000789019", "Microsoft Corporation"),
    "GOOGL": ("0001652044", "Alphabet Inc."),
    "AMZN":  ("0001018724", "Amazon.com, Inc."),
    "META":  ("0001326801", "Meta Platforms, Inc."),
    "NVDA":  ("0001045810", "NVIDIA Corporation"),
    "AVGO":  ("0001730168", "Broadcom Inc."),
    "ORCL":  ("0001341439", "Oracle Corporation"),
}

FORM_TYPES = ["10-K", "10-Q"]
YEARS_BACK = 4  # "last 3-4 years" per plan; err toward more data
# Fixed window start (Oct 5 audit fix). The window used to be `date.today() - 4 years`, so every re-run
# silently dropped older periods and filings and the corpus was not reproducible. 2022-09-17 is the
# effective cutoff of the original Sep 2026 run: re-running reproduces that corpus (plus fixes).
import datetime as _dt
WINDOW_START = _dt.date(2022, 9, 17)
# Fixed window END as well: filings (and XBRL facts) filed after the original run are excluded, so a re-run
# reproduces the corpus the eval set was labelled against ("most recent fiscal year in the dataset" must not
# drift as new 10-Qs appear). Move it forward deliberately, then regenerate the reference answers.
WINDOW_END = _dt.date(2026, 9, 17)
# 10-Q Item 1 spans shorter than this are TOC fragments, not financial statements (section_splitter).
MIN_WORDS_10Q_ITEM_1 = 300

# --- SEC etiquette (required, not optional) ---
# SEC blocks/rate-limits generic user agents. Replace with your real name +
# email before running — SEC explicitly asks for a way to contact you if
# your traffic causes problems. Do not ship a fake value.
SEC_USER_AGENT = "GroundedRAG Capstone Project shashank.varma.koppella@gmail.com"
# SEC asks for <=10 req/sec; we stay well under that.
MIN_REQUEST_INTERVAL_SECONDS = 0.3

SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik}.json"
COMPANY_FACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"
ARCHIVES_URL = "https://www.sec.gov/Archives/edgar/data/{cik_int}/{accession_nodash}/{primary_doc}"

# Item sections we care about extracting from 10-Ks / 10-Qs for the
# unstructured-text retrieval corpus. 10-Q MD&A is Item 2, not Item 7.
TARGET_ITEMS_10K = {
    "item_1": "Business",
    "item_1a": "Risk Factors",
    "item_7": "Management's Discussion and Analysis",
}
TARGET_ITEMS_10Q = {
    "item_1": "Financial Statements",
    "item_1a": "Risk Factors",
    "item_2": "Management's Discussion and Analysis",
}

# XBRL tags for revenue vary across companies/years (older filings, and some
# companies, use different GAAP tags for the same concept). Facts from ALL candidate tags are collected;
# per period the earliest-filed fact wins and, within the same filing, the earlier-listed tag wins
# (deterministic since the Oct 5 audit; it used to depend on an unstable sort).
REVENUE_TAG_CANDIDATES = [
    "RevenueFromContractWithCustomerExcludingAssessedTax",
    "RevenueFromContractWithCustomerIncludingAssessedTax",
    "Revenues",
    "SalesRevenueNet",
]
# ProfitLoss (net income including noncontrolling interests) is the fallback for filers that do not use
# NetIncomeLoss in every filing: AVGO's NetIncomeLoss facts come only from its FY2024 10-K, so its quarters
# and FY2025 were missing (Oct 5 audit). Deliberately NOT included: NetIncomeLossAvailableToCommonStockholders*
# (subtracts preferred dividends: a different number). Verify per company with scripts/inspect_xbrl_tags.py.
NET_INCOME_TAG_CANDIDATES = ["NetIncomeLoss", "ProfitLoss"]
# Fallback-only tags fill periods that no primary tag reports; they never beat a primary tag for the same
# period, even when filed earlier (ProfitLoss includes noncontrolling interests, so it must not displace
# NetIncomeLoss just because it appeared in an earlier filing).
FALLBACK_ONLY_TAGS = {"ProfitLoss"}
