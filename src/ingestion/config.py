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
# companies, use different GAAP tags for the same concept). Try in order,
# take the first that has data. This is a real EDGAR gotcha, not overkill.
REVENUE_TAG_CANDIDATES = [
    "RevenueFromContractWithCustomerExcludingAssessedTax",
    "RevenueFromContractWithCustomerIncludingAssessedTax",
    "Revenues",
    "SalesRevenueNet",
]
NET_INCOME_TAG_CANDIDATES = ["NetIncomeLoss"]
