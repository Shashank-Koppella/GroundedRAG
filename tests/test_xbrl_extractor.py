from datetime import date, timedelta

from src.ingestion.xbrl_extractor import extract_key_metrics


def _fact(val, end, fy, fp="FY", form="10-K", start=None, filed=None):
    return {"val": val, "end": end, "fy": fy, "fp": fp, "form": form, "start": start, "filed": filed}


def test_extract_key_metrics_uses_single_tag_when_only_one_has_data():
    company_facts = {
        "facts": {
            "us-gaap": {
                "RevenueFromContractWithCustomerExcludingAssessedTax": {
                    "units": {"USD": [_fact(100, "2025-09-30", 2025)]}
                },
                "NetIncomeLoss": {"units": {"USD": [_fact(10, "2025-09-30", 2025)]}},
            }
        }
    }
    df = extract_key_metrics(company_facts, "AAPL", years_back=4)
    revenue_rows = df[df["metric"] == "revenue"]
    assert len(revenue_rows) == 1
    assert revenue_rows.iloc[0]["tag_used"] == "RevenueFromContractWithCustomerExcludingAssessedTax"


def test_extract_key_metrics_merges_across_tags_instead_of_dropping_older_periods():
    """
    Regression test for the real bug: the old version picked only the first
    candidate tag with ANY data and used exclusively that tag, silently
    dropping periods reported under a different candidate tag (e.g. a company
    that changed its revenue-recognition tag partway through the lookback
    window). This reproduces that shape: one fiscal year under the newer tag,
    an earlier fiscal year under an older fallback tag. Both must survive.
    """
    company_facts = {
        "facts": {
            "us-gaap": {
                # newer tag — only has the most recent fiscal year
                "RevenueFromContractWithCustomerExcludingAssessedTax": {
                    "units": {"USD": [_fact(500, "2025-09-30", 2025)]}
                },
                # older tag — has an earlier fiscal year within the window
                "Revenues": {
                    "units": {"USD": [_fact(400, "2023-09-30", 2023)]}
                },
                "NetIncomeLoss": {"units": {"USD": [_fact(50, "2025-09-30", 2025)]}},
            }
        }
    }
    df = extract_key_metrics(company_facts, "AAPL", years_back=4)
    revenue_rows = df[df["metric"] == "revenue"].sort_values("fy")

    assert len(revenue_rows) == 2, "both fiscal years must survive, from two different tags"
    assert set(revenue_rows["fy"]) == {2023, 2025}
    assert set(revenue_rows["tag_used"]) == {
        "Revenues",
        "RevenueFromContractWithCustomerExcludingAssessedTax",
    }


def test_extract_key_metrics_prefers_earlier_candidate_on_duplicate_period():
    """If two tags both report the SAME period (can happen during a transition
    year), the earlier-listed (preferred) tag should win, not an arbitrary one."""
    company_facts = {
        "facts": {
            "us-gaap": {
                "RevenueFromContractWithCustomerExcludingAssessedTax": {
                    "units": {"USD": [_fact(999, "2025-09-30", 2025)]}
                },
                "Revenues": {
                    "units": {"USD": [_fact(111, "2025-09-30", 2025)]}
                },
                "NetIncomeLoss": {"units": {"USD": [_fact(50, "2025-09-30", 2025)]}},
            }
        }
    }
    df = extract_key_metrics(company_facts, "AAPL", years_back=4)
    revenue_rows = df[df["metric"] == "revenue"]
    assert len(revenue_rows) == 1
    assert revenue_rows.iloc[0]["val"] == 999
    assert revenue_rows.iloc[0]["tag_used"] == "RevenueFromContractWithCustomerExcludingAssessedTax"


def test_extract_key_metrics_filters_out_of_window_facts():
    old_date = (date.today() - timedelta(days=365 * 6)).isoformat()
    company_facts = {
        "facts": {
            "us-gaap": {
                "RevenueFromContractWithCustomerExcludingAssessedTax": {
                    "units": {"USD": [_fact(100, old_date, 2019)]}
                },
                "NetIncomeLoss": {"units": {"USD": []}},
            }
        }
    }
    df = extract_key_metrics(company_facts, "AAPL", years_back=4)
    assert df[df["metric"] == "revenue"].empty


def test_extract_key_metrics_no_data_returns_empty_dataframe():
    company_facts = {"facts": {"us-gaap": {}}}
    df = extract_key_metrics(company_facts, "AAPL", years_back=4)
    assert df.empty


def test_extract_key_metrics_dedupes_same_period_reported_across_multiple_filings():
    """
    Regression test for the real bug found against live AVGO data: SEC's
    company-facts JSON reports the SAME real fiscal period multiple times —
    once in its own original filing, then again as a prior-year comparative
    in later filings, each tagged with THAT filing's fy (not the value's own
    true fiscal year). A period ending 2022-10-30 showed up tagged fy=2022
    (original), fy=2023, and fy=2024 (twice) in real data. The old dedup key
    included fy, so all four survived as if they were different periods.
    This reproduces that exact shape: one end date, four fy labels, four
    filed dates spread across the following two years.
    """
    company_facts = {
        "facts": {
            "us-gaap": {
                "RevenueFromContractWithCustomerExcludingAssessedTax": {
                    "units": {"USD": [
                        _fact(1000, "2022-10-30", fy=2022, filed="2022-12-15"),  # original filing
                        _fact(1000, "2022-10-30", fy=2023, filed="2023-12-14"),  # comparative in next year's 10-K
                        _fact(1000, "2022-10-30", fy=2024, filed="2024-12-13"),  # comparative again
                        _fact(1000, "2022-10-30", fy=2024, filed="2024-12-13"),  # exact duplicate too
                    ]}
                },
                "NetIncomeLoss": {"units": {"USD": [_fact(100, "2022-10-30", fy=2022, filed="2022-12-15")]}},
            }
        }
    }
    df = extract_key_metrics(company_facts, "AVGO", years_back=6)
    revenue_rows = df[df["metric"] == "revenue"]

    assert len(revenue_rows) == 1, "one real period must produce exactly one row, not four"
    assert revenue_rows.iloc[0]["fy"] == 2022, "should keep the label from the EARLIEST-filed (original) occurrence"
    assert revenue_rows.iloc[0]["val"] == 1000


def test_extract_key_metrics_keeps_distinct_periods_separate():
    """Sanity counterpart to the dedup test above: two genuinely DIFFERENT
    periods must NOT get collapsed into one just because dedup got stricter."""
    company_facts = {
        "facts": {
            "us-gaap": {
                "RevenueFromContractWithCustomerExcludingAssessedTax": {
                    "units": {"USD": [
                        _fact(1000, "2024-10-30", fy=2024, filed="2024-12-13"),
                        _fact(1200, "2025-10-30", fy=2025, filed="2025-12-12"),
                    ]}
                },
                "NetIncomeLoss": {"units": {"USD": []}},
            }
        }
    }
    df = extract_key_metrics(company_facts, "AVGO", years_back=4)
    revenue_rows = df[df["metric"] == "revenue"]
    assert len(revenue_rows) == 2
    assert set(revenue_rows["end"]) == {"2024-10-30", "2025-10-30"}


def test_extract_key_metrics_logs_warning_on_genuine_value_conflict(caplog):
    """If the SAME real period reports DIFFERENT values across filings (an
    actual restatement, not just a repeat), that must be surfaced as a
    warning, not silently resolved by picking one arbitrarily."""
    import logging
    company_facts = {
        "facts": {
            "us-gaap": {
                "RevenueFromContractWithCustomerExcludingAssessedTax": {
                    "units": {"USD": [
                        _fact(1000, "2022-10-30", fy=2022, filed="2022-12-15"),
                        _fact(1050, "2022-10-30", fy=2023, filed="2023-12-14"),  # different value!
                    ]}
                },
                "NetIncomeLoss": {"units": {"USD": []}},
            }
        }
    }
    with caplog.at_level(logging.WARNING):
        extract_key_metrics(company_facts, "AVGO", years_back=6)
    assert any("DIFFERENT reported values" in rec.message for rec in caplog.records)
