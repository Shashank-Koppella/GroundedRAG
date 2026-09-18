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


def test_extract_key_metrics_keeps_quarterly_and_ytd_facts_separate():
    """
    Regression test for the real bug found against the actual pipeline run
    across all 8 companies: a 10-Q reports the SAME metric under the SAME
    tag with the SAME end date but TWO different start dates — once for the
    standalone quarter ("three months ended") and once for the fiscal-
    year-to-date cumulative ("six months ended"). Both are correct, real,
    different numbers. The dedup key without `start` collapsed these into a
    false "conflicting value" warning on nearly every 10-Q period, for
    every company, for both metrics — this reproduces that exact shape:
    same end date, two different start dates, two different (both correct)
    values, no filed-date signal to fall back on.
    """
    company_facts = {
        "facts": {
            "us-gaap": {
                "RevenueFromContractWithCustomerExcludingAssessedTax": {
                    "units": {"USD": [
                        # standalone quarter: Jan 1 - Apr 1 (3 months)
                        _fact(100, "2023-04-01", fy=2023, fp="Q2", form="10-Q",
                              start="2023-01-01", filed="2023-05-01"),
                        # year-to-date cumulative: Oct 1 (fiscal year start) - Apr 1 (6 months)
                        _fact(220, "2023-04-01", fy=2023, fp="Q2", form="10-Q",
                              start="2022-10-01", filed="2023-05-01"),
                    ]}
                },
                "NetIncomeLoss": {"units": {"USD": []}},
            }
        }
    }
    df = extract_key_metrics(company_facts, "AAPL", years_back=6)
    revenue_rows = df[df["metric"] == "revenue"]

    assert len(revenue_rows) == 2, "quarter-only and YTD-cumulative facts must NOT be collapsed into one"
    assert set(revenue_rows["val"]) == {100, 220}
    assert set(revenue_rows["start"]) == {"2023-01-01", "2022-10-01"}


def test_extract_key_metrics_no_false_conflict_warning_for_quarterly_vs_ytd(caplog):
    """Counterpart to the test above: the quarter-vs-YTD case must NOT trigger
    the genuine-restatement warning, since it isn't a conflict at all."""
    import logging
    company_facts = {
        "facts": {
            "us-gaap": {
                "RevenueFromContractWithCustomerExcludingAssessedTax": {
                    "units": {"USD": [
                        _fact(100, "2023-04-01", fy=2023, fp="Q2", form="10-Q",
                              start="2023-01-01", filed="2023-05-01"),
                        _fact(220, "2023-04-01", fy=2023, fp="Q2", form="10-Q",
                              start="2022-10-01", filed="2023-05-01"),
                    ]}
                },
                "NetIncomeLoss": {"units": {"USD": []}},
            }
        }
    }
    with caplog.at_level(logging.WARNING):
        extract_key_metrics(company_facts, "AAPL", years_back=6)
    assert not any("DIFFERENT reported values" in rec.message for rec in caplog.records)


def test_extract_key_metrics_conflict_warning_fires_even_with_null_start(caplog):
    """
    Regression test for a bug in the conflict-warning code ITSELF, caught
    while fixing the quarter-vs-YTD issue above: pandas.groupby() silently
    DROPS rows whose group key contains NaN/None by default. Once `start`
    became part of the dedup/conflict grouping key, any fact with a missing
    `start` (common for real 10-K annual facts, and the default in this
    test file's _fact() helper) would silently never be checked for
    conflicts at all — the warning just wouldn't fire, with no error.
    Requires groupby(..., dropna=False). This exact scenario (two facts,
    same everything except val, start=None on both) is what the ORIGINAL
    restatement-warning test used and it passed for the wrong reason before
    `start` existed in the key — this test pins the dropna=False fix so it
    can't silently regress again.
    """
    import logging
    company_facts = {
        "facts": {
            "us-gaap": {
                "RevenueFromContractWithCustomerExcludingAssessedTax": {
                    "units": {"USD": [
                        _fact(1000, "2022-10-30", fy=2022, filed="2022-12-15"),  # start=None
                        _fact(1050, "2022-10-30", fy=2023, filed="2023-12-14"),  # start=None, different value
                    ]}
                },
                "NetIncomeLoss": {"units": {"USD": []}},
            }
        }
    }
    with caplog.at_level(logging.WARNING):
        extract_key_metrics(company_facts, "AVGO", years_back=6)
    assert any("DIFFERENT reported values" in rec.message for rec in caplog.records)
