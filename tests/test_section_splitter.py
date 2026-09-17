from src.ingestion.section_splitter import html_to_text, split_into_items, _find_all_item_matches

# Synthetic filing shaped like a real 10-K: a tight table-of-contents block
# up top (short gaps between Item headers), then real sections with
# substantial body text between headers.
FAKE_10K_HTML = """
<html><body>
<div>
  <p>Item 1. Business</p>
  <p>Item 1A. Risk Factors</p>
  <p>Item 7. Management's Discussion and Analysis</p>
</div>
<div>
  <h2>Item 1. Business</h2>
  <p>{business_body}</p>
  <h2>Item 1A. Risk Factors</h2>
  <p>{risk_body}</p>
  <h2>Item 7. Management's Discussion and Analysis</h2>
  <p>{mda_body}</p>
  <h2>Item 8. Financial Statements</h2>
  <p>trailing content that should not leak into item 7</p>
</div>
</body></html>
""".format(
    business_body=" ".join(["business"] * 200),
    risk_body=" ".join(["risk"] * 200),
    mda_body=" ".join(["mda"] * 200),
)

TARGET_ITEMS_10K = {
    "item_1": "Business",
    "item_1a": "Risk Factors",
    "item_7": "Management's Discussion and Analysis",
}


def test_html_to_text_strips_tags():
    text = html_to_text(FAKE_10K_HTML)
    assert "<h2>" not in text
    assert "business" in text.lower()


def test_split_skips_toc_and_finds_real_sections():
    text = html_to_text(FAKE_10K_HTML)
    sections = split_into_items(text, TARGET_ITEMS_10K)

    assert set(sections.keys()) == {"item_1", "item_1a", "item_7"}

    # each extracted section should be dominated by its real body content,
    # not the one-line TOC entry
    assert sections["item_1"].count("business") >= 190
    assert sections["item_1a"].count("risk") >= 190
    assert sections["item_7"].count("mda") >= 190


def test_split_item7_does_not_leak_into_item8():
    text = html_to_text(FAKE_10K_HTML)
    sections = split_into_items(text, TARGET_ITEMS_10K)
    assert "trailing content" not in sections["item_7"]


def test_split_returns_empty_dict_when_no_items_found():
    sections = split_into_items("no item headers in this text at all", TARGET_ITEMS_10K)
    assert sections == {}


# --- Regression test for the real bug found against live EDGAR data ---
# Apple/Alphabet 10-Q MD&A (Item 2) opens with a forward-looking-statements
# paragraph that mentions "Item 1A" inline as a cross-reference. The old
# matcher (any "Item N" occurrence anywhere) treated that inline mention as
# a section boundary and truncated Item 2 to a couple hundred words, every
# single quarter. This reproduces that shape on synthetic data.
FAKE_10Q_WITH_CROSS_REFERENCE = """
<html><body>
<div>
  <p>Item 1. Financial Statements</p>
  <p>Item 1A. Risk Factors</p>
  <p>Item 2. Management's Discussion and Analysis</p>
</div>
<div>
  <h2>Item 1. Financial Statements</h2>
  <p>{financials_body}</p>
  <h2>Item 1A. Risk Factors</h2>
  <p>{risk_body}</p>
  <h2>Item 2. Management's Discussion and Analysis</h2>
  <p>This report contains forward-looking statements. Readers should
  consider the risks described under Item 1A of this report, among other
  factors, in evaluating our future results.</p>
  <p>{mda_body}</p>
</div>
</body></html>
""".format(
    financials_body=" ".join(["financials"] * 200),
    risk_body=" ".join(["risk"] * 200),
    mda_body=" ".join(["mda"] * 200),
)

TARGET_ITEMS_10Q = {
    "item_1": "Financial Statements",
    "item_1a": "Risk Factors",
    "item_2": "Management's Discussion and Analysis",
}


def test_split_does_not_truncate_at_inline_cross_reference():
    text = html_to_text(FAKE_10Q_WITH_CROSS_REFERENCE)
    sections = split_into_items(text, TARGET_ITEMS_10Q)

    assert "item_2" in sections
    # the real content ("mda" x200) must survive, not just the disclaimer
    # paragraph that mentions "Item 1A" inline
    assert sections["item_2"].count("mda") >= 190
    # and the inline cross-reference must not have been treated as a
    # boundary that clipped the section short
    assert len(sections["item_2"].split()) > 150


# --- Regression test for the MSFT pagination bug found against live data ---
# Microsoft's filings repeat the current section's header at the top of
# every rendered "page". The old approach (pick the single occurrence with
# the largest gap to its immediate next header) kept exactly one page and
# silently discarded the rest — same small size (~500-1000 words) across
# every item, every quarter, which is exactly what showed up in real output.
FAKE_10Q_WITH_PAGINATION_HTML = """
<html><body>
<div>
  <p>Item 1. Financial Statements</p>
  <p>Item 1A. Risk Factors</p>
  <p>Item 2. Management's Discussion and Analysis</p>
</div>
<div>
  <h2>Item 1. Financial Statements</h2>
  <p>{page1}</p>
  <h2>Item 1. Financial Statements</h2>
  <p>{page2}</p>
  <h2>Item 1. Financial Statements</h2>
  <p>{page3}</p>
  <h2>Item 1A. Risk Factors</h2>
  <p>{risk_body}</p>
  <h2>Item 2. Management's Discussion and Analysis</h2>
  <p>{mda_body}</p>
</div>
</body></html>
""".format(
    page1=" ".join(["pageone"] * 100),
    page2=" ".join(["pagetwo"] * 100),
    page3=" ".join(["pagethree"] * 100),
    risk_body=" ".join(["risk"] * 100),
    mda_body=" ".join(["mda"] * 100),
)


def test_split_stitches_repeated_same_item_headers_across_pages():
    text = html_to_text(FAKE_10Q_WITH_PAGINATION_HTML)
    sections = split_into_items(text, TARGET_ITEMS_10Q)

    assert "item_1" in sections
    # all three "pages" worth of content must be captured, not just page 1
    assert sections["item_1"].count("pageone") >= 90
    assert sections["item_1"].count("pagetwo") >= 90
    assert sections["item_1"].count("pagethree") >= 90
    # must stop at the next DIFFERENT item, not run past it
    assert "risk" not in sections["item_1"]
    assert "mda" not in sections["item_1"]


# --- Regression test for the real MSFT bug found via scripts/inspect_matches.py ---
# Real 10-Qs have: a bare "master TOC" (just "Item 1.", no title) where one
# gap between adjacent bare entries can exceed the gap threshold by chance
# (e.g. a "PART I. FINANCIAL INFORMATION" divider sitting between two TOC
# lines) — fooling a gap-only filter into treating the bare TOC entry as
# real content; bare "Item 1" pagination-repeat headers with no title; and
# a genuine Part I/Part II collision where "Item 1" means Financial
# Statements in Part I but Legal Proceedings in Part II. This reproduces
# that full shape on synthetic data.
FAKE_10Q_REAL_WORLD_SHAPE_HTML = """
<html><body>
<div>
Item 1.
{toc_divider}
Item 2.
Item 3.
Item 4.
Item 1.
Item 1A.
Item 2.
Item 5.
Item 6.
</div>
<div>
<h2>ITEM 1. FINANCIAL STATEMENTS</h2>
<p>{page1}</p>
<p>Item 1</p>
<p>{page2}</p>
<p>Item 1</p>
<p>{page3}</p>
<h2>ITEM 2. MANAGEMENT'S DISCUSSION AND ANALYSIS</h2>
<p>{mda_body}</p>
<h2>ITEM 3. QUANTITATIVE AND QUALITATIVE DISCLOSURES</h2>
<p>{market_risk_body}</p>
<h2>ITEM 4. CONTROLS AND PROCEDURES</h2>
<p>{controls_body}</p>
<p>Item 1, 1A</p>
<h2>ITEM 1. LEGAL PROCEEDINGS</h2>
<p>{legal_body}</p>
<h2>ITEM 1A. RISK FACTORS</h2>
<p>{risk_body}</p>
</div>
</body></html>
""".format(
    # a divider between two bare TOC entries, long enough to push that gap
    # past TOC_GAP_THRESHOLD_CHARS (400) purely by chance, same as real MSFT
    toc_divider=" ".join(["x"] * 120),
    page1=" ".join(["pageone"] * 100),
    page2=" ".join(["pagetwo"] * 100),
    page3=" ".join(["pagethree"] * 100),
    mda_body=" ".join(["mda"] * 100),
    market_risk_body=" ".join(["marketrisk"] * 100),
    controls_body=" ".join(["controls"] * 100),
    legal_body=" ".join(["legal"] * 100),
    risk_body=" ".join(["risk"] * 100),
)


def test_split_ignores_bare_toc_entry_with_anomalous_gap():
    text = html_to_text(FAKE_10Q_REAL_WORLD_SHAPE_HTML)
    sections = split_into_items(text, TARGET_ITEMS_10Q)

    assert "item_1" in sections
    # must be the REAL Part I Financial Statements content (all 3 pages),
    # not the bare TOC entry's 120-word divider
    assert sections["item_1"].count("pageone") >= 90
    assert sections["item_1"].count("pagetwo") >= 90
    assert sections["item_1"].count("pagethree") >= 90
    assert "x x x" not in sections["item_1"]


def test_split_resolves_part1_part2_item1_collision():
    text = html_to_text(FAKE_10Q_REAL_WORLD_SHAPE_HTML)
    sections = split_into_items(text, TARGET_ITEMS_10Q)

    # item_1 (Part I, Financial Statements) must NOT bleed into Part II's
    # "Item 1. Legal Proceedings", which reuses the same item number
    assert "legal" not in sections["item_1"]
    assert "mda" not in sections["item_1"]


def test_split_item2_and_item1a_unaffected_by_toc_and_collision():
    text = html_to_text(FAKE_10Q_REAL_WORLD_SHAPE_HTML)
    sections = split_into_items(text, TARGET_ITEMS_10Q)

    assert "item_2" in sections
    assert sections["item_2"].count("mda") >= 90
    assert "marketrisk" not in sections["item_2"]

    # item_1a target should resolve to Part II's real Risk Factors section
    assert "item_1a" in sections
    assert sections["item_1a"].count("risk") >= 90
    assert "legal" not in sections["item_1a"]


# --- Regression test for the AMZN/META bug: title split across HTML tags ---
# get_text(separator="\n") breaks at every tag boundary, not just block
# tags. Some filers put the item number and its title in adjacent-but-
# separate tags ("<b>Item 1A.</b> Risk Factors"), which lands them on two
# different lines once flattened. A title check that only looks at the
# SAME line as the item number misses these entirely — for Amazon and
# Meta specifically, this happened on every single header in every single
# filing, so the whole document produced zero matches.
FAKE_10K_WITH_SPLIT_TITLE_HTML = """
<html><body>
<div>
  <p>Item 1.</p>
  <p>Item 1A.</p>
  <p>Item 7.</p>
</div>
<div>
  <b>Item 1.</b>
  <span>Business</span>
  <p>{business_body}</p>
  <b>Item 1A.</b>
  <span>Risk Factors</span>
  <p>{risk_body}</p>
  <b>Item 7.</b>
  <span>Management's Discussion and Analysis</span>
  <p>{mda_body}</p>
</div>
</body></html>
""".format(
    business_body=" ".join(["business"] * 200),
    risk_body=" ".join(["risk"] * 200),
    mda_body=" ".join(["mda"] * 200),
)


def test_split_finds_title_split_onto_next_line_by_tag_boundary():
    text = html_to_text(FAKE_10K_WITH_SPLIT_TITLE_HTML)
    sections = split_into_items(text, TARGET_ITEMS_10K)

    assert set(sections.keys()) == {"item_1", "item_1a", "item_7"}
    assert sections["item_1"].count("business") >= 190
    assert sections["item_1a"].count("risk") >= 190
    assert sections["item_7"].count("mda") >= 190


def test_split_does_not_treat_two_bare_toc_entries_as_split_title():
    # guard: two consecutive bare "Item N." lines (a real TOC listing) must
    # NOT be treated as one entry's title borrowed from the next — that
    # would defeat the whole TOC filter this fix sits on top of.
    text = "Item 1.\nItem 1A.\nItem 7.\nmore text unrelated to any item"
    matches = _find_all_item_matches(text)
    assert matches == []
