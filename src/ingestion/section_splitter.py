"""
Split a filing's raw HTML into plain text, then into the Item sections we
care about (Item 1, 1A, 7 for 10-Ks; Item 1, 1A, 2 for 10-Qs).

KNOWN LIMITATIONS (flag these in interviews, don't pretend they aren't there):

1. TOC entries. Modern EDGAR filings list every Item in a table of contents
   near the top, so a naive "find the first occurrence of 'Item 1A'" grabs
   the TOC entry, not the actual section.

2. Inline cross-references (the more important one — found via real EDGAR
   data during Phase A, Day 1, not anticipated up front). MD&A and Risk
   Factors sections routinely contain boilerplate that references OTHER
   item numbers mid-paragraph — e.g. "...risks described under Item 1A of
   this report...". An earlier version of this matcher treated ANY
   occurrence of "Item N" anywhere in the flattened text as a section
   boundary, which meant a real section's END got clipped at the first
   such cross-reference inside its own body — not just its start being
   mis-picked. Real symptom this produced: Apple's and Alphabet's 10-Q
   MD&A (Item 2) was truncating to ~200-400 words every single quarter,
   consistently, because both companies open MD&A with a forward-looking-
   statements paragraph that references Item 1A early on.

3. Pagination artifacts (found via real EDGAR data — Microsoft's filings
   specifically). Some filings repeat the current section's header at the
   top of every rendered "page", so a single section is spread across
   several same-item header repeats. Fixed by treating a later occurrence
   of the SAME item as continuation, not a boundary — see
   _find_section_span below for the actual logic and why the earlier
   "pick the single occurrence with the largest gap" approach quietly kept
   one page and discarded the rest.

4. Bare index/TOC entries with an anomalous gap (found via real EDGAR data
   — Microsoft again, and the one that finally killed the gap-threshold
   approach as the primary signal). A "master TOC" line can be literally
   just "Item 1." with no title, and the gap-based filter (item #1 above)
   assumed TOC entries always sit close together — but if a divider like
   "PART I. FINANCIAL INFORMATION" happens to sit between two TOC lines,
   the gap between them can exceed the threshold by chance, making a bare
   TOC entry look like real content. The actual reliable signal, visible
   directly in real match data: every genuine section header carries its
   descriptive title inline ("ITEM 1. FINANCIAL STATEMENTS"), while every
   TOC entry and pagination-repeat artifact is bare ("Item 1.", "Item 1",
   "Item 3, 4"). Filtering to "has a real title" at the match-finding stage
   removes TOC entries AND bare pagination repeats in one pass, and as a
   side effect makes the same-item "stitching" in _find_section_span below
   mostly redundant (only the one titled header remains per section) —
   kept anyway for defense in depth against filings that DO repeat the
   full title on every page.
   This also resolves a second real issue for free: 10-Qs have two
   DIFFERENT real sections that both parse as "Item 1" — Part I's
   Financial Statements and Part II's Legal Proceedings. Both are
   genuinely titled, so title-filtering alone doesn't disambiguate them,
   but once the noise is removed, "first occurrence, bounded by the next
   DIFFERENT item" already picks Part I's correctly (it comes first).

FIX (for #1, #2, #4 above): only treat "Item N" as a header candidate when it is the ENTIRE
content of its own line (short, standalone) — a real header renders as its
own block in the flattened text; an inline cross-reference sits inside a
much longer paragraph line and gets filtered out. This one constraint fixes
both problems: TOC entries still match this filter (also short standalone
lines) but remain distinguishable from the real section via the gap
heuristic below, and inline cross-references no longer enter the match set
at all, so they can't falsely bound a section's end anymore.

This is still a heuristic, not a robust parser — genuinely short LEGITIMATE
sections exist too (e.g. many companies' 10-Q Item 1A is just "no material
changes from our Annual Report", correctly ~80-100 words) and can sit near
the gap threshold below. Worth spot-checking a sample after any threshold
change.

KNOWN, DELIBERATELY UNFIXED LIMITATION (decided Phase A Day 1, after 6
iterations against real data — documented rather than chased further):
10-Q Item 1 (Financial Statements) truncates to ~90-115 words for MSFT,
GOOGL, NVDA, ORCL, and META specifically (the original diagnosis found 4 of
these; real Day 3 data against the full 8-company corpus showed META has
the identical symptom too — a documentation correction, not a new bug).
Root cause, confirmed via scripts/inspect_matches.py: these filings' master
TOC has a bare "Item 1." entry whose gap to the next TOC line happens to
exceed both the TOC-gap-threshold AND get a false title borrowed from an
adjacent "PART I. FINANCIAL INFORMATION" divider (which is itself
title-cased, so the uppercase-start guard doesn't catch it). Each attempted
fix for this resurrected a previously-fixed bug elsewhere (chasing it broke
AMZN/META to zero chunks at one point). Accepted as-is because: (1) 10-K
Item 1 (Business) and all other target sections are unaffected and correct
for all 8 companies; (2) Item 1's content (raw financial statement tables)
is already covered better by the XBRL/SQL path than free-text retrieval
would cover it; (3) the eval question set's single-hop and multi-hop
numeric questions target Item 1A/Item 7/Item 2 and XBRL data, not Item 1
prose. If this becomes a real problem later: the fix is closing the
loophole structurally (require the borrowed title to ALSO independently
pass TOC_GAP_THRESHOLD_CHARS against ITS OWN next-match gap, not just
exist) rather than pattern-matching another one-off case.

BUG #7 — FOUND AND FIXED (Day 3, against live Oracle 10-K data): a real
Item 1A header can be split MID-WORD across a tag boundary, not just
between words. Oracle's 10-K renders "Item 1A." and the first letter of its
title in one tag/line ("Item 1A.\tR"), then the rest of the word in the
next ("isk Factors"). Bug 5's original fix (see above) only triggered its
lookahead-to-next-line when the current line's remainder was COMPLETELY
EMPTY, which is true for a whole-word split (Amazon/Meta's case) but false
here — the remainder is "R", a single non-empty character, so the lookahead
never fired and Oracle's entire Item 1A section (and by extension the
158,622 characters that should have been its own section) was silently
absorbed into Item 1 instead. Confirmed via scripts/inspect_matches.py:
the match list for Oracle's 10-K jumped directly from "Item 1. Business"
(with an anomalously huge gap to the next match) to "Item 1B", skipping
Item 1A entirely — and scripts/diagnose_orcl_item1a.py located the real
header text and its exact "R" / "isk Factors" split via repr() on the raw
characters. Fix: trigger the lookahead whenever has_title is False
(regardless of whether remainder is empty or a short fragment), and
concatenate remainder + the next line WITHOUT a space before re-checking
title validity — this correctly reconstructs "R" + "isk Factors" =
"Risk Factors" for the mid-word case while leaving the original
whole-word-split behavior unchanged (remainder == "" means combined ==
next_stripped, identical to before). Regression test:
test_split_finds_title_split_mid_word_across_tag_boundary.
"""

import re
from dataclasses import dataclass

from bs4 import BeautifulSoup

# Matches "Item N", "Item 1A.", etc. Used with re.match (anchored at the
# start of an already-stripped line), not finditer over raw text — see
# module docstring for why that distinction matters.
ITEM_HEADER_PATTERN = re.compile(
    r"item\s+(\d{1,2}[a-c]?)\.?\s*[—\-:]?\s*",
    re.IGNORECASE,
)

# A real header line is short ("Item 7. Management's Discussion and
# Analysis of Financial Condition and Results of Operations" is ~90 chars;
# the longest combined 10-K headers run a bit past that). A line this long
# containing an "Item N" match is prose, not a heading.
MAX_HEADER_LINE_CHARS = 200

TOC_GAP_THRESHOLD_CHARS = 400  # below this, treat the match as TOC noise

# A real header carries a title after "Item N." ("FINANCIAL STATEMENTS");
# a bare TOC/index/pagination entry doesn't ("Item 1.", "Item 3, 4"). Require
# at least this many letters remaining after the item-number match to count
# as a real header — small enough to not exclude short real titles, large
# enough to exclude "" and punctuation-only remainders like ", 4".
MIN_TITLE_LETTERS = 3


@dataclass
class ItemMatch:
    item_number: str  # e.g. "1a"
    start: int


def html_to_text(html: str) -> str:
    """
    Strip an SEC filing's HTML down to readable plain text.

    Prefers lxml (faster, more lenient with malformed HTML — SEC filings are
    not always well-formed) but falls back to the stdlib html.parser if lxml
    isn't installed, so a missing optional dependency degrades speed/
    robustness rather than crashing every single filing.
    """
    try:
        soup = BeautifulSoup(html, "lxml")
    except Exception:
        soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style"]):
        tag.decompose()
    text = soup.get_text(separator="\n")
    # collapse excess blank lines from table/div layout noise
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _has_real_title(remainder: str) -> bool:
    """True if what follows 'Item N.' looks like an actual title, not a
    bare index/TOC/pagination entry ("", ", 4", "1A")."""
    letters = re.findall(r"[A-Za-z]", remainder)
    return len(letters) >= MIN_TITLE_LETTERS


def _find_all_item_matches(text: str) -> list[ItemMatch]:
    """
    Find header candidates: lines whose ENTIRE stripped content is short,
    starts with "Item N", AND is followed by a real title — not any
    "Item N" occurring anywhere in the text, and not a bare index/TOC/
    pagination-repeat entry. See module docstring for why both constraints
    are needed.

    The title check looks at this line first, then — only if this line's
    remainder doesn't already pass the title check — scans forward past any
    blank lines (whitespace text nodes between tags produce these; skip up
    to 3) for a title on a later line, concatenated directly onto whatever
    remainder this line had (no space inserted). Real filings don't
    reliably keep the item number and its title in the same tag:
    get_text(separator="\\n") inserts a break at every tag boundary, so
    "<b>Item 1A.</b> <span>Risk Factors</span>" in two adjacent tags
    becomes two lines with blank lines between them from the whitespace in
    the source HTML. Some filers' templates split it this way on every
    single header (Amazon, Meta — the whole filing produced zero matches
    without this), some do it inconsistently (Oracle, Microsoft, Alphabet —
    specific sections went missing), some never do it (Apple, Nvidia,
    Broadcom — worked without this). Oracle goes a step further: its split
    can land MID-WORD, not just between words ("Item 1A." + "R" on one
    line, "isk Factors" continuing on the next) — concatenating remainder
    and the next line directly (not "remainder + ' ' + next_stripped") is
    what makes "R" + "isk Factors" reassemble into "Risk Factors" correctly
    for that case while leaving the empty-remainder / whole-word-split case
    unchanged.

    Two guards keep this from over-matching: the candidate line must NOT
    itself be another "Item N" match (two bare item numbers near each
    other is a TOC/index listing, not a split title) — and the combined
    (remainder + next line) text must start with an uppercase letter, since
    real section titles always do ("Risk Factors", "RISK FACTORS") while
    unrelated prose sitting after a TOC's last entry usually doesn't ("more
    information about our...").
    """
    lines = text.split("\n")
    line_offsets = []
    offset = 0
    for line in lines:
        line_offsets.append(offset)
        offset += len(line) + 1  # +1 accounts for the "\n" split() consumed

    matches = []
    for i, line in enumerate(lines):
        stripped = line.strip()
        if not stripped or len(stripped) > MAX_HEADER_LINE_CHARS:
            continue
        m = ITEM_HEADER_PATTERN.match(stripped)
        if not m:
            continue

        remainder = stripped[m.end():]
        has_title = _has_real_title(remainder)

        if not has_title:
            j = i + 1
            skipped = 0
            while j < len(lines) and not lines[j].strip() and skipped < 3:
                j += 1
                skipped += 1
            if j < len(lines):
                next_stripped = lines[j].strip()
                # Concatenate directly, no space: this covers two distinct real shapes.
                # (1) remainder == "" (Bug 5's original AMZN/META case): the WHOLE title is
                #     on the next line, so combined == next_stripped, unchanged from before.
                # (2) remainder is a short non-empty fragment, e.g. "R" (found via live ORCL
                #     data, Sep 17): the header's own line ends mid-word — "R" from "Item 1A.
                #     R" — and the rest of the word ("isk Factors") continues on the next
                #     line/tag. Concatenating without a space reconstructs "Risk Factors"
                #     correctly; a space would break it into two words. This is a more
                #     extreme version of Bug 5's tag-boundary split: there, the split fell
                #     cleanly between words; here, it falls inside one. Requiring the
                #     COMBINED string to start uppercase and pass the title-length check
                #     (rather than checking next_stripped alone) is what makes this work for
                #     mid-word splits, since the continuation line itself starts lowercase
                #     ("isk Factors") and would fail an isupper() check on its own.
                combined = remainder + next_stripped
                if (
                    next_stripped
                    and len(next_stripped) <= MAX_HEADER_LINE_CHARS
                    and not ITEM_HEADER_PATTERN.match(next_stripped)
                    and combined[:1].isupper()
                    and _has_real_title(combined)
                ):
                    has_title = True

        if not has_title:
            continue

        leading_ws = len(line) - len(line.lstrip())
        matches.append(
            ItemMatch(item_number=m.group(1).lower(), start=line_offsets[i] + leading_ws)
        )

    return matches


def _find_section_span(
    all_matches: list[ItemMatch], item_number: str, text_len: int
) -> tuple[int, int] | None:
    """
    For one target item: start = the first non-TOC occurrence; end = the
    next occurrence of a DIFFERENT item after that start.

    Deliberately does NOT treat a later occurrence of the SAME item number
    as an end boundary — some filings (Microsoft's, discovered via real
    data) repeat the current section's header at the top of every rendered
    "page", so the real section is spread across several same-item header
    repeats with real content between each. An earlier version of this
    function picked the single occurrence with the largest individual gap
    to its immediate next header, which for a paginated filing meant
    keeping exactly one page's worth of content and silently discarding
    the rest — same size (~500-1000 words) in every item, every quarter,
    which was the actual symptom. Treating same-item repeats as
    continuation rather than a boundary fixes that without needing to
    special-case pagination directly.
    """
    non_toc_same_item = []
    for i, m in enumerate(all_matches):
        if m.item_number != item_number:
            continue
        next_start = all_matches[i + 1].start if i + 1 < len(all_matches) else text_len
        gap = next_start - m.start
        if gap >= TOC_GAP_THRESHOLD_CHARS:
            non_toc_same_item.append(m)

    if not non_toc_same_item:
        return None
    start = non_toc_same_item[0].start  # earliest valid (non-TOC) occurrence

    # End = next header of a DIFFERENT item, checked against ALL matches
    # (not just non-TOC ones) — any real header legitimately bounds this
    # section regardless of how much text follows it in turn.
    end = text_len
    for m in all_matches:
        if m.start > start and m.item_number != item_number:
            end = m.start
            break

    return start, end


def split_into_items(text: str, target_items: dict[str, str]) -> dict[str, str]:
    """
    target_items: {"item_1a": "Risk Factors", ...} (see config.py)
    Returns: {"item_1a": "<section text>", ...} — missing key if not found.
    """
    matches = _find_all_item_matches(text)
    if not matches:
        return {}

    sections = {}
    for key in target_items:
        item_num = key.replace("item_", "")
        span = _find_section_span(matches, item_num, len(text))
        if span is None:
            continue
        start, end = span
        sections[key] = text[start:end].strip()

    return sections
