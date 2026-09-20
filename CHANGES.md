# CHANGES — code review pass (this session)

Full repo pulled from GitHub and reviewed file-by-file. Your existing 16 tests were run
against your actual code first, unmodified — all 16 passed, confirming the Day 1 log's
six-bug debugging narrative is accurate. `section_splitter.py`, `chunker.py`, and
`edgar_client.py` were not touched.

## Real bugs found and fixed

**1. `src/ingestion/pipeline.py` — chunk records had no unique ID and dropped word-offset data**
The JSONL writer built each record as `{"text": ..., "chunk_index": ..., **c.metadata}` —
`chunk_index` resets to 0 for every section (it's local to each `chunk_text()` call), so it
collides across filings/items, and it never included the `Chunk` dataclass's `start_word`/
`end_word` at all. Added `make_chunk_id(ticker, accession_number, item_key, chunk_index)` —
globally unique, and pulled out as its own function specifically so it's unit-testable
without network access. `start_word`/`end_word` are now written into every record.
See `tests/test_pipeline.py`.

**2. `src/ingestion/xbrl_extractor.py` — silently dropped financial periods on a tag change**
`_first_available_tag` picked only the *first* GAAP tag with any data and used exclusively
that tag for the whole company. If a company's revenue tag changed partway through the
4-year lookback window (a real, documented EDGAR behavior around ASC 606 and other
transitions), the periods under the other tag were silently dropped, not merged. Replaced
with `_all_available_tag_facts()`, which collects from every candidate tag and lets
downstream de-duplication (on ticker/metric/fy/fp/form/end) prefer the earlier-listed tag
on genuine conflicts. Also fixed a related fragility: an empty result used to return a
DataFrame with no columns at all, which breaks on `df["metric"]`-style access; now always
returns the full expected schema even when empty. See `tests/test_xbrl_extractor.py`.

**2b. `src/ingestion/xbrl_extractor.py` — same real fiscal period counted multiple times
(found against live AVGO data, not anticipated up front)**
After running the pipeline against real EDGAR data, `AVGO_facts.csv` had 33 rows where
~16-18 were expected. Root cause, found by inspecting the raw CSV directly (same
diagnose-before-fixing discipline as the Day 1 section-splitter bugs): SEC's company-facts
JSON reports each real fact multiple times — a 10-K reports its own fiscal year AND prior-
year comparatives, and each comparative appearance is tagged with THAT filing's `fy`, not
the value's own true fiscal year. A period ending 2022-10-30 showed up tagged fy=2022
(the original 10-K), then again as fy=2023, and fy=2024 (twice) from later filings
reporting it as a comparative. The dedup key from fix #2 above still included `fy`, so all
four survived as if they were different periods — this directly threatened the multi-hop-
numeric eval questions ("two most recent fiscal years"), which need real periods to be
distinct and correctly counted. Fixed: dedup key became `(ticker, metric, form, end)` —
`end` is the real, reliable period identifier; `fy` is not, since it reflects which filing
reported the value rather than the value's own period. Among duplicates, the
earliest-`filed` occurrence is kept (the original filing, not a later restatement copy).
If duplicates for the same real period report genuinely DIFFERENT values, that's now
logged as a warning rather than silently resolved, since that would be an actual
restatement worth a manual look, not a repeat-reporting artifact.

**2c. `src/ingestion/xbrl_extractor.py` — fix #2b's dedup key was still too loose (found
against the actual pipeline run across all 8 companies)**
Running the pipeline for real flooded the logs with "2-3 DIFFERENT reported values" warnings
on nearly every 10-Q period, for every company, for both metrics — too uniform across the
whole corpus to be genuine restatements. Root cause: duration-type XBRL facts (revenue, net
income) are defined by a (start, end) PAIR, not by `end` alone. A 10-Q routinely reports the
SAME metric under the SAME tag with the SAME end date but TWO different start dates — once
for the standalone quarter ("three months ended") and once for the fiscal-year-to-date
cumulative ("six/nine months ended"). Both numbers are correct and real; they are NOT
duplicates. Fix #2b's key (`end` only, no `start`) collapsed these into false conflicts.
Fixed: dedup key is now `(ticker, metric, form, start, end)`, so quarter-only and
YTD-cumulative facts correctly survive as separate rows. **Known follow-up, deliberately not
solved here** (matches Day 1's scope — extract and save, don't build the SQL table yet):
this table now legitimately contains both quarterly and YTD figures for many 10-Q periods,
distinguishable only by `start`; a "quarterly revenue" question needs the ~1-quarter-duration
row specifically, which is a Phase B (Day 8) SQL-table-build concern, not a Day 1 one.

**2d. Bug in the fix for 2c itself, caught by its own regression test**: adding `start` to
the conflict-detection `groupby()` call meant any fact with `start=None` (common for real
10-K annual facts, and NaN generally) silently dropped out of that group entirely —
`pandas.groupby()` excludes NaN-keyed rows by default. The actual dedup (`drop_duplicates`)
wasn't affected, only the diagnostic warning silently stopped firing for facts with a
missing `start`. Fixed with `groupby(..., dropna=False)`.

See `tests/test_xbrl_extractor.py` — 5 new tests across 2b/2c/2d, including one that pins
the `dropna=False` fix specifically so it can't silently regress.

**3. Day 3 eval-harness scripts didn't match this repo's real field names**
Built in a prior session without access to this repo, so they assumed `company`/`section`
fields. Your actual pipeline writes `ticker` and `item` (e.g. `"item_1a"`). Fixed across
`data/eval_set/eval_questions.json` (now has `ticker` + a machine-readable `target_item_key`
alongside the human-readable `target_section`), `src/eval/keyword_baseline.py`,
`src/eval/scoring.py`, `src/eval/generate_embeddings.py`, `src/eval/qdrant_setup.py`, and
`scripts/label_eval_set.py` — whose hardcoded chunks path (`data/processed/{ticker}/chunks.jsonl`)
was also wrong; your real path is `data/processed/chunks/{ticker}.jsonl`. Re-verified
end-to-end against synthetic chunks shaped exactly like your real pipeline output.

**4. `requirements.txt`** — uncommented `sentence-transformers`, `qdrant-client`, `rank_bm25`
now that Day 3 code actually needs them.

**5. `scripts/inspect_chunks.py`** — was summing each chunk's word count, which double-counts
the ~15% overlap between adjacent chunks and inflates every reported section length. Now
uses `max(end_word)` per section (only possible because fix #1 preserves those fields),
with a fallback + warning for chunks written before this fix.

## Test count

16 original → **45** (17 new: 4 for `make_chunk_id`, 11 for `xbrl_extractor.py` across
fixes 2, 2b, 2c, 2d, 2 for the Oracle mid-word title-split fix) + 12 for the Day 3 scoring
code = 45 passing, 0 failing.

## After applying this update

Two independent reasons to re-run the pipeline before doing anything else:

1. Your `data/processed/xbrl/*_facts.csv` files were generated before fixes 2c/2d and still
   have the quarter-vs-YTD false-conflict-warning problem (harmless — no bad data was written,
   `drop_duplicates` wasn't affected — but you'll want a clean log to see any real conflicts).
2. Your `data/processed/chunks/ORCL.jsonl` was generated before the Item 1A mid-word-split fix
   and has zero `item_1a` chunks for Oracle, with that content currently folded into `item_1`
   instead. Only ORCL is affected — every other company's chunks are unaffected by this fix.

Re-run `python -m src.ingestion.pipeline` (safe to re-run — overwrites `data/processed/`
deterministically) to regenerate everything with both fixes applied. Expect the XBRL warning
flood to disappear entirely, and ORCL's `inspect_chunks` output to show a normal `item_1a` row
per filing instead of none — then `sh_019` can finally be labeled.

## Day 3 labeling session (Sep 17) — findings

**3. GENUINE BUG, not previously documented — found, diagnosed, AND FIXED:
`src/ingestion/section_splitter.py` never detected Oracle's real Item 1A header.**
First surfaced via `python -m scripts.inspect_chunks`: all 4 ORCL 10-Ks show `item_1` and
`item_7` rows but no `item_1a` row at all, unlike every other company. Confirmed by
`scripts/label_eval_set.py`'s candidates for `sh_019`: both were 10-Q boilerplate deferring
to "the factors discussed in Part I, Item 1A ... of our Annual Report on Form 10-K" — i.e.
the system could find references TO Oracle's Risk Factors section but never extracted the
section itself. Root cause narrowed via `python -m scripts.inspect_matches ORCL --form 10-K
--index 0`: the match list jumped from `Item 1. Business` (gap of 158,622 characters — 5-10x
every other company's Item 1 span) straight to `Item 1B`, skipping Item 1A entirely. Exact
mechanism pinpointed via `scripts/diagnose_orcl_item1a.py`, which found the real header text
split **mid-word** across a tag boundary: `'Item 1A.\tR'` on one line, `'isk Factors'`
continuing on the next — a more extreme version of the Day 1 AMZN/META tag-split bug (Bug 5),
which only handled splits falling cleanly *between* words. The lookahead-to-next-line logic
only fired when the current line's remainder was **completely empty** (`not remainder.strip()`),
which is true for a whole-word split but false here — the remainder is `"R"`, a non-empty
single character — so the lookahead never triggered and Oracle's entire Item 1A section was
silently absorbed into Item 1.

**Fixed**: the lookahead now triggers whenever `has_title` is False (not just when remainder
is empty), and concatenates `remainder + next_line` **without inserting a space** before
re-checking title validity — `"R" + "isk Factors"` correctly reassembles into `"Risk Factors"`,
while the original empty-remainder case (`"" + next_line`) is unchanged. Regression tests:
`test_split_finds_title_split_mid_word_across_tag_boundary`,
`test_split_mid_word_title_combines_without_inserting_a_space`. **`sh_019` can now be labeled
— re-run the ingestion pipeline first (see "After applying this update" below), then
`scripts/label_eval_set.py` or `scripts/search_chunks.py` for Oracle specifically.**

**4. Documentation correction, not a code bug: the Day 1 log's list of companies affected by
the known 10-Q Item 1 truncation limitation (MSFT/GOOGL/NVDA/ORCL) is incomplete.** Real
`inspect_chunks` output shows META has the IDENTICAL symptom (~100-108 words, every 10-Q,
consistently) but was never listed. The underlying limitation and its accepted-tradeoff
reasoning (documented in `section_splitter.py`'s module docstring) still hold — this is a
correction to which companies it affects (5, not 4), not a new problem or a reason to revisit
the accept-as-is decision.

**5. Hand-labeling results for the 20 single-hop questions**: user ran
`scripts/label_eval_set.py` but skipped every question (0/20 labeled by hand). Claude
reviewed all 20 candidate sets against the actual filing text and applied labels via the new
`scripts/apply_reviewed_labels.py`. Round 1: 11 labeled directly from `label_eval_set.py`'s
top-8 keyword candidates, 9 left unresolved. Round 2, using `scripts/search_chunks.py`'s
wider (top-20, full-text) search: 8 more resolved with real filing text. Round 3, after the
section_splitter fix (bug #3 above) unblocked Oracle's Item 1A content: the last question,
`sh_019`, resolved. **Final: 20/20 single-hop questions labeled**, all with real,
verified gold `chunk_id`s and reasoning recorded in each question's `notes` field.

## Status as of Sep 17 (end of Day 3 labeling)

All 20 single-hop questions are labeled with real, verified gold `chunk_id`s against the
actual corpus. `gold_sql_description` fields for the 20 multi-hop-numeric questions remain
specs rather than hardcoded numbers on purpose — those get consumed directly once the SQL
tool exists in Phase B (Oct 8); no separate labeling step is needed for them. The eval
harness (Recall@k/MRR/bootstrap CI + keyword baseline) is built, tested (45/45 passing), and
ready to score any retriever built from Day 5 onward.

## Day 5 follow-up: two eval-set gold-label corrections found via the near-miss diagnostic

Running `scripts/diagnose_day5_ceiling.py` flagged 3 questions where a chunk adjacent to
gold sat in the hybrid retriever's top-10 without gold itself being counted (a possible
Recall@k understatement from single-chunk labeling on overlapping-window chunks). Reading
all three adjacent chunks' actual text against their gold chunks directly (not guessed)
produced three different outcomes, which is itself worth recording: the diagnostic flags
candidates, it doesn't validate them.

**sh_001 (AAPL, China manufacturing risk): false positive, no change.** The adjacent chunk
is about an unrelated risk (new-product-transition risk), not China/manufacturing. The
structural adjacency was coincidental, not a content near-miss.

**sh_003 (AAPL, foreign currency risk): genuine near-miss, gold widened from 2 to 3 chunk
ids.** `AAPL_000032019325000079_item_1a_0041` and `_0042` are two overlapping-window
chunks of one continuous foreign-currency-risk paragraph — `_0041` states the general USD
exposure, `_0042` continues with hedging/derivative detail. Both genuinely answer the
question; scoring only `_0042` penalized a retriever for finding an equally-correct chunk
from the same passage. `_0041` added to the existing 2-chunk-id list (the pre-existing
second id, the near-identical passage in the 2024 filing, is untouched).

**sh_011 (AMZN, AWS segment operating income driver): genuine Day 3 labeling BUG, not a
near-miss — corrected, not widened.** The originally-labeled gold chunk
(`AMZN_000101872426000004_item_7_0025`) discusses the FTC lawsuit settlement and the
North America/International segments' operating income. It never mentions AWS. The actual
answer — "the increase in AWS operating income in 2025...is primarily due to increased
sales, partially offset by spending on technology infrastructure" — is in `_0026`, one
chunk further than the diagnostic's own +/-1 adjacency check looked. Found by reading
forward past what the automated check flagged, not by the check itself. Gold corrected to
`_0026` alone; the old id did not partially answer the question, so it isn't kept as a
second id the way sh_003's was.

**What this is worth recording, beyond the two fixes:** an automated near-miss diagnostic
that flags "something adjacent scored well" is a lead, not a verdict — it can point at a
real chunking artifact (sh_003), a real labeling bug it wasn't even designed to catch
(sh_011, found by reading past the flagged chunk), or nothing at all (sh_001). All three
required reading the actual filing text before touching a label, same discipline as every
other gold-label decision in this project.
