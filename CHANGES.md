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
distinct and correctly counted. Fixed: dedup key is now `(ticker, metric, form, end)` —
`end` is the real, reliable period identifier; `fy` is not, since it reflects which filing
reported the value rather than the value's own period. Among duplicates, the
earliest-`filed` occurrence is kept (the original filing, not a later restatement copy).
If duplicates for the same real period report genuinely DIFFERENT values, that's now
logged as a warning rather than silently resolved, since that would be an actual
restatement worth a manual look, not a repeat-reporting artifact. See
`tests/test_xbrl_extractor.py::test_extract_key_metrics_dedupes_same_period_reported_across_multiple_filings`.

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

16 original → **40** (12 new: 4 for `make_chunk_id`, 8 for `xbrl_extractor.py` — 5 from the
tag-merge fix, 3 from the duplicate-period fix) + 12 for the Day 3 scoring code = 40 passing,
0 failing.

## After applying this update

Your `data/processed/xbrl/*_facts.csv` files were generated BEFORE fix #2b and still have
the duplicate-period problem. Re-run `python -m src.ingestion.pipeline` (safe to re-run —
it overwrites `data/processed/` deterministically) to regenerate them with the fix applied,
before doing any numeric analysis or filling in `gold_sql_description` answers.

## What's still genuinely not done (not a bug — needs your real corpus)

`gold_chunk_ids` are empty for all 20 single-hop questions, and `gold_sql_description`
fields are specs rather than numbers, for the reason explained in `src/eval/README.md`:
this session never had your `data/processed/` chunks or XBRL figures, only the repo's
source code. Run `python -m src.ingestion.pipeline` locally, then
`python scripts/label_eval_set.py` to fill in real ground truth before Day 5.
