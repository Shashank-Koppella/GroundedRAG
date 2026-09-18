# Day 3 — Eval Harness

## What's built

- **`data/eval_set/eval_questions.json`** — 50 hand-designed questions: 20 single-hop
  (Item 1A/Item 7 prose, all 8 companies — **all 20 labeled with real gold `chunk_id`s**,
  see below), 20 multi-hop-numeric (XBRL-derived, including 2 genuinely cross-company
  comparisons and 2 non-trivial derived metrics), 10 out-of-scope (10 distinct refusal
  reasons, not 10 copies of the same one).
- **`src/eval/keyword_baseline.py`** — trivial term-overlap retriever, the floor every
  later retriever (BM25, dense, hybrid+RRF, reranked) has to beat.
- **`src/eval/scoring.py`** — Recall@k/MRR with bootstrapped 95% CIs, scored against any
  retriever exposing `.rank(query, ticker, k)`. Also includes `score_sql_routing_sanity()`
  — a pre-agent false-confidence probe on the multi-hop-numeric questions.
- **`src/eval/generate_embeddings.py`** — bge-base-en-v1.5 embedding generation, with the
  query/passage instruction-prefix asymmetry handled explicitly.
- **`src/eval/qdrant_setup.py`** — collection creation + upsert, dense-only for now,
  same collection gets sparse vectors added on Day 5 (no migration).
- **`scripts/label_eval_set.py`** — interactive local labeling helper for attaching real
  `gold_chunk_ids` to the single-hop questions.
- **`scripts/search_chunks.py`** — wider (top-20, full-text) fallback search for when
  `label_eval_set.py`'s top-8 keyword candidates don't surface the real answer.
- **`scripts/apply_reviewed_labels.py`** — applies Claude's reviewed gold-label judgments
  (made by reading real filing text) directly into `eval_questions.json`.
- **`tests/`** — 45 unit tests, all passing (12 for scoring, the rest for the ingestion
  pipeline and its bug fixes — see `CHANGES.md` at the repo root).

## Labeling status: DONE — 20/20 single-hop questions have real, verified gold labels

Labeled across three rounds, documented in full in `CHANGES.md`:
1. 11 questions from `label_eval_set.py`'s top-8 keyword candidates.
2. 8 more from `search_chunks.py`'s wider search, after the keyword baseline's top-8 missed
   the real answer for those.
3. The last one (`sh_019`, Oracle's Item 1A competition question) after fixing a real bug
   in `section_splitter.py` — Oracle's Item 1A header split mid-word across a tag boundary
   ("R" / "isk Factors"), so the section was never being extracted at all until the fix.

Every labeled question's `notes` field records the confidence level and the specific quote
that justified the label — worth a skim before trusting any single one for something that
matters (a few are MODERATE confidence, meaning plausible but not 100% textually confirmed).

**`gold_sql_description` fields are plain-English specs, not hardcoded numbers**, on
purpose: precise fiscal-year revenue/net-income figures live in your XBRL SQL table, and
these get consumed directly once the SQL tool exists in Phase B (Oct 8) — no separate
labeling step needed, since the "gold answer" for these is just "run this SQL query."

**Embedding generation and Qdrant stand-up are written but not run.** This sandbox's
network allowlist doesn't include `huggingface.co`, so the model download can't happen in
this chat's container. Run `generate_embeddings.py` locally, then `qdrant_setup.py`
against a local `docker run -p 6333:6333 qdrant/qdrant`.

## A design point worth stating in interviews

Recall@k/MRR only apply to the 20 single-hop questions. The 20 multi-hop-numeric questions
don't have a "gold chunk" — their correct behavior is SQL routing, not retrieval — so they're
scored separately via `score_sql_routing_sanity()`, which checks whether the baseline
retriever falsely returns a confident-looking prose chunk for a question that actually
needs a computed number. That's a real, measurable failure mode worth catching *before*
the agent exists, and it's a cleaner story than folding everything into one Recall@k number
and hand-waving the multi-hop questions in.
