# GroundedRAG — Optimized Project Plan & Knowledge Base

> **Purpose of this file:** Upload this to the "GroundedRAG" Claude Project knowledge base, replacing the previous version. Every new conversation started inside that Project will have access to it. Append each session's summary (template at the bottom) to a running log so this file doubles as post-project interview study material.
>
> **What changed from the original plan:** Two optimizations. First, the original 8-week timeline is compressed to the 13-day window (Oct 2–14) already reserved for this project in the 37-day interview-prep plan — full scope, tighter execution, no wasted motion. Second, every step now carries a **Model + Effort** recommendation, so Claude Pro quota goes toward the moments that actually need Opus-level reasoning instead of being spent evenly across trivial and hard tasks alike.

---

## ORIENTATION — read this first if you are a new chat session

**Repo:** [github.com/Shashank-Koppella/GroundedRAG](https://github.com/Shashank-Koppella/GroundedRAG)

**Today's actual calendar date this doc was last updated:** Sep 18, 2026 (still ahead of the official Oct 2 build-window start — Day 1, Day 3, and now Day 5 all landed early, in three extended prep sessions; this is intentional and not a schedule slip. See Section 2 for the Model + Effort tier system and Section 7 for the day-by-day plan these sessions map onto.)

**Where we are, in one line:** Phase A, Day 1 (ingestion), Day 3 (eval harness skeleton), and Day 5 (hybrid retrieval build) are all **DONE**. Day 4 (light reading day) is still **NOT STARTED** but is unblocking slack, not a prerequisite. Day 6 (buffer / hard-negative mining prep) is the next real task.

**Concrete state of the repo right now:**
- 8 companies ingested from live SEC EDGAR (AAPL, MSFT, GOOGL, AMZN, META, NVDA, AVGO, ORCL) — 11,245 chunks total, all 8 companies' Item 1A/Item 7/Item 2 sections clean, one documented/accepted limitation on 10-Q Item 1 for 5 of 8 companies (see Day 1 and Day 3 summaries below).
- 50-question eval set (20 single-hop / 20 multi-hop-numeric / 10 out-of-scope). All 20 single-hop questions have real, verified gold `chunk_id` labels — including two corrections made during Day 5 (see Day 5 summary: sh_003 widened to 3 gold ids, sh_011's gold label was found to be flatly wrong and corrected).
- Recall@k/MRR scoring + bootstrapped 95% CI code, tested against a keyword-search baseline.
- All 8 companies embedded with `bge-base-en-v1.5` (11,245 embedded chunks).
- Local Qdrant instance running in Docker, **with all 11,245 points confirmed present** (`groundedrag_chunks` collection) — this took a real fix during Day 5; see below. `src/eval/qdrant_setup.py` now prints a post-upload point count and warns if it doesn't match, specifically so this class of bug can't recur silently.
- **`src/retrieval/` is fully built**: `bm25_retriever.py`, `dense_retriever.py`, `hybrid_retriever.py` (hand-written RRF, k=60), `reranker.py` (`cross-encoder/ms-marco-MiniLM-L-6-v2`). All expose the same `.rank(query, ticker=None, k=10) -> List[chunk_id]` interface as the Day 3 keyword baseline, so `src/eval/scoring.py` runs against every one of them unchanged.
- **Day 5's full 4-stage ablation table is measured and final** (keyword baseline → BM25 → RRF hybrid → +reranker). See the Day 5 session summary below for the numbers and their interpretation — including a genuinely useful finding (the off-the-shelf reranker is currently net-negative on R@10 despite real headroom, which is the measured case for Phase C's LoRA fine-tune) and an honest scope limit (a 60% recall ceiling even with a perfect reranker, meaning ~40% of failures are a retrieval/chunking problem no reranker can fix).
- `scripts/run_day5_ablation.py` (full 4-stage table) and `scripts/diagnose_day5_ceiling.py` (recall-ceiling breakdown + hard-negative dump for Phase C) are both built, tested against the real corpus, and reusable — re-run either any time the retrieval stack changes.
- Test suite: **83/83 passing** (45 from Day 1/Day 3, 38 new from Day 5 — see Day 5 summary for the exact breakdown).
- **Opus sessions used: 3/5.** This is tighter than the plan intended — see the Day 5 summary's honest accounting of why one of those three was spent on a process mistake (switching to Opus before the code had even run once) rather than genuine reasoning work. **Practically: only 2 Opus sessions remain for 3 plan-flagged moments** (Phase B routing design, Phase C LoRA go/no-go, Phase D failure analysis) — the next session needs to make a call about consolidating two of those into one, or downgrading one to Sonnet extended thinking. Flag this explicitly at the start of Phase B.
- No known open code bugs. Two eval-set label corrections were made and documented (Day 5 summary + `CHANGES.md`); everything else that surfaced was investigated, and two of three flagged candidates turned out **not** to be bugs (documented as negative findings, not silently dropped).

**If you are a new chat picking this up:** read Section 2 (Model + Effort Guide) before doing anything, state the tier for your first task before starting it, then go straight to Section 7's Day 6 entry and the "Open questions / next steps" at the bottom of the Day 5 summary in the Session Summaries Log. Do not re-derive or re-litigate any decision documented in the Day 1, Day 3, or Day 5 summaries below — those bugs are fixed, tested, and closed.

---

## 1. Project Overview

**GroundedRAG** — an agentic RAG system over SEC filings, built with the evaluation harness as the centerpiece rather than an afterthought, shipping all three core pieces *and* all three stretch goals: guardrails, a fine-tuned reranker, and a deployed, observable system. Built as a portfolio project to replace a weaker ML project on the resume, targeting AI/ML engineering roles. This is the capstone referenced in the 37-day interview-prep plan (Sep 17 – Oct 23) — its build window is fixed at **Oct 2 – Oct 14**, after which the remaining days go to interview practice, not further building.

**Repo:** [github.com/Shashank-Koppella/GroundedRAG](https://github.com/Shashank-Koppella/GroundedRAG)

**Context:** Final year CS (Data Science), graduating April 2027.

**Relationship to prior work:** This project is a deliberate extension of an earlier VLM hallucination detector project (LLaVA-1.5-7B, attention-based probing, 5-fold CV, bootstrap testing, 0.906 ROC-AUC) — same eval-heavy, statistically rigorous mindset, applied to a RAG/agent system instead of a vision-language model. The throughline across both projects: measuring hallucination/faithfulness rigorously rather than eyeballing outputs.

**Background going in:** Comfortable with Python, PyTorch, Hugging Face Transformers, Scikit-learn. Full-stack experience (Django, Angular, REST APIs, MongoDB/SQL). New to RAG and agentic systems specifically.

**The scope call:** full scope — three core pieces plus all three stretch goals — inside 13 days is ambitious but achievable if the schedule stays disciplined: every day below has a fixed deliverable, and the two light days are the only slack in the system. The payoff is a genuinely complete portfolio piece: hybrid retrieval with a fine-tuned reranker, an agent with tool-use and injection guardrails, and a deployed system with observability — evaluated end to end at every stage, not just at the finish line.

---

## 2. Model & Effort Guide (read this before Day 1)

Claude Pro's usage limit is one shared pool across every model and feature in this Project — Opus and Sonnet draw from the same 5-hour and weekly budget, and Opus and extended thinking both consume that budget noticeably faster than standard Sonnet (Anthropic's own guidance: roughly 45 messages per 5-hour window on a lighter model, fewer on a heavier one — [support.claude.com/en/articles/8324991](https://support.claude.com/en/articles/8324991-about-claude-s-pro-plan-usage)). The numbers aren't published exactly and shift over time, so treat this as a relative-spend heuristic, not a fixed budget, and check **Settings → Usage** if you want the current picture. Fuller scope means more days that legitimately call for the higher tiers — that's fine, the discipline is in scoping *which* moments, not in avoiding them.

Three tiers, used deliberately rather than picking whatever's already open:

| Tier | Use for | Why |
|---|---|---|
| **Haiku 4.5, standard** | Boilerplate: README scaffolding, docstrings, commit messages, config files, renaming/formatting passes, simple data-pulling scripts | Rote generation work that doesn't need reasoning depth — the cheapest possible way to burn a message |
| **Sonnet 5, standard** | The default. Most implementation: chunking code, embedding pipeline, FastAPI routes, Streamlit/Angular UI, most debugging, the LoRA training script itself | Handles the large majority of this project's actual coding well; this is where most of your quota should go |
| **Sonnet 5, extended thinking** | Medium-difficulty design calls: the eval question taxonomy, interpreting ablation results, debugging a genuinely confusing bug, the guardrails threat model | A step up in reasoning without paying the Opus premium |
| **Opus 5, extended thinking** | Reserved for ~4 genuinely hard moments only (flagged below): the RRF/reranker ablation read, the agent routing-logic design, the LoRA fine-tune go/no-go decision, the final error-analysis synthesis | These are the moments where being wrong costs you a redesign, not a re-prompt — worth the quota |

**Budget guidance for the 13-day sprint:** aim for no more than 4–5 Opus-with-extended-thinking sessions across the whole project, each scoped tightly (one clear question, not an open-ended "help me build X"). Everything else defaults to Sonnet. With the full stretch scope back in, a couple of Sonnet-extended-thinking days will show up more often than in a minimal-scope version — that's the correct place for that extra reasoning to go, not Opus by default.

**Opus sessions used so far: 3/5** (as of end of Day 5 — see Session Summaries Log). **This is off the plan's intended pace**: the plan assumed 1 Opus session per flagged moment (4 total, one held in reserve), but Day 5 alone consumed 1 Opus session on a genuine ablation read (as intended) *and* a second one on a process mistake — switching to Opus before the code had even run once, so it was spent on a mechanical dependency-version fix instead of reasoning. See the Day 5 summary for the full account. **Practical consequence for Phase B onward: only 2 Opus sessions remain for 3 still-flagged moments** (agent routing-logic design, LoRA go/no-go, failure-analysis synthesis). Phase B should open by deciding how to handle this — likely candidates are downgrading the LoRA go/no-go to Sonnet extended thinking (Day 5's ceiling diagnostic already did most of that reasoning), or accepting that one flagged moment doesn't get Opus. Don't default into spending the 4th session without that decision being explicit.

**Session structure:** one new conversation per phase (four phases below), inside the GroundedRAG Project, so each session inherits this file plus all prior summaries without re-explaining context. Long single threads burn quota faster regardless of which model you're on — start a fresh conversation at each phase boundary even if the old one hasn't hit a limit yet.

**Standing instructions that apply to every session, not just this one (established Day 3, carry forward always):**
- **Always give run commands in full** — concrete, ready-to-run commands for every company/case needed, never a templated command with a placeholder (e.g. never `python script.py TICKER`; always the real ticker, spelled out for each of the 8 companies where relevant).
- **Any new or changed files are always delivered as a zip**, with `GroundedRAG/` as the top-level folder, so it extracts directly into the existing Desktop project folder without nesting or manual reorganizing. Verify with a clean-extraction test before sending.
- **State the Model + Effort tier before starting any task, every time, without being asked** (established Day 1, restated explicitly Day 5). When a session is manually switching models in the UI rather than an agent picking automatically, the tier stated in a reply is guidance for the model the *next* prompt should use, not the one that already produced the current reply — say so if there's any ambiguity.

---

## 3. Full Scope — everything ships

**Core (non-negotiable):**
1. **Hybrid RAG pipeline** — BM25 + dense embeddings + reranker over SEC filings
2. **Agent/tool-use layer** — the agent decides when to retrieve vs. use another tool (calculator, SQL query) vs. answer directly
3. **Custom evaluation harness** — Recall@k/MRR for retrieval, faithfulness/hallucination rate for generation (NLI-based), all reported with bootstrapped confidence intervals

**Stretch (all three, committed — not optional):**
4. **Prompt-injection guardrails** on retrieved text — extends the eval-heavy identity into safety, a genuinely hot topic for agentic systems right now
5. **LoRA/QLoRA fine-tuning** of the reranker on hard negatives mined from the Phase A ablation — the strongest ML-research signal in the project, and now evaluated with a clean before/after comparison against the baseline reranker
6. **Deployment** — FastAPI + Docker as the non-negotiable baseline, with basic observability (structured logging + a metrics endpoint) and an optional Angular frontend if Day 13 has room; AWS deploy is the one item that stays genuinely optional — treat it as a bonus, not a deliverable, since it adds infrastructure risk on the last two days for comparatively little interview value over a solid local Docker deployment

**Fallback order if a day genuinely runs long:** AWS deploy first, then the Angular frontend (Streamlit is already the working UI, so this is pure polish), then — only as a last resort — trim the LoRA fine-tune to fewer training steps rather than dropping it. The core three and guardrails are the floor; everything below that line is where schedule pressure gets absorbed.

---

## 4. Domain & Corpus: SEC 10-K/10-Q Filings

**What:** 8 companies across one sector (e.g. AAPL, MSFT, GOOGL, AMZN, META, NVDA, and two more from the same sector for a fuller comparison set), last 3–4 years of 10-Ks + 10-Qs.

**Source:** Free via SEC EDGAR — `data.sec.gov` for structured XBRL financial data, `efts.sec.gov` for full-text search. Public domain, no copyright/licensing concerns.

**Why this corpus, specifically:** It naturally splits into two data shapes, which is what makes the agent's tool-routing layer *necessary* rather than decorative:
- **Unstructured text** (Item 1 Business, Item 1A Risk Factors, Item 7 MD&A) → target for BM25 + dense + reranker retrieval
- **Structured financials** (revenue, net income, etc. from XBRL) → parsed into a small SQLite table → target for the SQL tool

A question like "what was Apple's risk factor around China manufacturing" needs retrieval. A question like "what was YoY revenue growth for Microsoft in FY24" needs SQL + calculator — a model answering from retrieved text alone will hallucinate the number. This distinction is the interview-defensible reason the agent needs tool-use at all.

**Full scope:** 8 companies from Day 1 — the extra breadth is worth it for the fine-tuned reranker's training signal in Phase C (more hard-negative diversity) and makes the finished system a more convincing demo.

**Final company list (locked in Day 1):** AAPL, MSFT, GOOGL, AMZN, META, NVDA, AVGO, ORCL. AVGO and ORCL were the two "same sector" companies left open by the original plan — added to round out 8 mega-cap tech names. CIKs verified against live EDGAR filings on 2026-09-16:

| Ticker | CIK | Name |
|---|---|---|
| AAPL | 0000320193 | Apple Inc. |
| MSFT | 0000789019 | Microsoft Corporation |
| GOOGL | 0001652044 | Alphabet Inc. |
| AMZN | 0001018724 | Amazon.com, Inc. |
| META | 0001326801 | Meta Platforms, Inc. |
| NVDA | 0001045810 | NVIDIA Corporation |
| AVGO | 0001730168 | Broadcom Inc. |
| ORCL | 0001341439 | Oracle Corporation |

---

## 5. Tool Stack

| Piece | Choice | Rationale |
|---|---|---|
| Vector DB | **Qdrant** (local Docker) | Native hybrid search (dense + sparse) with built-in fusion; free; same setup carries straight into the deployment stretch goal without migration |
| Embeddings | **BAAI/bge-base-en-v1.5** | Strong on retrieval benchmarks; runs on free Colab T4 or CPU at this corpus size; swapping models is ~2 lines, giving a free extra ablation for the eval harness if Phase A has room |
| Sparse/BM25 | `rank_bm25`, fused with dense via hand-written **Reciprocal Rank Fusion (RRF)** first | Doing this by hand (before reaching for Qdrant's built-in sparse vectors) means being able to explain the fusion math in an interview, not just "the vector DB handles it" — **built and tested Day 5, see summary below** |
| Reranker | `cross-encoder/ms-marco-MiniLM-L-6-v2`, baseline then LoRA-fine-tuned | Free, fast, CPU-friendly; the fine-tuning target for stretch goal #5 — **baseline measured Day 5: currently net-negative on R@10 versus not reranking at all, which is the measured case for the Phase C fine-tune, not a reason to drop the reranker step** |
| Agent LLM | **Groq free tier** (Llama 3.3 70B) | Fast, free, OpenAI-compatible function calling; used for both tool-routing decisions and answer generation |
| Faithfulness eval | **Local NLI model** (`cross-encoder/nli-deberta-v3-base`), not an LLM-judge | Free, deterministic, avoids the "judge shares the generator's blind spots" problem; consistent with the eval-heavy mindset from the VLM project |
| Agent framework | **Hand-rolled ReAct-style loop** — no LangChain/LlamaIndex | Full control and full ability to explain every routing decision under interview pressure; the framework choice itself is a stated, defensible tradeoff |
| Guardrails | Input/output filtering on retrieved chunks before they reach the agent's context — pattern-based + a lightweight classifier pass | Cheap, fast, and directly testable with a small set of hand-crafted injection probes |
| Deployment | FastAPI + Docker (baseline), structured JSON logging + a `/metrics` endpoint (observability), Angular frontend (if time allows), AWS (bonus only) | Docker Compose is the one deployment artifact every reviewer can actually run; everything past that is additive polish |

**Total cost: $0** (Groq free tier + local open-source models; AWS free tier only if that bonus item happens).

**Risk flagged and optimized for:** Groq's free tier has rate limits that can throttle iterative agent development if you're calling it on every debug loop. Mitigation: during Phase B's iterative agent-routing debugging, cache LLM responses for repeated test questions locally instead of re-calling Groq each time; only hit the live API for genuinely new prompts or eval runs.

**New risk flagged Day 5, applies to Phase B/C:** a `qdrant-client` version drift already broke the retrieval code once (1.19.1 removed `.search()` in favor of `.query_points()` — see Day 5 summary). `requirements.txt` now pins `qdrant-client>=1.14` specifically to document the floor this code needs; if Phase C or D ever bumps dependencies, re-check `qdrant-client`'s changelog before assuming the API is stable.

---

## 6. Core Eval Concepts (fundamentals — know these cold)

- **Recall@k**: did the correct chunk appear anywhere in the top-k retrieved results? Binary per query, then averaged.
- **MRR (Mean Reciprocal Rank)**: `1/rank` of the first correct result, averaged across queries. Captures *how high* the right answer ranked, not just whether it appeared — Recall@k and MRR fail in different ways, so report both. **Day 5 produced a concrete empirical demonstration of this, not just the theory** — see the ablation table in the Day 5 summary: RRF fusion doubled Recall@10 while leaving MRR flat, and the reranker raised MRR/R@1 while *lowering* Recall@10. Either metric alone would have told an incomplete, and in one case backwards, story.
- **RRF (Reciprocal Rank Fusion)**: combines two ranked lists (BM25 + dense) without needing calibrated/comparable scores between them:
  `score(doc) = Σ 1 / (k + rank_i(doc))` summed across each retriever's ranking, typically `k = 60`. **Hand-implemented and unit-tested Day 5** (`src/retrieval/hybrid_retriever.py::reciprocal_rank_fusion`) — tests pin the exact formula against hand-computed scores, not just "it runs."
- **Faithfulness/hallucination rate**: does the generated answer's claims actually follow from the retrieved context? Measured here via an NLI model (entailment/contradiction/neutral classification of answer sentences against retrieved chunks), not an LLM-judge.
- **Bootstrapped confidence intervals**: resample the eval set with replacement many times, recompute the metric each time, report the interval — same statistical machinery used in the VLM hallucination project's bootstrap testing, applied here to retrieval and faithfulness metrics.
- **LoRA/QLoRA fine-tuning (new for this project):** freeze the base reranker's weights and train small low-rank adapter matrices instead of the full model — far cheaper to train, and the before/after comparison against the frozen baseline is itself an ablation, not just an upgrade. **Day 5 produced the actual go/no-go evidence for this**: a measured 60% recall ceiling (gold chunk reachable in the reranker's candidate pool) against a measured 25%/35% actual (reranked/hybrid-alone), meaning there's real headroom a domain-tuned reranker could close — see the Day 5 summary's ceiling diagnostic.
- **Recall ceiling diagnostic (new concept, added Day 5):** before assuming a reranker (or any downstream component) will fix a low score, check whether the correct answer is even *in the pool that component sees*. If it isn't, no amount of reranking/fine-tuning that component can help — the problem is upstream. Day 5 built `scripts/diagnose_day5_ceiling.py` specifically to answer this before committing Phase C's fine-tune to a target that might be the wrong component.

---

## 7. Day-by-Day Plan (Oct 2 – Oct 14, four phases, full scope)

Each day is the same 2–4 hour block already set aside in the interview-prep plan. Full scope means these days are fuller than a minimal version would be — that's the tradeoff for shipping everything, and it's why the schedule below has almost no slack outside the two light days.

### Phase A — Corpus, retrieval, baseline eval (Oct 2–6)
*New conversation for this phase.*

**Thu, Oct 2 — Ingestion pipeline**
- Pull filings via SEC EDGAR API for all 8 companies; separate unstructured text sections from XBRL structured data
- Chunk the text (fixed-size + overlap — don't overengineer this early)
- **Model + effort:** Sonnet 5, standard. Routine data-pulling and parsing at higher volume than a 5-company version, but still no reasoning depth needed.
- **STATUS: DONE (see full session log below).** Built ahead of schedule, starting Sep 16 (setup + Day 1 work both landed in one long session before the official Oct 2 start).

**Fri, Oct 3 — Eval harness skeleton, before any retrieval exists**
- Hand-label a 40–60 question eval set across three types: single-hop retrieval, multi-hop numeric, out-of-scope (should refuse) — larger than a minimal version, since this set now also has to support the LoRA fine-tune's hard-negative mining in Phase C
- Write Recall@k/MRR scoring code against a trivial keyword-search baseline
- Generate embeddings (bge-base-en-v1.5), stand up the Qdrant index
- **Model + effort:** Sonnet 5, extended thinking, for the eval question taxonomy — this set now does double duty (evaluation *and* fine-tuning data), so it's worth the extra reasoning pass. Scoring code stays on Sonnet standard.
- **Why eval-first matters:** building the harness before the system it evaluates is unusual and worth stating explicitly in interviews — it forces the eval set to be independent of whatever retrieval quirks show up later.
- **STATUS: DONE (see Phase A, Day 3 session summary below).** 50-question eval set built (20 single-hop / 20 multi-hop-numeric / 10 out-of-scope), all 20 single-hop questions have real verified gold labels (**two later corrected Day 5 — see below**), Recall@k/MRR/bootstrap-CI scoring code built and tested, all 8 companies embedded (11,245 chunks), local Qdrant instance running. This session also found and fixed two more real ingestion bugs (XBRL duplicate-period dedup, Oracle Item 1A extraction) that only surfaced once the Day 1 pipeline ran end-to-end against every company's real filings.

**Sat, Oct 4 — Light day: RAG architecture read, no heavy coding**
- Read two solid write-ups on production RAG patterns; plan the RRF fusion implementation and sketch the guardrails threat model on paper
- **Model + effort:** none required — reading day. Sonnet 5 standard if you want a sounding board.
- **STATUS: NOT STARTED.** Still open — not blocking, can be slotted in whenever convenient. Worth doing before Phase B's guardrails threat model specifically.

**Sun, Oct 5 — Hybrid retrieval build**
- BM25 baseline → RRF fusion with dense embeddings (hand-written, `k = 60`) → baseline reranker on top
- Run the eval harness after *each* addition, keep every number — this produces the ablation table that later becomes the LoRA before/after comparison's baseline
- **Model + effort:** Sonnet 5 standard for implementation. **Opus session #1** (extended thinking) once the numbers are in: *"here are my Recall@k/MRR numbers at each stage — does this ablation pattern make sense, and which failure cases look like good hard-negative candidates for fine-tuning later?"* — scoping it to also flag fine-tuning candidates now saves a repeat pass in Phase C.
- **STATUS: DONE (see Phase A, Day 5 session summary below).** All 4 stages built and measured against the real 11,245-chunk corpus. Final numbers: keyword baseline R@10=0.10/MRR=0.015 → BM25 R@10=0.15/MRR=0.082 → RRF hybrid R@10=**0.35**/MRR=0.080 → +reranker R@10=0.25/MRR=**0.133**. Two real infrastructure bugs found and fixed (a `qdrant-client` API removal, and a point-id collision that silently corrupted the Day 3 Qdrant upload down to ~19% of the real data). A recall-ceiling diagnostic determined the reranker has genuine headroom (60% ceiling vs. 25% actual) — the measured go signal for Phase C's LoRA fine-tune — while also finding a real, separate ~40% retrieval/chunking ceiling no reranker can fix. Two eval gold-label issues found and corrected. Full narrative, all numbers, and new interview Q&A in the Day 5 summary below.

**Mon, Oct 6 — Buffer / hard-negative mining prep**
- Absorb anything that ran long in Phase A; if on schedule, start mining hard negatives from the ablation's failure cases for the Phase C fine-tune
- **Model + effort:** Sonnet 5 standard.
- **STATUS: NOT STARTED. This is the next task.** A real head start exists: `data/eval_set/day5_ceiling_diagnostic.json` already carries a `hard_negatives` field per question (chunks the fused retriever ranked above gold), and sh_011 specifically is a named, understood hard-negative *pattern* (retriever finds the right neighborhood/context chunk but not the specific sentence-bearing chunk) worth deliberately oversampling for, not just mining incidentally.

### Phase B — Generation, faithfulness, agent, guardrails (Oct 7–9)
*New conversation for this phase — Phase A's summary carries over via the knowledge base.*

**Tue, Oct 7 — Generation + faithfulness eval**
- RAG answer generation over retrieved chunks (Groq/Llama 3.3 70B)
- NLI-based faithfulness scoring (`nli-deberta-v3-base`) with bootstrapped confidence intervals
- **Model + effort:** Sonnet 5 standard for the generation pipeline; Sonnet 5 extended thinking for the bootstrap CI code, since a subtly wrong resampling implementation gives confident-looking but meaningless intervals.

**Wed, Oct 8 — Agent/tool-use layer**
- Hand-rolled ReAct-style routing across: retrieve / SQL query / calculator / answer-directly
- Build the SQLite table from XBRL data; wire up the calculator and SQL tools
- Set up local response caching for repeated Groq calls (see risk note in Section 5)
- **Model + effort: Opus session #2** (extended thinking) for the routing-logic design itself — the decision boundary for retrieve vs. SQL vs. calculator vs. answer-directly is the hardest reasoning task in the project and the piece most likely to get grilled in an interview. Sonnet 5 standard for the mechanical implementation once the logic is decided. **Budget note: only 2 Opus sessions remain as of Day 5 for 3 still-flagged moments (this one, Phase C's LoRA go/no-go, Phase D's failure analysis) — decide explicitly at the start of this session how the shortfall gets absorbed, don't let it happen by default.**

**Thu, Oct 9 — Agent testing + guardrails**
- Re-run the multi-hop numeric eval questions end to end against the full agent
- Build prompt-injection guardrails: pattern-based filtering plus a lightweight classifier pass on retrieved chunks before they reach the agent's context; test against a set of hand-crafted injection probes
- **Model + effort:** Sonnet 5 extended thinking for the guardrails threat model (what injection patterns actually matter for this corpus, and what a reasonable classifier boundary looks like) — a step above routine implementation, but not Opus-level; Sonnet standard for the probe-testing code.

### Phase C — Fine-tuning, full integration (Oct 10–12)
*New conversation for this phase.*

**Fri, Oct 10 — LoRA/QLoRA fine-tune, day 1**
- Prepare the training set from Phase A/B hard negatives and hand-labeled positives; set up the PEFT/LoRA config for the reranker
- Kick off training (Colab T4 or equivalent) — this runs unattended for part of the day, so pair it with lighter tasks
- **Model + effort: Opus session #3** (extended thinking) for the go/no-go design call: *"given this hard-negative set and reranker architecture, is this LoRA config likely to actually move the needle, or are there failure modes I should adjust for before spending the training run?"* Sonnet standard for the training script itself. **Day 5 already did much of the empirical groundwork this question needs** (the 60%-ceiling-vs-25%-actual finding, and the specific "finds context not sentence" failure pattern from sh_011) — this session should open by reading that, which may mean this go/no-go is a lighter lift than the plan originally assumed, and a candidate for downgrading to Sonnet extended thinking given the Opus shortfall.

**Sat, Oct 11 — Light day: fine-tune monitoring + Docker basics**
- Monitor/complete the fine-tune run; re-run the Phase A ablation harness with the fine-tuned reranker swapped in for a clean before/after comparison
- Start the Dockerfile for the core pipeline (lighter task, fits the light-day budget)
- **Model + effort:** Sonnet 5 standard for both — this is evaluation and routine containerization, not a design decision. `scripts/run_day5_ablation.py` and `scripts/diagnose_day5_ceiling.py` are both already built and reusable for this re-run — swap the reranker instance in and re-run, no new scoring code needed.

**Sun, Oct 12 — FastAPI backend + full system integration**
- Wrap the complete pipeline (fine-tuned reranker, agent, guardrails) in a FastAPI backend with clean endpoints
- Finish Dockerizing; confirm `docker-compose up` brings up Qdrant + the API together
- Add structured logging and a basic `/metrics` endpoint for observability
- **Model + effort:** Sonnet 5 standard, or Haiku 4.5 standard for the boilerplate route/schema scaffolding and logging setup to save Sonnet quota for the integration glue.

### Phase D — Full evaluation, polish, docs (Oct 13–14)
*New conversation for this phase.*

**Mon, Oct 13 — Full evaluation harness run + error analysis + polish**
- Run the complete harness end to end across the full system (fine-tuned reranker, agent, guardrails) — retrieval accuracy, groundedness score, latency
- Real failure analysis: which question types still break, and why
- If time remains: start the optional Angular frontend (Streamlit already covers the working-demo requirement, so this is pure polish — see fallback order in Section 3)
- **Model + effort: Opus session #4** (extended thinking), reserved specifically for the failure analysis — feed it the full set of wrong/borderline answers and ask for a genuine pattern read: *"here are the questions we still get wrong — what do they have in common, and is it a retrieval failure, a routing failure, a guardrails false-positive, or a generation failure in each case?"* This is what turns a metrics table into an interview-ready narrative. Everything else on this day stays on Sonnet standard. **Given the Opus shortfall flagged Day 5, this is the fourth session against a 2-remaining budget — one of the three Phase B/C/D moments needs to have already been downgraded by the time this day arrives.**

**Tue, Oct 14 — Wrap-up & documentation**
- Write the README: problem statement, architecture diagram, full results table (ablation, LoRA before/after, guardrails probe results), what you'd improve next
- Finish the Angular frontend if it's in progress, or note it as a documented next step if not — do not let it block the README/GitHub push
- Push to GitHub, pin it, record a short demo GIF
- Fill in the resume capstone bullet (template below) with real numbers
- **Model + effort:** Haiku 4.5 standard for README scaffolding and docstrings; Sonnet 5 standard for the architecture diagram description and final proofread. No Opus needed today — every hard decision has already been made.

**Resume bullet template — fill in with your real Oct 14 numbers:**

> Built GroundedRAG, an agentic RAG system over SEC 10-K/10-Q filings with hybrid retrieval (BM25 + dense + RRF + a LoRA-fine-tuned reranker) and tool-routing (retrieval / SQL / calculator); achieved [X]% Recall@5 and [X] MRR on a 40–60 question held-out eval set (fine-tuned reranker improved Recall@5 by [X] points over baseline), with an NLI-based faithfulness rate of [X]% (bootstrapped 95% CI: [X]–[X]%); added prompt-injection guardrails and deployed via FastAPI + Docker with basic observability.

---

## 8. Interview Q&A Log (running — add to this each session)

- *Why hybrid retrieval instead of just dense embeddings?* → Dense embeddings miss exact-match signals (ticker symbols, specific dollar figures, legal terms) that BM25 catches; hybrid covers both failure modes.
- *Why NLI for faithfulness instead of an LLM-judge?* → Determinism, cost, and avoiding the judge sharing the generator's blind spots — same reasoning as attention-probing over black-box judging in the VLM project.
- *Why does the agent need a calculator/SQL tool instead of letting the LLM compute it?* → LLMs are unreliable at multi-step arithmetic over retrieved numbers; grounding numeric answers in a deterministic tool eliminates a specific, measurable hallucination mode.
- *Why build the agent loop by hand instead of using LangChain?* → Full control over and full ability to explain every routing decision — a stated tradeoff, not an oversight.
- *Why SEC filings instead of a more familiar domain like arXiv abstracts?* → The corpus's split between unstructured text and structured XBRL data makes the SQL/calculator tools load-bearing rather than bolted-on — a model answering "YoY revenue growth" from retrieved text alone will hallucinate the number, which is exactly the failure mode the project is built to catch.
- *Why fine-tune the reranker instead of just using a bigger pretrained one?* → The LoRA fine-tune trains directly on this corpus's hard negatives — cases the baseline reranker specifically got wrong — so it targets this system's actual failure modes rather than generic retrieval quality.
- *Why guardrails on retrieved text specifically, rather than just on the final output?* → Injected instructions live in the untrusted retrieved context, not the user's query — filtering only the output misses the point where the agent's behavior could actually be hijacked.
- *Why was AWS deployment the one thing left optional?* → A working `docker-compose up` demonstrates the same engineering competency with far less infrastructure risk in the final two days; AWS is real bonus value if time allows, not a requirement the story depends on.

**Added Phase A, Day 1 (ingestion — see full session log below for the complete debugging narrative these are drawn from):**

- *Why does your section splitter require a standalone line, not just any "Item N" match anywhere in the text, to count as a header?* → Real MD&A/Risk Factors sections contain inline cross-references to other item numbers ("see risks described under Item 1A of this report") that a naive matcher treats as false section boundaries — found via real EDGAR data: Apple's and Alphabet's 10-Q MD&A was truncating to ~200-400 words every single quarter before this fix, because both companies open MD&A with a forward-looking-statements paragraph that references Item 1A early on.
- *Why does a section's end boundary skip over repeated occurrences of the same item number, instead of stopping at the next occurrence of anything?* → Some filers (Microsoft) repeat the current section's header at the top of every rendered "page" of a long section. Treating a same-item repeat as an end boundary — rather than a continuation — silently keeps exactly one page's worth of content and discards the rest, which produced a very specific, diagnosable symptom: uniform ~500-1000 word sections across every item type, every quarter, for one company only.
- *Why require the header to carry a real title, rather than trusting gap size alone to separate a TOC entry from a real section?* → Gap size is a proxy that can fail by chance. A "PART I. FINANCIAL INFORMATION" divider sitting between two bare TOC entries can push their gap past the threshold, making a bare TOC line look like real content. A genuine section header always carries its descriptive title inline or on an adjacent line from the same HTML tag; a bare index/TOC/pagination-repeat entry never does — a more structurally reliable signal than distance alone.
- *Your title-requirement fix briefly broke Amazon and Meta completely (zero chunks). What happened, and what does that teach you about text-flattening pipelines?* → `BeautifulSoup.get_text(separator="\n")` inserts a line break at every HTML tag boundary, not just block-level ones. Some filers render the item number and its title in two adjacent-but-separate tags ("`<b>Item 1A.</b>` `<span>Risk Factors</span>`"), which lands them on two different lines once flattened — and some filers do this for literally 100% of their headers. The lesson: a text-flattening step that looks clean isn't the same as one that's structurally reliable; different filing agents produce meaningfully different HTML shapes for what renders identically in a browser, and a corpus of 8 companies is enough to hit several of them.
- *Ten-Qs have "Item 1" mean two different things — how did you handle that?* → Part I's Item 1 is Financial Statements; Part II's Item 1 is Legal Proceedings. Both are genuine, correctly-titled sections that happen to share an item number. Once TOC and pagination noise are filtered out of the match set, "first chronological occurrence, bounded by the next *different* item number" resolves this correctly for free — Part I's real content always precedes Part II's in document order.
- *You eventually stopped fixing a known bug (10-Q Item 1 truncation for 4 of 8 companies) instead of continuing to debug it. Walk through that decision.* → After 6 iterations against real data, each fix for one company's HTML quirk was resurrecting a previously-fixed bug for a different company — a genuine signal the current approach (line-based heuristics on flattened text) had stopped converging on this specific edge case, not just that more iteration was needed. Rather than keep spending time chasing a moving target, I evaluated the actual cost of leaving it: Item 1's content is financial-statement tables, which the XBRL/SQL tool-routing path already covers more reliably than free-text retrieval would; the eval question taxonomy targets Item 1A/Item 7/Item 2 and structured financial data, not Item 1 prose; and the root cause plus the structural fix that *would* resolve it are documented in code for later if it becomes an actual problem. This is a real engineering tradeoff under a time-boxed deadline — knowing when to stop is as much a skill as knowing how to debug.
  - *Correction added Day 3:* this limitation was later found to affect **5** of 8 companies, not 4 — META has the identical symptom and was missed from the original list. Documentation-only correction; the underlying decision and reasoning above are unchanged.

**Added Phase A, Day 3 (eval harness + XBRL/section-splitter follow-on bugs — see full session log below):**

- *Tell me about a bug you found during evaluation, not during initial development.* → Oracle's Item 1A (Risk Factors) was never extracted, for any of its four 10-Ks — completely invisible until the eval-labeling process specifically needed it. Found via a symptom in an unrelated diagnostic (`inspect_chunks`: zero `item_1a` rows for ORCL, every other company had them), narrowed via a purpose-built diagnostic (`inspect_matches`: the header match list jumped straight from Item 1 to Item 1B, skipping Item 1A, with a 158,622-character gap — 5-10x every other company's Item 1 span), root-caused via a third purpose-built diagnostic (`diagnose_orcl_item1a.py`, which used `repr()` on the raw text around "risk factors" to reveal the exact hidden split). The root cause: Oracle's HTML splits the header **mid-word** across a tag boundary (`"Item 1A.\tR"` then `"isk Factors"` on the next line) — a more extreme version of a tag-split bug already fixed on Day 1, which only triggered its lookahead when the current line's remainder was completely empty; here the remainder was a non-empty single character (`"R"`), so the old fix's trigger condition never fired. This is a concrete example of a class of bug that only surfaces once you actually try to *use* the data for something (labeling), not during initial ingestion smoke-testing.
- *How do you deduplicate financial data from a source that doesn't guarantee uniqueness?* → SEC's XBRL company-facts JSON reports the same real fact multiple times across filings — once in the filing that originally reported it, then again as a prior-year/prior-quarter comparative in every later filing, each time tagged with the REPORTING filing's `fy`/`fp`, not the value's own true fiscal period. Deduplicating on `fy` (the SEC-provided label) looks reasonable but is wrong; the fix is to key on the fact's actual `(start, end)` date pair, which is the real period identifier. There's a second layer to this: duration-type facts like revenue aren't uniquely identified by `end` alone either — a 10-Q reports both a standalone-quarter figure and a year-to-date-cumulative figure under the same tag and same end date, distinguished only by `start`. Both numbers are correct and both need to survive deduplication; only genuinely conflicting values for the same `(start, end)` pair should trigger a warning.
- *You said a fix's own regression test caught a new bug in the fix itself. What happened?* → After keying XBRL deduplication on `(ticker, metric, form, start, end)`, the conflict-detection code used `pandas.groupby()` on that same key to check for real value conflicts. `pandas.groupby()` silently excludes any row where a group-by column is `None`/`NaN` — and `start` is legitimately `None` for many real 10-K annual facts (they don't have a "year-to-date" ambiguity to resolve, so SEC doesn't always populate a start date the same way). The dedup itself (`drop_duplicates`) wasn't affected, but the conflict-detection warning silently stopped firing for any fact with a missing `start` — a real bug that was invisible until a regression test written specifically for the *previous* fix failed for an unrelated reason. The general lesson: writing a test for fix N can surface a bug in fix N itself, which is a different and complementary thing to writing a test that just confirms fix N did what was intended.
- *Why doesn't Recall@k/MRR apply to every question in your eval set?* → The 20 multi-hop-numeric questions in the 50-question set have no single "gold chunk" by design — the correct system behavior for them is SQL tool-routing (built in Phase B), not text retrieval. Scoring them with Recall@k/MRR would either be meaningless or would silently reward the wrong behavior (a confident-looking but hallucinated prose answer). Instead, a separate sanity check (`score_sql_routing_sanity()`) checks whether the keyword baseline falsely returns a confident prose chunk for a question that actually needs a computed number from structured data — catching a real pre-agent failure mode early rather than papering over it with a blended metric.

**Added Phase A, Day 5 (hybrid retrieval build — three infrastructure bugs, two eval label corrections, one honest process mistake, and the recall-ceiling methodology — see full session log below):**

- *Your uploads all reported success, yet dense retrieval barely did anything. What happened, and what does it teach you about trusting a pipeline's own success signal?* → All 8 `qdrant_setup.py` upload commands from Day 3 genuinely completed without error — and the collection still ended up holding roughly 19% of the real data (~2,192 of 11,245 points). The root cause: each per-company run assigned Qdrant point ids via `enumerate()`, restarting at 0 every time, and Qdrant's `upsert` treats a colliding id as an update, not an error — so MSFT's points silently overwrote AAPL's, GOOGL's overwrote MSFT's, and so on, leaving only the last company (ORCL) and the tail of the second-to-last (AVGO) actually present. Every individual upload command was telling the truth about itself; none of them could see that they were destroying each other. The fix was to derive each point id deterministically from the chunk_id Day 3 had already built (`uuid5`) instead of a positional counter, and to add a post-upload point-count check so the *aggregate* state gets verified, not just each command's own exit status. The general lesson: a pipeline that reports success per-step can still be silently wrong in aggregate if steps can overwrite each other's output — the check that would have caught this was one line ("how many points does the collection actually hold now?") that nobody had asked.
- *How did you know something was wrong before you'd even found the bug?* → The ablation table showed BM25→RRF-hybrid changing Recall@1/3/5 by exactly zero. A working dense retriever fused with BM25 on paraphrased natural-language questions should not be that inert — that contradicted a strong prior about what dense embeddings are for, which made "the measurement is broken" more likely than "dense doesn't help here." The check took two minutes: diff the per-question retrieved lists between the BM25-only and hybrid stages. They were byte-identical for 16 of 20 questions — and a real fusion of two genuine ranked lists essentially never reproduces one of them exactly, so identical output meant one input list was empty. That one check turned "an unexplained flat result" into "dense retrieval is returning nothing for 16 companies," which was traceable in minutes from there.
- *Tell me about a time you made things worse before making them better, and how you handled it.* → When told to run the ablation script, I advised switching to Opus *before* the very first run — reasoning that a fresh table deserved the reserved interpretation tier. The first run failed immediately on a mechanical issue (a removed `qdrant-client` API method), so that Opus session was spent on a dependency-version fix, not the reasoning work it was meant for. The right sequence was: get a successful run on the cheap tier first, only escalate once there's something worth interpreting. I said so plainly rather than quietly not counting the spend, and it left the project with only 2 Opus sessions for 3 still-flagged plan moments — a real, disclosed budget consequence of a sequencing mistake, not something to paper over.
- *Why did you build a "recall ceiling" diagnostic instead of just fine-tuning the reranker once the ablation showed it underperforming?* → The ablation alone couldn't distinguish two very different problems: "the reranker sees the right chunk but scores it wrong" (a reranker problem, LoRA fixes it) versus "the reranker never sees the right chunk at all" (a retrieval/chunking problem, no reranker fix touches it). Measuring "is gold in the top-50 candidate pool, even if it's not in the top-10 output" separates these directly. The result: a 60% ceiling against a 25% actual — real, fixable headroom — but the ceiling itself caps at 60%, not 100%, meaning roughly 40% of failures are retrieval-side and would not move even with a perfect reranker. Committing Phase C's fine-tune without this check risked spending it on the wrong half of the problem.
- *A diagnostic flagged three "near-miss" cases where an adjacent chunk scored well but wasn't credited as correct. What did you do with that, and why does it matter that the three cases turned out differently?* → I read all three adjacent-chunk pairs against their gold chunks directly rather than trusting the heuristic. One (sh_001) was a false positive — the adjacent chunk discussed an unrelated risk, purely coincidental structural proximity. One (sh_003) was genuine — two overlapping-window chunks of one continuous currency-risk paragraph, independently confirmed by dense retrieval ranking the "missed" chunk at position 2 (not just adjacency, real semantic relevance) — so the gold label was correctly widened to include it. The third (sh_011) turned out not to be a near-miss question at all: reading past what the automated ±1 check even looked at, I found the *original* gold label was simply wrong — it pointed at a chunk about an FTC settlement that never mentions AWS, while the actual driver sentence for the AWS-specific question was one chunk further. That's the point worth stating in an interview: an automated diagnostic generates leads, not verdicts, and treating all three flagged cases identically would have either missed a real labeling bug (sh_011) or inflated results on a coincidence (sh_001).
- *Why report Recall@k AND MRR, beyond "the plan says so" — do you have real evidence they diverge?* → Yes, from Day 5's own table, not just the Day 3 definition. RRF fusion took Recall@10 from 0.15 to 0.35 (more than double) while MRR moved by only 0.005 — it pulled correct chunks into the top-10 without improving their ordering. The reranker then took Recall@10 *down* from 0.35 to 0.25 while taking MRR *up* from 0.080 to 0.133 and Recall@1 from 0.00 to 0.10 — it sharpened the head of the ranking at the cost of the tail. A report using only one metric would have called one of these two stages either a win or a loss in the wrong direction.

---

## 9. Session Summary Template (copy this section each session, fill in, append below)

```
### Phase [A/B/C/D] Summary — [date]

**Decisions made & why:**
-

**Metrics this session:**
-

**Fundamentals covered:**
-

**New interview Q&A:**
-

**Opus sessions used so far (target: ≤5 total):**
-

**Open questions / next steps:**
-
```

---

## Session Summaries Log

*(Append each session's filled-out template here, in order.)*

### Phase A Summary — 2026-09-16 (Setup + Day 1: Ingestion pipeline)

**Context:** This session ran ahead of the official Oct 2 build-window start (today is Sep 16 — noted and treated as legitimate early prep, not a schedule violation). Covered: repo/environment setup, then the full Day 1 ingestion pipeline, including six real bugs found and fixed (one accepted as a documented limitation) against live SEC EDGAR data.

#### Setup (pre-Day-1)

- Scaffolded the repo: `src/{ingestion,retrieval,agent,eval,guardrails,api}`, `data/{raw,processed,eval_set}`, `notebooks/`, `docs/`, `tests/`, with `.gitignore`, `.env.example`, `.vscode/` config, and a stub `README.md`
- Pushed to GitHub, set up a local venv, installed Phase-A-only dependencies (requirements.txt seeded per-phase rather than all upfront)
- Decision: track `data/eval_set/` in git (small, hand-labeled, worth a reviewer being able to open it directly); gitignore `data/raw/` and `data/processed/` (regeneratable, and 8 companies × 4 years of filings gets large)

#### Day 1: Ingestion pipeline — build

Built `src/ingestion/`: `config.py` (company list + CIKs, form types, target Item sections per form type), `edgar_client.py` (rate-limited SEC API client — submissions list, document fetch, XBRL company-facts fetch, with SEC-required User-Agent handling), `section_splitter.py` (HTML→text, then Item-section extraction), `chunker.py` (fixed-size word-based chunking with overlap, deliberately simple per the plan's "don't overengineer this early" note), `xbrl_extractor.py` (pulls revenue/net income from XBRL facts, with a fallback tag list since revenue is reported under different GAAP tags across companies/years), `pipeline.py` (orchestrator).

Company list finalized: AAPL, MSFT, GOOGL, AMZN, META, NVDA, AVGO, ORCL (see Section 4 above for CIKs, verified against live EDGAR).

#### Day 1: Bugs found and fixed against real data (the actual work of this session)

Six real, previously-invisible EDGAR HTML issues were found and fixed by running the pipeline against live data and inspecting actual output — not assumed away. In order:

**Bug 1 — Missing `lxml` dependency.** Pipeline crashed on all 8 companies immediately (`bs4.exceptions.FeatureNotFound`). `lxml` was listed in requirements.txt but not actually installed in the local environment. Fixed by installing it, and made `html_to_text` fall back to the stdlib `html.parser` if `lxml` isn't importable, so a missing optional dependency degrades gracefully instead of crashing every filing. Verified the fallback path independently by running the test suite with `lxml` fully uninstalled.

**Bug 2 — Inline cross-references treated as section boundaries.** Symptom: Apple's and Alphabet's 10-Q MD&A (Item 2) truncated to ~200–400 words every single quarter, with no variance. Root cause: the original matcher treated *any* occurrence of "Item N" anywhere in the flattened text as a candidate section boundary — including cross-references inside a section's own body ("...risks described under Item 1A of this report..."), which both companies' MD&A opens with as boilerplate. This clipped the real section's *end*, not just its start. Fix: only treat "Item N" as a header candidate when it is the entire content of its own line (short, standalone) — an inline cross-reference sits inside a much longer paragraph line and gets filtered out. Regression test: `test_split_does_not_truncate_at_inline_cross_reference`.

**Bug 3 — Pagination artifacts (Microsoft).** After fixing Bug 2, Microsoft's sections were still uniformly small (~500–1100 words) across every item type, every quarter, unchanged by the cross-reference fix. Root cause, confirmed via a custom diagnostic (`scripts/inspect_matches.py`, built specifically to dump real match data instead of continuing to guess): Microsoft's filings repeat the current section's header at the top of every rendered "page." The original boundary logic picked the single occurrence with the largest individual gap to its immediate next header — which for a paginated filing meant keeping exactly one page's content and silently discarding the rest. Fix: section start = first non-TOC occurrence; section end = the next occurrence of a **different** item number (repeated same-item headers along the way are treated as continuation, not a boundary). Regression test: `test_split_stitches_repeated_same_item_headers_across_pages`.

**Bug 4 — Bare TOC entry with an anomalous gap, plus Part I/Part II item-number collision.** The pagination fix (Bug 3) caused a new regression: Item 1 collapsed to ~90–115 words for GOOGL/NVDA/META/ORCL/MSFT. Diagnosed via the same match-dumping script: a "master TOC" entry that is literally just `"Item 1."` (no title) had a gap of 740 characters to the next TOC line — above the 400-character TOC threshold purely by chance, because a "PART I. FINANCIAL INFORMATION" divider happened to sit between them. This fooled the gap-only filter into treating the bare TOC entry as real content. Separately, 10-Qs have two *different* real sections that both parse as "Item 1" — Part I's Financial Statements and Part II's Legal Proceedings. Fix: require a real descriptive title after "Item N." (not just short line length) to count as a header candidate at all — TOC/index/pagination entries are consistently bare, real headers consistently aren't. This also resolved the Part I/Part II collision for free, since "first chronological occurrence, bounded by next *different* item" already picks correctly once noise is filtered. Regression tests: `test_split_ignores_bare_toc_entry_with_anomalous_gap`, `test_split_resolves_part1_part2_item1_collision`, `test_split_item2_and_item1a_unaffected_by_toc_and_collision`.

**Bug 5 — Title split across HTML tags.** The title-requirement fix (Bug 4) caused a worse regression: Amazon and Meta returned **zero chunks** across every filing. Root cause: `get_text(separator="\n")` inserts a line break at every tag boundary, not just block-level ones. Some filers render the item number and its title in two adjacent-but-separate tags (`<b>Item 1A.</b>` then `<span>Risk Factors</span>`), landing them on different lines once flattened — Amazon and Meta do this for 100% of their headers; Oracle/Microsoft/Alphabet do it inconsistently (explaining their partial gaps); Apple/Nvidia/Broadcom never do it (explaining why they were unaffected throughout). Fix: look ahead past blank lines (from tag-boundary whitespace) for a title on a later line, but only when the current line's remainder is empty, and only if that later line isn't itself another "Item N" match (which would mean a genuine TOC listing, not a split title) and starts with an uppercase letter (real titles always do; trailing unrelated prose usually doesn't). Regression tests: `test_split_finds_title_split_onto_next_line_by_tag_boundary`, `test_split_does_not_treat_two_bare_toc_entries_as_split_title`.

**Bug 6 — Accepted, documented limitation (not fixed).** The blank-line lookahead from Bug 5's fix resurrected Bug 4's exact symptom for MSFT/GOOGL/NVDA/ORCL's 10-Q Item 1 (~90–115 words again). Confirmed via `scripts/inspect_matches.py` that the lookahead was very likely borrowing a false "title" from the same "PART I. FINANCIAL INFORMATION" divider — which is itself title-cased, so the uppercase-start guard from Bug 5 doesn't catch it. **Decision: stop here rather than attempt a seventh fix.** After 6 iterations, each fix for one company's quirk was resurrecting a previously-fixed bug for another company — a real signal the line-based heuristic approach had stopped converging on this specific edge case. Reasoning for accepting it: (1) Item 1A (Risk Factors) and Item 7/Item 2 (MD&A) — the sections the eval question taxonomy actually targets — are correct and clean across all 8 companies; (2) Item 1's content is financial-statement tables, already covered better by the XBRL/SQL tool-routing path than free-text retrieval would cover it; (3) 10-K Item 1 (Business) is unaffected for all 8 companies, only 10-Q Item 1 (Financial Statements) for 4 of 8 companies is affected; (4) the root cause and the structural fix that would actually resolve it (require a borrowed title to independently pass its own gap check, not just exist) are documented directly in `section_splitter.py`'s module docstring for later if it ever becomes a real problem. This is documented as a deliberate scope decision, not a silently-shipped bug.

> **Correction, added Day 3 (see below):** this limitation was later found to affect Item 1 of 10-Qs for **5** of 8 companies, not 4 — META has the identical symptom and was missed from the original scan. Documentation-only correction to which companies are affected; the underlying decision and reasoning above are unchanged.

#### Final ingestion state (after all Day 1 fixes)

- All 8 companies successfully ingested: AAPL/MSFT/AMZN/NVDA/AVGO/ORCL = 16 filings each, GOOGL = 13, META = 9 (younger filing history)
- Item 1A (Risk Factors) and Item 7/Item 2 (MD&A) correct and clean for all 8 companies
- Item 1 (Financial Statements/Business): correct for 10-Ks (all 8 companies) and for 10-Qs of AAPL/AMZN/AVGO; known truncation (~90–115 words) for 10-Q Item 1 of MSFT/GOOGL/NVDA/ORCL, documented as an accepted limitation (see Bug 6 above; corrected to 5 companies on Day 3)
- XBRL structured facts (revenue, net income) extracted for all 8 companies
- 16 unit tests across `tests/test_chunker.py` and `tests/test_section_splitter.py`, all passing — every test added this session is a direct regression test for a real bug found against live data, not written speculatively
- Two diagnostic scripts built and kept in the repo for future debugging: `scripts/inspect_chunks.py` (per-filing, per-item word/chunk counts, flags anything under 500 words) and `scripts/inspect_matches.py` (dumps every header-line match with position/gap for one real filing — this is what actually found Bugs 3, 4, and 6; guessing from symptoms alone had gotten Bug 3's diagnosis wrong once before this tool existed)

**Metrics this session:**
- 8/8 companies ingested successfully
- Final chunk counts (pre-Day-3-fixes): AAPL 763, MSFT ~1500, GOOGL 768, AMZN 1749, META 1676, NVDA 1241, AVGO 2192, ORCL ~1230
- XBRL fact rows: 30–60 per company depending on filing history length
- 16/16 unit tests passing
- 6 real bugs found and diagnosed against live data; 5 fixed with regression tests, 1 accepted as a documented limitation

**Fundamentals covered:**
- SEC EDGAR API structure: `submissions` endpoint (filing list) vs. `companyfacts` endpoint (XBRL) vs. `Archives` (actual documents), and the User-Agent/rate-limit etiquette required to use them without being blocked
- XBRL tag variance: revenue is reported under different GAAP tags depending on company and filing era (`RevenueFromContractWithCustomerExcludingAssessedTax`, `Revenues`, `SalesRevenueNet`, etc.) — a fallback-tag-list is the pragmatic fix, not a single hardcoded tag name
- The core lesson underlying 3 of the 6 bugs: flattening HTML to text with `get_text(separator="\n")` inserts breaks at *every* tag boundary, not just block-level ones — meaning "the text looks clean" and "the text is structurally reliable for parsing" are different claims, and different filing agents (different companies, effectively) produce meaningfully different HTML shapes for content that renders identically in a browser
- Debugging methodology: get real diagnostic data before proposing a fix, every time — the session's second-worst outcome (the AMZN/META zero-chunks regression) happened specifically because a fix was applied based on a plausible-but-wrong theory instead of a match dump; the eventual custom diagnostic tool (`inspect_matches.py`) is what actually found the true root causes

**New interview Q&A:** see Section 8 above — six new entries added, each walking through one bug's symptom → root cause → fix → what it teaches about the underlying data (this is deliberately the most interview-dense part of the whole session; SEC filing HTML parsing quirks are exactly the kind of "here's a real bug I found and how I diagnosed it" story that's stronger than "I wrote a parser and it worked").

**Opus sessions used so far (target: ≤5 total):**
- 0/5. The entire ingestion debugging chain — six bugs, six diagnostic/fix cycles — stayed on Sonnet 5 standard throughout. None of it required Opus-level reasoning; it required careful iteration against real data, which is a different skill and a cheaper one to exercise.

**Open questions / next steps:**
- Day 1 (ingestion) is functionally done. Data exists and is usable for all 8 companies.
- Next task per the plan: **Phase A, Day 3 — eval harness skeleton** (hand-label 40–60 questions across single-hop/multi-hop-numeric/out-of-scope types; Recall@k/MRR scoring code; generate embeddings; stand up Qdrant). Flagged in the plan as **Sonnet 5 extended thinking** for the question taxonomy specifically, Sonnet standard for the scoring code.
- Day 4 in the calendar is a light/reading day (no coding) — not yet done, can be slotted in whenever convenient before Day 5's hybrid retrieval build.
- Recommended before Day 3: a quick manual spot-check of 2–3 real chunks per company, just to eyeball chunk boundaries/quality directly rather than trusting word-count metrics alone.
- Per the plan's session-structure guidance, start a **new conversation** for the next phase of work (this thread has now covered setup through all of Day 1 and is a reasonable stopping point) — Phase A's summary above will carry over via the knowledge base.

---

### Phase A Summary — 2026-09-17 (Day 3: Eval harness skeleton, + Day 1 follow-on bug fixes)

**Context:** This session picked up directly from the Day 1 summary above. Scope was Day 3 per the plan (eval harness skeleton: hand-labeled question set, Recall@k/MRR scoring vs. a keyword baseline, embeddings, Qdrant index) — but running the eval-labeling process against the real Day 1 pipeline output surfaced two more real, previously-invisible ingestion bugs (an XBRL deduplication issue found against live AVGO/all-company data, and a complete extraction miss for Oracle's Item 1A section) that had to be fixed before the eval set could be honestly labeled. This session therefore covers more ground than Day 3's stated scope alone: it closes out the last open threads from Day 1's ingestion work as well. **This session's full first-round eval-harness scripts were originally drafted without direct repo access**, using placeholder `company`/`section` field names; once repo access was granted, all Day 3 code was corrected to the real pipeline's `ticker`/`item` field names before anything else proceeded — noted here because it was a genuine early misstep, not because it affected the final delivered code.

#### Day 3: Eval harness — build

Built `src/eval/`: `keyword_baseline.py` (trivial top-k keyword-overlap retrieval, used as the floor every later retriever must beat), `scoring.py` (Recall@k, MRR, and bootstrapped 95% confidence intervals via resampling), `generate_embeddings.py` / `scripts/generate_all_embeddings.py` (bge-base-en-v1.5 embedding generation, looping over all 8 companies with the model loaded once), `qdrant_setup.py` (local Qdrant collection creation + per-company upload). Built `data/eval_set/eval_questions.json`: 50 questions — 20 single-hop retrieval, 20 multi-hop-numeric, 10 out-of-scope/should-refuse — each with `ticker`, `target_item_key`, the question text, `gold_answer_type`, and either `gold_chunk_ids` (single-hop) or `gold_sql_description` (multi-hop-numeric, a spec for the Phase B SQL tool to consume directly, not a hardcoded number).

Built supporting labeling infrastructure once it became clear hand-labeling needed real candidate data, not blind guessing: `scripts/label_eval_set.py` (interactive labeling tool, shows each question's top-8 keyword-search candidates for a human or Claude to judge), `scripts/search_chunks.py` (wider top-20 full-text fallback search for questions the top-8 view missed), `scripts/apply_reviewed_labels.py` (applies Claude's own reviewed judgments — read directly from real filing text the user pasted back — into `eval_questions.json` in bulk, rather than requiring the interactive tool to be re-run by hand for every question).

#### Day 3: Bugs found and fixed (5 in total this session, continuing the Day 1 numbering as 1–3 since two were direct follow-ons to Day 1 code)

**Bug 1 — `pipeline.py`: no unique `chunk_id`, dropped `start_word`/`end_word`.** The JSONL writer built each chunk record as `{"text": ..., "chunk_index": ..., **metadata}` — `chunk_index` resets to 0 for every section (it's local to each chunking call), so it collides across filings and items, and the writer never included the chunk's own `start_word`/`end_word` at all, even though the `Chunk` dataclass already carried them. Both were silent gaps that would have blocked honest eval scoring entirely — there was no way to cite a specific, unique chunk as a gold label, and no way to compute real section word-counts without the offsets. Fixed with a new, independently testable function:
```python
def make_chunk_id(ticker: str, accession_number: str, item_key: str, chunk_index: int) -> str:
    accession_nodash = accession_number.replace("-", "")
    return f"{ticker}_{accession_nodash}_{item_key}_{chunk_index:04d}"
```
`start_word`/`end_word` are now written into every chunk record. 4 new regression tests in `tests/test_pipeline.py` (dash-stripping, uniqueness across filings/items, zero-padding).

**Bug 2 — `xbrl_extractor.py`: silently dropped financial periods across a GAAP tag change.** `_first_available_tag()` picked only the *first* candidate GAAP tag with any data and used exclusively that tag for a company's entire history. If a company's revenue tag genuinely changed partway through the 4-year lookback window (a real, documented EDGAR behavior — e.g. around ASC 606 transitions), every period reported under the other tag was silently dropped rather than merged in. Fixed by replacing it with `_all_available_tag_facts()`, which merges facts across every candidate tag and lets downstream deduplication resolve genuine conflicts by preferring the earlier-listed (generally more standard) tag. Also fixed a related fragility: an empty extraction result used to return a DataFrame with no columns at all, which breaks immediately on any `df["metric"]`-style downstream access; it now always returns the full expected schema (`ticker, metric, tag_used, fy, fp, form, start, end, val`) even when empty.

**Bug 2b — same real fiscal period counted multiple times, found against live AVGO data.** After running the pipeline for real, `AVGO`'s extracted facts had 33 rows where roughly 16–18 were expected. Root cause, found by inspecting the raw extracted rows directly (same diagnose-before-fixing discipline as every Day 1 bug): SEC's XBRL company-facts JSON reports each real fact multiple times — once in the filing that originally reported it, then again as a prior-period comparative in every later filing that references it — and each of those repeated appearances is tagged with the *reporting* filing's `fy`, not the value's own true fiscal year. A single real period (fiscal quarter ending 2022-10-30) showed up tagged `fy=2022` from the original 10-K, then again as `fy=2023`, then twice more as `fy=2024` from later filings citing it as a comparative. The original dedup key included `fy`, so all four survived as if they were four different periods — a direct threat to any eval question relying on "the two most recent fiscal years" being counted correctly. Fixed: dedup key became `(ticker, metric, form, end)` — `end` is the real, reliable period identifier; `fy` describes which filing said it, not which period it is. Among duplicates, the earliest-`filed` occurrence is kept (the original filing, not a later restatement copy); genuinely different values for what should be the same period are now logged as a warning rather than silently resolved, since that would indicate an actual restatement worth a manual look.

**Bug 2c — fix 2b's dedup key was still too loose, found against the full 8-company pipeline run.** Running the pipeline for real (not just against AVGO) flooded the logs with "2–3 DIFFERENT reported values" conflict warnings on nearly every 10-Q period, for every company, for both revenue and net income — far too uniform across the whole corpus to be genuine restatements. Root cause: duration-type XBRL facts (revenue, net income) are defined by a `(start, end)` date *pair*, not by `end` alone. A 10-Q routinely reports the same metric under the same tag with the same end date but two different start dates — once for the standalone quarter ("three months ended") and once for the fiscal-year-to-date cumulative ("six/nine months ended"). Both numbers are correct and both are real, distinct facts; fix 2b's `end`-only key was collapsing them into a false conflict. Fixed: dedup key became `(ticker, metric, form, start, end)`, so quarterly and YTD-cumulative facts correctly survive as separate rows.

**Bug 2d — a bug in fix 2c's own conflict-detection logic, caught by fix 2c's own regression test.** Adding `start` to the `groupby()` call used for conflict detection meant any fact with `start=None` — common for real 10-K annual facts, which don't always carry an explicit start date the same way quarterly facts do — silently dropped out of that grouping entirely, because `pandas.groupby()` excludes `NaN`/`None`-keyed rows by default. The actual deduplication (`drop_duplicates`) was unaffected; only the diagnostic conflict-warning logic silently stopped checking any fact with a missing `start`. This bug was not found by inspection — it was found because `test_extract_key_metrics_logs_warning_on_genuine_value_conflict`, a regression test written for fix 2c, failed for a reason that had nothing to do with 2c's actual logic. Fixed with `df.groupby(dup_key, dropna=False)["val"].nunique()`. This is a concrete, narratable example of a fix's own fix having its own bug — worth remembering that writing a regression test for a fix can surface problems in the fix itself, not just confirm the original bug is gone. 11 new tests total across fixes 2/2b/2c/2d in `tests/test_xbrl_extractor.py`, including one that pins the `dropna=False` behavior specifically so it can't silently regress.

**Bug 3 — `section_splitter.py`: Oracle's real Item 1A header was never detected, for any of its four 10-Ks.** Symptom first surfaced via `python -m scripts.inspect_chunks`: every other company had `item_1a` rows for every 10-K; Oracle had zero, across all four. Confirmed independently via `scripts/label_eval_set.py`'s candidate search for the one Oracle-related single-hop question (`sh_019`): both top candidates were 10-Q boilerplate deferring to "the factors discussed in Part I, Item 1A ... of our Annual Report on Form 10-K" — i.e. the system could find text *referencing* Oracle's Risk Factors section but had never actually extracted the section itself. Root cause narrowed via `python -m scripts.inspect_matches ORCL --form 10-K --index 0`: the header-match list jumped straight from `Item 1. Business` to `Item 1B`, skipping Item 1A entirely, with a 158,622-character gap between Item 1 and Item 1B — 5 to 10 times the span of every other company's actual Item 1 section, meaning Item 1A's real content was still in there, just unrecognized and folded into Item 1. Exact mechanism pinpointed with a new purpose-built diagnostic, `scripts/diagnose_orcl_item1a.py`, which searched the raw flattened text for "risk factors" occurrences and printed `repr()` of the surrounding characters to reveal any hidden formatting: Oracle's HTML splits the header **mid-word** across a tag boundary — `"Item 1A.\tR"` on one line, `"isk Factors"` continuing on the next line. This is a more extreme version of the Day 1 AMZN/META tag-split bug (Bug 5): that fix's lookahead-to-the-next-line logic only fired when the *current* line's remainder was completely empty (`not remainder.strip()`), which is true for a clean whole-word split but false here — Oracle's remainder is `"R"`, a single non-empty character — so the Day 1 fix's trigger condition never activated for this case.

Fixed: the lookahead now triggers whenever the title check fails at all (regardless of whether the remainder is empty or a short non-empty fragment), and concatenates `remainder + next_line` **without inserting a space** before re-checking title validity, so `"R" + "isk Factors"` correctly reassembles into `"Risk Factors"` — while the original Day 1 empty-remainder case (`"" + next_line`) is unaffected. 2 new regression tests: `test_split_finds_title_split_mid_word_across_tag_boundary` and `test_split_mid_word_title_combines_without_inserting_a_space` (the latter specifically pins the no-space concatenation behavior). The first version of the test fixture for this bug was actually the wrong shape (modeled as a 3-tag split instead of Oracle's real 2-way mid-line split) — caught because the fix still failed the test even though the fix logic itself was correct, which forced a closer look at the fixture rather than the fix. Corrected fixture: `FAKE_10K_WITH_MID_WORD_SPLIT_TITLE_HTML`, matching Oracle's real HTML shape (`<p>Item 1A.\tR</p><span>isk Factors</span>`).

**Documentation correction (not a code bug):** the Day 1 log's list of companies affected by the known 10-Q Item 1 truncation limitation (originally MSFT/GOOGL/NVDA/ORCL) was incomplete. Real `inspect_chunks` output confirms META has the identical symptom (~100–108 words, every 10-Q, consistently) and was missed from the original scan. Corrected to 5 of 8 companies in `section_splitter.py`'s module docstring, in Section 8 above, and in this doc. The underlying decision to accept the limitation (see Day 1's Bug 6) is unchanged — this is purely a correction to which companies it applies to.

#### Day 3: Hand-labeling the eval set

The 20 single-hop questions needed real, verified gold `chunk_id`s — no fabricated or guessed IDs. Three rounds: **Round 1**, using `scripts/label_eval_set.py`'s interactive top-8 keyword-candidate view, resolved 11 of 20 questions directly. **Round 2**, using the new `scripts/search_chunks.py` (wider top-20, full-text, no truncation) for the 9 questions the top-8 view missed, resolved 8 more by reading the actual candidate text. **Round 3**, after the Oracle Item 1A fix above unblocked real Oracle Risk Factors content, resolved the last question, `sh_019` (Oracle's OCI multicloud competition discussion against Azure/AWS/Google Cloud), against the now-correctly-extracted chunks. **Final: 20/20 single-hop questions labeled**, each with real, verified `gold_chunk_ids` and reasoning recorded in the question's `notes` field. Applied in bulk via `scripts/apply_reviewed_labels.py` rather than requiring every judgment to be re-entered through the interactive tool by hand.

Worth noting for the Day 5 hybrid-retrieval write-up: roughly 9 of the 20 single-hop questions (45%) needed the wider top-20 search to find the real answer — the keyword baseline's top-8 candidates missed the correct chunk on nearly half the single-hop set. This is a concrete, citable number for explaining why hybrid retrieval (not keyword search alone) matters for this corpus.

#### Day 3: Embeddings + Qdrant

All 8 companies embedded with `bge-base-en-v1.5` via `scripts/generate_all_embeddings.py` (model loaded once for the whole run, ~55s load time, then ~4,106s total embedding time across all 8 companies). Confirmed complete against the user's own full run output: 11,245 total embedded chunks written to `data/processed/embeddings/{TICKER}.jsonl`, matching exactly the post-fix chunk counts from the re-run Day 1 pipeline (AAPL 763, MSFT 1612, GOOGL 768, AMZN 1749, META 1676, NVDA 1241, AVGO 2192, ORCL 1244). One environment bug hit and fixed along the way: `python scripts/generate_all_embeddings.py` (direct script invocation, not `-m` module invocation) does not add the repo root to `sys.path`, so `from src.ingestion.config import COMPANIES` failed with `ModuleNotFoundError: No module named 'src'`. Fixed with an explicit `sys.path.insert(0, str(REPO_ROOT))` before that import.

A local Qdrant instance was brought up via Docker (`docker run -p 6333:6333 -p 6334:6334 qdrant/qdrant`, confirmed running cleanly, version 1.19.1). The user had not used Docker before this session; walked through Docker Desktop install → WSL2 requirement → PATH-not-refreshed-in-open-terminal troubleshooting before the container ran successfully. The 8 per-company `qdrant_setup.py` upload commands (AAPL with `--recreate` to create the collection, the remaining 7 without) were issued — **and, per Day 5's follow-on investigation, this is where the real point-id collision bug originated: every command reported success individually, but the collection ended up holding only ~19% of the real data. See the Day 5 summary below for the full root cause and fix; it was not caught at the time this session ended.**

#### Final ingestion state (after all Day 3 fixes)

- 50-question eval set: 20 single-hop (20/20 real, verified gold labels) / 20 multi-hop-numeric (spec-based `gold_sql_description`, no separate labeling needed — consumed directly by the Phase B SQL tool) / 10 out-of-scope.
- Recall@k/MRR scoring + bootstrapped 95% CI code built and tested against the keyword baseline.
- All 8 companies re-ingested with every Day 3 ingestion fix applied (chunk IDs, XBRL dedup, Oracle Item 1A) — 11,245 chunks total.
- All 8 companies embedded (bge-base-en-v1.5), confirmed complete.
- Local Qdrant running; uploads issued for all 8 companies, reported success — **later found on Day 5 to have silently corrupted itself; see Day 5 summary.**
- Test suite: 16 (Day 1) → **45**, all passing (4 for `make_chunk_id`, 11 for `xbrl_extractor.py` across fixes 2/2b/2c/2d, 2 for the Oracle mid-word split, 12 for the Day 3 scoring/eval code, plus the original 16 Day 1 tests untouched).
- XBRL fact rows per company, post-fix: AAPL 48, MSFT 48, GOOGL 48, AMZN 60, META 48, NVDA 48, AVGO 27, ORCL 48.

**Decisions made & why:**
1. **50-question eval set (20/20/10 split), all real** — no fabricated numbers, no guessed chunk IDs anywhere, including for questions the labeling infrastructure initially couldn't resolve; built more tooling (`search_chunks.py`, `apply_reviewed_labels.py`) rather than shortcut the standard.
2. **Multi-hop-numeric questions get no "gold chunk"** — their correct behavior is SQL routing (Phase B), not retrieval, so Recall@k/MRR only ever applies to the 20 single-hop questions; conflating this into one blended metric would hide a real, distinct failure mode. Added `score_sql_routing_sanity()` (not originally scoped for Day 3, but a direct and justified consequence of having 20 numeric questions already in the set) to check whether the keyword baseline falsely returns a confident-looking prose chunk for a question that actually needs a computed number.
3. **`gold_sql_description` fields stay as specs, not hardcoded numbers** — they get consumed directly by the SQL tool once it exists in Phase B (Oct 8); no separate labeling pass is ever needed for them.
4. **Every chunk record now carries a globally-unique `chunk_id`** plus `start_word`/`end_word` — neither existed in the original Day 1 pipeline output; both were silent gaps that would have blocked any real eval scoring or accurate section-length reporting.
5. **Standing instructions established this session, carrying forward to all future sessions:** always give run commands in full (no templated placeholders — a real command was given with a literal `TICKER` placeholder mid-session and the user hit an error running it verbatim); always deliver new/changed files as a zip with `GroundedRAG/` as the top-level folder, verified via clean extraction before sending.

**Metrics this session:**
- 8 companies, 16 filings each (13 for GOOGL, 9 for META), 11,245 chunks total post-fix.
- 20/20 single-hop eval questions labeled with real gold chunks (11 resolved directly, 8 via wider search, 1 after the Oracle fix unblocked it).
- Test suite: 16 → 45, all passing.
- 11,245/11,245 chunks embedded successfully across all 8 companies.
- 6 real bugs found and fixed this session (1, 2, 2b, 2c, 2d, 3) plus 1 documentation correction, all with regression tests except the doc correction.

**Fundamentals covered:**
- Why Recall@k/MRR can't apply uniformly to every question type: multi-hop-numeric questions have no gold chunk by design; their correct behavior is tool routing, not retrieval — conflating this into one blended metric would hide a real, distinct failure mode.
- SEC XBRL company-facts data is not naturally deduplicated by SEC: the same real fact appears multiple times across filings as comparatives, tagged with the *reporting* filing's fiscal year, not the value's own true period — anyone working with SEC XBRL data needs a dedup strategy keyed on the fact's actual `(start, end)` period, not the `fy`/`fp` labels SEC provides (which describe "which filing said this," not "which period this is"). The mid-word HTML tag split (Bug 3) is the same underlying phenomenon — structured extraction code assuming clean boundaries that don't exist in the real-world source — showing up on the unstructured side of the pipeline instead of the structured side.
- A found bug's own fix can have its own bug: fix 2d wasn't found by inspection, it was found because a regression test for fix 2c failed for an unrelated reason — a concrete, narratable example of why regression tests matter even (especially) for the fix itself, not just the original bug.
- Docker fundamentals covered from scratch this session (user's first time using Docker): Docker Desktop install requirements (WSL2 on Windows), why a freshly-installed Docker isn't recognized in an already-open terminal (PATH refresh), and running a local service container with explicit port mapping (`-p 6333:6333 -p 6334:6334`).

**New interview Q&A:** see Section 8 above — three new entries added this session (the Oracle mid-word-split bug narrative, the XBRL fy-vs-period deduplication lesson with the quarter-vs-YTD nuance, and why Recall@k/MRR is scoped to single-hop questions only), plus a correction to one Day 1 entry (5 companies affected by the Item 1 truncation limitation, not 4).

**Opus sessions used so far (target: ≤5 total):**
- 0/5. Every bug this session — including the multi-round XBRL dedup diagnosis and the Oracle mid-word-split root-cause hunt — stayed on Sonnet 5 standard or Sonnet 5 extended thinking. The XBRL and Oracle diagnoses were genuinely non-obvious (this doc's own iteration history shows that), but never ambiguous enough in a way that justified escalating past Sonnet extended thinking — they were solved by building better diagnostics and reading real data carefully, not by needing deeper reasoning about an ambiguous tradeoff.

**Open questions / next steps:**
1. ~~Confirm the 8 `qdrant_setup.py` upload commands completed cleanly~~ — **confirmed by the user at the start of Day 5 as "successfully executed," which was true at the individual-command level; the aggregate corruption this masked was found and fixed during Day 5 itself. See Day 5 summary.**
2. **Day 5 scope per the plan: hybrid retrieval** — BM25 baseline → RRF fusion with dense embeddings (hand-written, `k = 60`) → baseline reranker on top, scored after each addition against this session's keyword baseline and the 20 real gold labels. `rank_bm25` is already in `requirements.txt`, not yet used — that's a Day 5 task, not a Day 3 one. Prerequisites (embeddings, gold labels, keyword baseline, Qdrant instance) are all in place pending the confirmation in item 1.
3. **Day 4 (light reading day) has not been done yet** — not blocking, can be slotted in whenever convenient before Day 5's coding work, per the plan's original design.
4. The moderate-confidence labels among the 20 single-hop questions (flagged in each question's own `notes` field in `eval_questions.json`) are worth a quick manual re-verification pass before trusting them for anything that materially affects a reported metric — **Day 5's near-miss investigation acted on exactly this concern for 3 questions and found one genuine mislabel (sh_011); the remaining 17 are unexamined and still worth a pass if time allows.**
5. No known open code bugs as of this update. The only previously-flagged item (META missing from the documented Item 1 truncation list) was a documentation correction, already applied — not a code issue.
6. Per the plan's session-structure guidance and the user's explicit instruction to wrap up this session once nearing usage limits: **start a new conversation for Day 5** (this thread has now covered Day 1 setup through all of Day 3, plus embeddings and Qdrant setup, and is a natural stopping point) — this summary and the repo state above will carry over via the knowledge base.

---

### Phase A Summary — 2026-09-18 (Day 5: Hybrid retrieval build)

**Context:** Picked up directly from the Day 3 summary above, in a new chat session per the plan's session-structure guidance. This session had a device-linked working environment (able to read/write files and run commands directly in the user's actual `Desktop\GroundedRAG` folder rather than a sandbox copy), which is why fixes below could be verified against the real local Qdrant instance and the real 11,245-chunk corpus rather than handed over untested. Scope was Day 5 per the plan (BM25 → RRF hybrid → reranker, scored after each stage) — but, matching the pattern from both prior sessions, running the real ablation against real infrastructure surfaced genuine bugs that had to be fixed before the numbers could be trusted, plus two eval-label corrections found via a purpose-built recall-ceiling diagnostic.

#### Day 5: Repo analysis and build

Opened by cloning the repo fresh and reading every file in `src/eval/`, `src/ingestion/`, `tests/`, and `data/eval_set/eval_questions.json` directly, cross-checking every claim in the Day 3 summary against the actual code and data (chunk counts, test counts, schema fields) rather than trusting the doc alone — everything matched exactly, no drift found.

Built `src/retrieval/`, empty since scaffolding: `bm25_retriever.py` (`BM25Retriever`, wrapping `rank_bm25.BM25Okapi`, sharing `keyword_baseline.py`'s exact tokenizer so a Recall@k delta reflects BM25's ranking function and not a tokenization difference), `dense_retriever.py` (`DenseRetriever`, wrapping Qdrant + `generate_embeddings.embed_queries`'s asymmetric bge instruction prefix), `hybrid_retriever.py` (`HybridRetriever` + a standalone, independently unit-tested `reciprocal_rank_fusion()` function, `k=60`, pulling a 100-candidate pool from each retriever before fusing so RRF has enough of each ranking to actually combine), `reranker.py` (`RerankerRetriever`, wrapping `cross-encoder/ms-marco-MiniLM-L-6-v2`, reranking the hybrid retriever's top-50). All four expose the identical `.rank(query, ticker=None, k=10) -> List[chunk_id]` interface as Day 3's `KeywordBaseline`, so `src/eval/scoring.py` runs against every one of them completely unchanged — confirmed, not assumed, by running the actual scoring code against each.

Two runner scripts built: `scripts/run_day5_ablation.py` (all 4 stages, one run, writes the full per-question detail to JSON) and, later in the session once the ablation raised a real question the table alone couldn't answer, `scripts/diagnose_day5_ceiling.py` (checks whether gold is even in the reranker's candidate pool, breaks down BM25-only/dense-only/neither-found questions, flags adjacent-chunk near-misses, and dumps hard negatives for Phase C).

#### Day 5: Bugs found and fixed (4 in total this session, numbered fresh since none are direct Day 1/3 code follow-ons, though two are follow-ons to Day 3's *data*)

**Bug 1 — `qdrant-client` 1.19.1 removed `QdrantClient.search()`.** First real run against the user's actual Qdrant instance failed with `AttributeError: 'QdrantClient' object has no attribute 'search'`. `.search()` was deprecated in qdrant-client 1.10 and removed outright by 1.14+; the installed version (1.19.1, same one Day 3 confirmed running) no longer has it. Root-caused by reading the installed package's own source rather than guessing from memory — confirmed `.query_points()` is the replacement, with two differences that both mattered: the vector argument is named `query=`, not `query_vector=`, and the return value is a `QueryResponse` object wrapping `.points`, not a bare list of hits. Fixed in both `dense_retriever.py` (the new Day 5 code) and `src/eval/qdrant_setup.py::sanity_search()` — the latter had the exact same latent bug since Day 3, just never exercised, since Day 3's `__main__` only calls `create_collection`/`upsert_chunks`, both of which are unaffected. `requirements.txt` pinned to `qdrant-client>=1.14` to document the floor this code needs. Regression tests rewrote the test double to expose *only* `.query_points()` (no `.search()` at all), specifically so a fake mirroring the old, wrong assumption couldn't let this regress silently — the original tests had passed precisely because the fake provided a method the real library no longer has.

**Process note, not a code bug, but worth recording honestly:** the assistant advised switching to Opus *before* this first run, reasoning the ablation table deserved the reserved interpretation tier once it existed. That run failed immediately on Bug 1, so that Opus session was spent on a mechanical dependency-version fix rather than genuine reasoning. Correct sequencing, stated afterward: get a successful run on the cheap tier first, escalate only once there's something worth interpreting. This cost the project 1 of its 5 Opus sessions for no reasoning benefit — see the Opus accounting below.

**Bug 2 — Qdrant point-id collision silently corrupted the Day 3 upload down to ~19% of the real data.** The first fully successful ablation run (after Bug 1's fix and after resolving two pure infrastructure issues — a stopped/wrong Docker container, and Qdrant simply not running — neither of which was a code bug) produced a table where BM25→RRF-hybrid changed Recall@1/3/5 by exactly zero. That contradicted a strong prior (a working dense retriever fused with BM25 on paraphrased questions should not be inert), which made "the measurement is broken" the more likely read than "dense doesn't help here." Verified in under two minutes: diffed the per-question retrieved chunk lists between the BM25-only and hybrid stages — byte-identical for 16 of 20 questions, which a genuine two-list fusion essentially never produces by chance. Root cause: `qdrant_setup.py::upsert_chunks()` assigned each point's Qdrant id via `id=i` from `enumerate()` — and since the script runs once per company, that counter restarted at 0 on every one of the 8 Day 3 upload commands. Qdrant's `upsert` treats a colliding id as an update, not an error, so each company's upload silently overwrote the previous one's points; the surviving collection held only ORCL's points plus the tail of AVGO's (~2,192 of the real 11,245). Every individual Day 3 upload command genuinely reported success — the bug was invisible at the level each command could see, and only showed up as an aggregate property nobody had checked. Fixed with `make_point_id()`, deriving a deterministic `uuid5` from the chunk_id Day 3's `make_chunk_id()` had already built (globally unique, idempotent — re-uploading a company updates its own points rather than duplicating or colliding). Also added a post-upload point-count check to `qdrant_setup.py`'s `__main__` that prints the collection's running total and warns explicitly if it's smaller than expected, so this exact class of silent aggregate corruption can't recur without a visible warning. 7 new regression tests in `tests/test_qdrant_setup.py`, including one that directly simulates two sequential per-company uploads and asserts all points from both survive.

After the fix, the user re-uploaded all 8 companies (`--recreate` on the first, not on the remaining 7) and re-ran the ablation: dense now changes the ranking on 20/20 questions (was 4/20 before the fix), and Stage 2's Recall@10 more than doubled (0.15 → 0.35) with no further code changes.

**Bug 3 — sh_011's gold label was flatly wrong, found via a near-miss diagnostic that wasn't looking for this.** `scripts/diagnose_day5_ceiling.py` flagged 3 questions (sh_001, sh_003, sh_011) where a chunk adjacent (±1) to the labeled gold sat in the hybrid retriever's top-10 without gold itself being counted — a possible Recall@k understatement given 250-word chunks with 40-word overlap. Reading all three adjacent-chunk pairs' actual text against their gold chunks directly (not trusting the heuristic) produced three different outcomes. sh_001 was a false positive: the adjacent chunk discussed an unrelated risk (new-product-transition risk, not the asked-about China-manufacturing risk) — coincidental structural proximity only, no change made. sh_003 was a genuine near-miss (see Bug 4 below). sh_011 (*"What does Amazon's MD&A cite as a driver of change in AWS segment operating income?"*) turned out not to be an adjacency question at all: reading two chunks past what the automated ±1 check even looked at, the originally-labeled gold chunk (`AMZN_..._item_7_0025`) was found to discuss the FTC lawsuit settlement and North America/International segment operating income — it never mentions AWS anywhere. The actual answer — *"the increase in AWS operating income in 2025...is primarily due to increased sales, partially offset by spending on technology infrastructure"* — is in `_0026`, one chunk further. This was a genuine Day 3 labeling error that the diagnostic's own ±1 window wasn't even designed to catch; it was found only by continuing to read past what the tool flagged. Corrected `gold_chunk_ids` to `["_0026"]` alone (the old id doesn't partially answer the question, so it wasn't kept as a second gold id the way sh_003's was). Documented in `eval_questions.json`'s `notes` field and in `CHANGES.md`, matching the exact convention Day 1/Day 3 established for documenting a correction without rewriting the original historical record (`scripts/apply_reviewed_labels.py`, the Day 3 labeling script, was left untouched as a historical log, same as Day 1's original bug count was never edited when Day 3 corrected it).

**Bug 4 (not a bug — a legitimate label widening) — sh_003's chunking-boundary near-miss.** The adjacent chunk (`AAPL_..._item_1a_0041`) is the literal immediately-preceding sentence of the *same* continuous foreign-currency-risk paragraph as the existing gold chunk (`_0042`) — not a coincidence, confirmed by two independent signals: reading both chunks' full text end-to-end shows one continuous argument split by the chunker's window boundary, and dense retrieval (uninfluenced by the gold label) independently ranked `_0041` at position 2, strong evidence it's a genuinely good semantic match and not merely nearby. `gold_chunk_ids` widened from 2 to 3 entries (the pre-existing second id — the same passage recurring near-verbatim in the 2024 filing — was untouched). After the correction and a full re-run, `_0041` scored a real hit: hybrid rank 8, promoted to rank 3 by the reranker — independent confirmation the widening captured a real, retrievable match rather than inflating the count on paper.

**Net effect of the two label corrections on the aggregate numbers, checked explicitly rather than assumed:** Stage 2 Recall@10 held at 7/20 (0.35) — sh_003 flipping from miss to hit and sh_011 flipping from (incorrectly-scored) hit to (correctly-scored) miss canceled out in raw count. Stage 2 MRR dropped slightly (0.087 → 0.080), honestly reflecting that the lost "hit" was well-ranked under the wrong label (1/5) while the gained one is weaker (1/8). Every stage was re-run and re-verified per-question against the corrected labels before treating any number as final.

#### Day 5: The final, corrected ablation table

| Stage | Recall@1 | Recall@3 | Recall@5 | Recall@10 | MRR |
|---|---|---|---|---|---|
| 0. Keyword baseline | 0.00 | 0.00 | 0.05 | 0.10 | 0.015 |
| 1. BM25 | 0.05 | 0.10 | 0.10 | 0.15 | 0.082 |
| 2. RRF hybrid (BM25 + dense, k=60) | 0.00 | 0.10 | 0.10 | **0.35** | 0.080 |
| 3. + baseline reranker | **0.10** | 0.15 | 0.20 | 0.25 | **0.133** |

**BM25 alone (Stage 1) is weak, and this was independently verified as a real finding, not a bug**, before building further: the gold chunk almost always has *some* nonzero token overlap (~99% of a company's chunks score nonzero against any given question, since neither the shared `tokenize()` nor `rank_bm25` filter stopwords), but its rank varies from 1 to 785 out of 700–2,200 candidates. Root cause: 250-word fixed chunks fragment topically coherent passages, and the eval questions are paraphrased rather than verbatim quotes, so lexical overlap alone doesn't discriminate well. This is exactly the gap dense/hybrid retrieval is meant to close, and Stage 2's numbers confirm it does, for Recall@10 specifically.

**RRF (Stage 1→2) is a recall mechanism, not a precision one**: more than doubled Recall@10 while leaving MRR essentially flat — it pulls correct chunks into the top-10 without improving their ordering, which is expected from a rank-position-only fusion formula that discards score magnitude.

**The reranker (Stage 2→3) traded recall for precision, and the trade is real and diagnosable, not noise.** Per-question: promoted 4 questions (two straight to rank 1, which is the entire Recall@1 gain), demoted 1, and — critically — **rescued zero** of the ~40 candidates it saw at ranks 11–50 across the whole eval set, while losing 2 marginal hits (ranks 9 and 10). A cross-encoder with real in-domain signal should surface *something* from that pool; zero rescues across 20 questions is a specific, citable data point that the off-the-shelf `ms-marco-MiniLM-L-6-v2` (trained on short web-passage/natural-language-query pairs) has a real domain gap against 250-word SEC legal-financial prose.

#### Day 5: The recall-ceiling diagnostic — separating a reranker problem from a retrieval problem

Built `scripts/diagnose_day5_ceiling.py` specifically because the ablation table alone can't distinguish "the reranker sees the right chunk but scores it wrong" (fixable by fine-tuning the reranker) from "the reranker never sees the right chunk at all" (not fixable by reranking, full stop — this is exactly the go/no-go question Phase C's plan-flagged Opus session was reserved for). Results, on the corrected labels:

- **Gold in hybrid top-10 (measured Recall@10): 7/20 (35%).**
- **Gold in hybrid top-50 (the reranker's actual candidate pool — the ceiling): 12/20 (60%).** A perfect reranker could reach 5 more questions than the current one does (5/20 achieved vs. 12/20 reachable) — real, measured headroom, and the case for Phase C's LoRA fine-tune.
- **But the ceiling itself caps at 60%, not 100%**: 8/20 questions (40%) have gold beyond rank 50 even at a 200-deep search, and 3 of those aren't in the top 200 of *either* retriever. No reranker — fine-tuned or not — can recover a chunk it never sees. This is a genuine, separate retrieval/chunking ceiling, most likely explained by the same 250-word fixed-chunking fragmentation implicated in the BM25 finding above. Worth investigating on a future light/buffer day, but explicitly **out of scope for what a LoRA-fine-tuned reranker can fix**.
- **Adjacent-chunk near-miss rate: unchanged at 3/20 (15%) after the label corrections** — composition shifted (sh_003 resolved out of the flagged set as a genuine hit; sh_011 entered it, now showing its *old, wrong* gold chunk as the "adjacent" match) but the aggregate rate held, a useful stability check that the corrections didn't just move noise around.
- BM25-only finds: 1. Dense-only finds: 2. Found by neither (within 200 deep): 1. The two retrievers mostly fail on the *same* questions rather than compensating for each other — worth keeping in mind for Phase C's hard-negative selection, since a hard negative both retrievers agree on is a stronger training signal than one only weak in a single retriever.

#### Final state at end of Day 5

- `src/retrieval/` fully built: BM25, dense (Qdrant), RRF hybrid (`k=60`, hand-written and unit-tested against hand-computed scores), baseline cross-encoder reranker. All share one interface; all run through the unmodified Day 3 scoring code.
- Full 4-stage ablation table measured against the real, corrected 11,245-chunk corpus and 20 corrected gold labels — see table above. Written to `data/eval_set/day5_ablation_results.json` with full per-question detail.
- Recall-ceiling diagnostic run and written to `data/eval_set/day5_ceiling_diagnostic.json`, including a `hard_negatives` field per question — a direct input for Day 6's mining task.
- 3 real infrastructure bugs found and fixed (`qdrant-client` API removal, the point-id collision, plus the Docker-container/service-not-running issues that were pure ops rather than code). 1 real eval-labeling bug found and fixed (sh_011). 1 legitimate label widening (sh_003). 1 candidate investigated and correctly ruled out as a non-issue (sh_001) — logged as a negative finding, not silently dropped.
- Test suite: 45 (Day 1 + Day 3) → **83**, all passing (6 BM25, 8 dense — including 2 regression tests pinning the `query_points()` migration, 9 hybrid/RRF, 8 reranker, 7 qdrant_setup point-id — 38 new tests total).
- `CHANGES.md` updated with the Day 5 label-correction entry, matching the project's established convention for documenting a data correction without rewriting the historical record of what was originally applied and why.
- `requirements.txt` pinned `qdrant-client>=1.14`.

**Decisions made & why:**
1. **Point ids derived deterministically from `chunk_id` (`uuid5`), never from a positional counter** — the entire Bug 2 class of silent corruption is structurally impossible once ids are derived from something already globally unique, rather than reset per-invocation.
2. **A collection's point count gets verified after every upload, not just each command's own exit status** — Bug 2 was invisible specifically because nothing asked the aggregate question; this is now a standing check, not a one-off fix.
3. **A near-miss/adjacency diagnostic is a lead generator, not a verdict** — every flagged candidate gets its actual text read before a label changes, and the diagnostic's own scope limit (it only checked ±1) was itself the reason sh_011 needed reading *past* what it flagged, not just trusting what it did flag.
4. **Two gold-label changes both required reading real text before touching `eval_questions.json`, same discipline as Day 3** — one widening (sh_003, independently corroborated by dense retrieval's own ranking) and one correction (sh_011, the label simply didn't answer the question). The historical labeling script (`scripts/apply_reviewed_labels.py`) was left untouched as a record, with the correction documented separately — same convention as the Day 1→Day 3 META correction.
5. **Bug 1's Opus spend on a mechanical fix is recorded honestly as a process mistake, not quietly absorbed** — the sequencing lesson (validate on the cheap tier before escalating to the reserved tier) is now a standing note in Section 2, and the resulting Opus-budget shortfall (2 sessions left for 3 still-flagged moments) is flagged explicitly for Phase B to address at its outset rather than discovered under pressure later.

**Metrics this session:**
- Point-id bug: real collection size went from ~2,192 (19% of expected) to the full 11,245 after the fix and re-upload.
- Dense retrieval's contribution to the fused ranking: 4/20 questions changed pre-fix → 20/20 post-fix.
- Final ablation table: see above (Recall@10 climbs 0.10 → 0.15 → 0.35 → 0.25 across the 4 stages; MRR climbs 0.015 → 0.082 → 0.080 → 0.133).
- Recall ceiling: 35% achieved, 60% reachable with a perfect reranker, ~40% out of reach of any reranker.
- Reranker: 4 promotions (2 to rank 1), 1 demotion, 0 rescues from ranks 11–50, 2 losses at ranks 9–10.
- Eval labels: 1 corrected (sh_011), 1 widened (sh_003), 1 investigated and correctly left unchanged (sh_001).
- Test suite: 45 → 83, all passing.
- Opus sessions this project: 0 → 3/5 (1 on the genuine ablation read the plan intended, 1 lost to sequencing, 1 more on... — see full accounting below).

**Fundamentals covered:**
- A pipeline step reporting success per-call doesn't guarantee correctness in aggregate if calls can silently overwrite each other's output — Qdrant's upsert-on-collision behavior turned 8 individually-correct upload commands into a nearly-empty collection, and the fix that actually prevents recurrence is a verification step, not just a smarter id scheme.
- A result that contradicts a strong, well-founded prior is more likely a broken measurement than a genuine finding, and that hypothesis is usually checkable in minutes (the byte-identical-list diff) before committing to either read.
- Recall@k and MRR measure genuinely different things, now with a concrete empirical demonstration rather than just the Day 3 definition: RRF fusion and the reranker moved these two metrics in *opposite directions* at different stages of the same pipeline.
- Before fine-tuning a component to fix a low score, check whether that component even has access to the right answer — a recall-ceiling check separates "this component is bad at its job" from "this component was never given the material to succeed," and conflating them risks spending a fine-tuning budget on the wrong target.
- An automated diagnostic that flags a candidate (the ±1 adjacency check) is a lead, not a verdict, in both directions: it can miss a real bug outside its designed scope (sh_011, found only by reading further than the tool looked) just as easily as it can flag a coincidence (sh_001).

**New interview Q&A:** see Section 8 above — six new entries added this session, covering the point-id collision bug and its "success per-step, wrong in aggregate" lesson, the byte-identical-lists detection method, the Opus-sequencing mistake told straight rather than smoothed over, the recall-ceiling methodology and why it was built before committing to Phase C's fine-tune, the three-different-outcomes near-miss investigation (false positive / genuine widening / genuine mislabel), and the empirical Recall@k-vs-MRR divergence now backed by this session's own numbers instead of just the Day 3 theory.

**Opus sessions used so far (target: ≤5 total):**
- **3/5.** Full accounting: one session was spent on the intended ablation read once the corrected table existed (the genuine Opus session #1 from the plan, covering the per-question reranker analysis, the RRF-vs-reranker recall/precision tradeoff read, and the recall-ceiling go/no-go framing for Phase C). One session was spent, in the assistant's own words, on a sequencing mistake — advising a switch to Opus before the code had run even once, so it was burned on the mechanical `qdrant-client` API fix instead of reasoning. **A third Opus-model turn occurred mid-session** (the user switched back to Opus before pasting the point-id-collision-revealing table) — recorded here for an honest running total even though the diagnosis that turn produced (the byte-identical-lists check, the root cause) was mechanically simple enough that it likely didn't need the Opus tier either; this is flagged as a second, smaller instance of the same sequencing lesson, not hidden. **Practical consequence, stated plainly for Phase B:** only 2 Opus sessions remain against 3 still-flagged plan moments (agent routing-logic design, LoRA go/no-go, failure-analysis synthesis). Phase B must decide explicitly how this gets absorbed — most likely candidate is downgrading the LoRA go/no-go to Sonnet extended thinking, since Day 5's ceiling diagnostic already did most of the reasoning that decision needs.

**Open questions / next steps:**
1. **Day 6 (buffer / hard-negative mining prep) is the next task.** `data/eval_set/day5_ceiling_diagnostic.json` already has a `hard_negatives` field per question ready to mine from, and sh_011's specific failure pattern (retriever finds the surrounding context, not the sentence-bearing chunk) is a named, understood hard-negative *type* worth deliberately oversampling for rather than mining incidentally.
2. **The Opus budget shortfall (2 remaining for 3 flagged moments) needs an explicit decision at the start of Phase B**, not a default. Candidate: downgrade Phase C's LoRA go/no-go to Sonnet extended thinking, since Day 5 already produced most of the needed evidence (the 60%-ceiling-vs-25%-actual finding).
3. **The ~40% retrieval/chunking ceiling** (gold not found by either retriever even at 200-deep search) is unexplained beyond "likely fixed-chunking fragmentation" and worth a real investigation on a buffer/light day — it directly bounds how good Phase C's fine-tune can make the system look, independent of how well the fine-tune itself goes.
4. **Day 4 (light reading day) is still not done** — unblocking slack, not a dependency; can still be slotted in whenever convenient.
5. **17 of the 20 single-hop gold labels are still unexamined** by the kind of direct-text-verification Day 5 applied to 3 of them. Not urgent (no signal they're wrong), but the same process that found sh_011 wrong hasn't been run against the rest.
6. No known open code bugs as of this update.
7. Per the plan's session-structure guidance: **start a new conversation for Day 6.** This session's summary above, the corrected `eval_questions.json`, the built `src/retrieval/` package, and both runner scripts all carry over via the knowledge base and the repo itself.

---
