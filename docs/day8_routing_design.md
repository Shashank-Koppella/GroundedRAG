# Day 8 routing design: retrieve vs structured vs refuse

*Opus design session, Oct 5 2026. Inputs: the RAG-only numeric baseline, the Day 8 facts tool and calculator, the Day 7 faithfulness results, and a blind routing probe set built in this session. Implementation is Day 9 work (Sonnet); the build order is at the end.*

## 1. Decisions in one screen

1. **Route on what the answer depends on, not on the topic.** A question is `structured` when its answer is a company-total revenue or net-income value (or arithmetic on such values), `retrieve` when the answer is stated in filing text, `hybrid` when both are asked for, and `refuse` when the corpus cannot answer it. "What drove revenue growth" mentions revenue but is a text question; "how much did Nvidia bring in" names no metric but is a structured one.
2. **Four routes, not the planned four tools.** The calculator is a step inside `structured`, never a route of its own. "Answer directly" from model knowledge is dropped: in a system whose point is grounding, an ungrounded route is a liability. Refusal is a route with a typed reason.
3. **Plan → guard → execute → compose, not a free-form ReAct loop.** One LLM call writes a typed plan. Deterministic code checks it, runs the tools, and renders the answer. The loop is bounded: at most one re-plan, and only when the guard rejects the plan. A tool's refusal is relayed, not retried around.
4. **On the structured path, the LLM never writes a number.** Values come from the facts table, arithmetic from the calculator, and the sentence from a template that names the periods. Day 7 showed the NLI scorer cannot check numbers, so this path is made checkable by construction instead.
5. **The rules are a guard, and the guard only moves toward refusal.** `src/agent/route_rules.py` holds precise refusal triggers (real-time, advice, forecast, non-filing sources, companies outside the eight) that overrule the planner. The rules never upgrade a refusal into an answer.
6. **Whether the rules or the planner owns the route decision is decided by measurement, by a rule fixed now** (Section 5). The rules alone already score 45/48 on blind questions, so the planner has to earn the job.
7. **Retrieve answers to "latest/most recent" questions get a deterministic recency check.** If the cited filing is not the company's latest of its form, the answer says so. On the baseline this flags all 3 wrong answers.

## 2. Evidence

### 2.1 RAG-only on the 20 numeric questions: 0 correct, 3 confidently wrong

`python -m scripts.run_day7_generate --retriever reranked --k 5 --context-mode retrieved --prompt-version v2 --types multi_hop_numeric`, compared with `data/eval_set/day8_mhn_reference.json`:

| outcome | n | detail |
|---|---|---|
| refused | 17 | the model judged the retrieved context insufficient. **Not checked per question** whether the context really lacked two fiscal years; this is a hypothesis, not a finding |
| answered, wrong | 3 | each cites a real chunk that states the number, but for an older year |
| answered, right | 0 | |

The three answers, checked against the chunk text:

- **mhn_006 (GOOGL net income):** said +23% (2023 vs 2022, from the 10-K filed 2024-01-31). True latest change is +32.01% (FY2025 vs FY2024). The 23% is stated in the chunk and is correct for 2023.
- **mhn_013 (AVGO revenue):** said +8% (FY2023 vs FY2022, 10-K filed 2023-12-14). True latest change is +23.87%. Two years stale.
- **mhn_007 (AMZN revenue):** said "12% for 2024 versus 2023" (10-K filed 2025-02-07). The chunk's table lists consolidated growth as `12 11` under columns `2023 2024`, so 12% is 2023's growth and 2024's was 11.0% (637,959 / 574,785). The model misread the columns. The true answer, FY2025 vs FY2024, is 12.38%, so **this wrong answer would pass a "does the number round to the right value" check by coincidence.**

Why retrieval cannot fix this: "the two most recent fiscal years in the dataset" is a property of the whole corpus. No single chunk can show that it is the latest, so even perfect top-k retrieval cannot certify recency. Those questions belong to a tool that sees the whole table.

A second retrieval finding: for mhn_017 (Apple vs Microsoft), the multi-company question ran unfiltered and the top 5 came back MSFT and **NVDA**, not Apple. Multi-company retrieval has to run once per company with a ticker filter.

### 2.2 Rules alone: perfect on the eval set, which proves little; 45/48 on blind questions

`src/agent/route_rules.py` was written and frozen (sha256 `09d78c22…56af7`) before any probe question existed. *(Revision 2, Oct 5 audit, sha256 `8a80dedd…c91e25`: precision fixes for ordinary questions the audit found wrongly refused — "invest in", "arm"/"visa"/"hp"/"sap" as company names, "sales and marketing" read as revenue. It changes **none** of the 48 probe predictions (verified row by row) and still scores 50/50 on the eval set, so every number below stands. A brand-as-segment change was tried and reverted because it broke a probe item; see CHANGES.md.)*

| set | route accuracy | note |
|---|---|---|
| eval set (50) | 50/50, tickers exact | rules written with these questions visible; the set is templated, so this only shows it cannot separate routers |
| blind probe (48) | **45/48 = 0.938, 95% CI 0.854–1.000** (n=48) | written by a separate agent from the route definitions only; labels reviewed against the definitions, none changed |

Probe breakdown: plain 16/16, paraphrase 15/15, trap 14/17; tickers exact on all 34 correctly routed answerable questions; refusal reason right on 10/11 (rp_039 was tagged `real_time` because of "right now"; the gold reason is `advice`).

**All 3 misses are critical, and all 3 are open-vocabulary failures:**

- rp_024, "how much Nvidia **brought in**… percent difference": no metric word, so it fell to the default, `retrieve`.
- rp_043, "Why did Alphabet's net income change? **Include the actual figures.**": the second-sentence request for numbers is not a recognized hybrid join, so `retrieve`.
- rp_045, "How does Intel's data center revenue **stack up against** Nvidia's?": the comparison idiom is not in the compare list, so the Intel veto did not fire, and the question went to `retrieve`.

The structural point: when the rules have no evidence they default to `retrieve`, which is exactly where the baseline produced stale numbers. A default of `refuse` would trade those misses for over-refusals. Neither default fixes open vocabulary, and a paraphrase-robust planner is the candidate fix. The three misses are pinned as strict `xfail` tests so a later fix is noticed and re-measured, not claimed.

### 2.3 Carried from Day 7 and the Day 8 tools

- NLI faithfulness cannot verify numbers, and prompt v2 forbids the generator from computing. Derived figures therefore must come from the calculator.
- The tools refuse rather than guess: no row returns an empty list; a non-positive base raises `CalcError`; a hole in the quarters gives `complete: False`; a stale series gives a caveat (AVGO net income).

## 3. Routes

| route | answer depends on | executes | answer is written by |
|---|---|---|---|
| `structured` | company-total revenue / net income values, or arithmetic on them | `FactsDB` typed lookups, then `calculate` | template (no LLM) |
| `retrieve` | statements in filing text, including figures for metrics the table does not hold (segments, gross/operating margin, EPS, capex, headcount, buybacks) | hybrid retriever (per ticker), then generator, prompt v2 | generator (existing Day 7 path) |
| `hybrid` | both, explicitly asked in one question | both, independently | template paragraph + generator paragraph, with provenance kept separate |
| `refuse` | anything outside the corpus | nothing | template with the typed reason |

Refusal reasons: `real_time`, `forecast`, `advice`, `not_in_filings`, `uncovered_company`, `no_covered_company`, plus `data_unavailable` (raised by the executor, see 4.4).

A figure the text states for an unsupported metric goes to `retrieve`, not `refuse`. Its known risk is the stale year, which the recency check in 4.5 addresses.

## 4. Architecture

```
question
  → guard.pre  (route_rules.hard_refusal: veto → refuse)
  → planner    (1 LLM call, temp 0, cached; emits a typed Plan)
  → guard.post (schema, metric/op whitelist, ticker check, rules-vs-planner disagreement logged)
       ├─ reject → one re-plan with the rejection message → still invalid → rules-only fallback or refuse
  → executor   (deterministic; tools only)
  → composer   (template for numbers; generator for text; recency note)
  → answer + trace
```

### 4.1 Plan schema (planner output, validated before anything runs)

```json
{
  "route": "structured | retrieve | hybrid | refuse",
  "tickers": ["NVDA", "AVGO"],
  "refuse_reason": null,
  "structured": {
    "operation": "value | pct_change | margin | ratio | mean_quarters | compare",
    "metric": "revenue | net_income",
    "numerator": "revenue | net_income",
    "denominator": "revenue | net_income",
    "periods": {"kind": "latest_fy | last_n_fy | fy | recent_quarters", "n": 2, "fiscal_year": null},
    "compare": {"of": "pct_change | value | margin", "pick": "max | min"}
  },
  "text_question": "the part to answer from filings (retrieve / hybrid only)"
}
```

- `ratio` and `margin` carry an explicit numerator and denominator. mhn_020 asks for revenue ÷ net income, and a silent swap would still produce a plausible number. The template prints "revenue ÷ net income", so the order is visible.
- **Fiscal-year naming:** "fiscal year N" means the fiscal year whose period end date falls in calendar year N. This was checked against every fiscal-year end date in the table: NVDA's year ending 2025-01-26 is its "fiscal 2025", and AVGO's year ending 2024-11-03 is "fiscal 2024". The raw XBRL `fy` label is never used (Day 8 traps). A calendar-year question about a non-calendar-year company is answered with the fiscal period stated explicitly.

### 4.2 Guard (deterministic; `route_rules.py` plus validation)

- **Pre-veto:** `hard_refusal(question)` fires, so the route is `refuse` and the planner is not called. Triggers are tuned for precision, because a false positive here cannot be undone.
- **Post-checks on the plan:**
  - schema-valid;
  - metric and operation are in the whitelist;
  - every planner ticker is named in the question (`find_tickers`), unless the question names none and is corpus-wide. An inferred ticker is allowed but flagged `ticker_inferred` in the trace.
  - A `structured` plan with an unsupported metric is rejected with the message "metric not in the facts table; the filing text may state it (route `retrieve`)". That triggers the single re-plan.
- **Disagreement:** the rules' route is always computed and logged next to the planner's, which gives Day 9 a free paired comparison. Under policy B (Section 5), a planner `refuse` against a rules answer stands; under either policy the guard never overrules a refusal.

### 4.3 Executor (deterministic)

| operation | recipe |
|---|---|
| `value` | `latest_fiscal_year(t, m)` or `quarter_window(t, m, 1)` *(the Oct 5 audit found and fixed `recent_quarters(n=1)` returning 16 quarters)* |
| `pct_change` | `fiscal_years(t, m, 2)` → check the two years are consecutive (0.9–1.1 years apart) → `pct_change(old, new)` |
| `margin` / `ratio` | latest fiscal year of each metric → **same end date required**, else `data_unavailable` → `pct` / `ratio` |
| `mean_quarters` | `recent_quarters(t, m, n)` → `complete` must be true → `mean(...)`; a derived Q4 adds a caveat |
| `compare` | run the sub-operation per ticker → if any leg is `unanswerable`, the comparison is unanswerable; caveats from every leg carry over → `max`/`min` |

- Every lookup result carries `status ∈ {ok, caveat, unanswerable}`. The staleness check that was in `build_mhn_reference.py` (one metric's latest fiscal year is more than 0.8 years behind the company's other metric) has moved into `FactsDB` (`latest_fiscal_year`, `fiscal_year_pair`; Day 9 step 1, done), so the agent gets it, not only the reference builder.
- **Comparison periods:** each company's own latest fiscal year is used and both period ends are printed. When the ends differ by more than 6 months (AAPL Sep-2025 vs MSFT Jun-2026), a note is added. This is a note, not a caveat; the question asked for each company's latest year.
- **Retrieve route, multi-company:** retrieve once per ticker with the ticker filter, k=5 each, and merge (fix for the mhn_017 finding). For a single ticker, keep the Day 7 path unchanged.

### 4.4 Failure handling (what the loop does when something says no)

| event | behaviour | why |
|---|---|---|
| tool returns `unanswerable` (stale, hole, non-positive base, periods misaligned) | answer = refusal with the tool's reason (`data_unavailable`); **no fallback to retrieval** | the baseline shows what text fallback produces for these questions: a stale number with a real citation |
| plan names a metric the table lacks | one re-plan, steered to `retrieve` | the text states such figures (route definitions) |
| plan invalid twice | use the rules-only route if the rules fired positive evidence; otherwise refuse ("could not interpret") | bounded; no third LLM call |
| Groq error / timeout | an **error**, not a refusal, recorded as such | keeps refusal rates honest (Day 7 convention) |
| tool calls exceed 12 (8 companies × lookups is the worst case) | stop, report as error | runaway guard |

### 4.5 Composer

- **Structured:** template only, for example: "Nvidia's revenue rose 65.47% from $X (fiscal year ended 2025-01-26) to $Y (fiscal year ended 2026-01-25). Source: XBRL facts, Form 10-K." Each caveat is printed verbatim and never summarized away.
- **Retrieve:** the existing `RAGGenerator`, prompt v2, built so the message payload is **byte-identical** to the Day 7 pipeline for single-company questions. With the payload-hash cache, the agent's single-hop answers are then provably identical to the Day 7 v2 run, so there is zero regression by construction.
- **Recency note:** if the question asks for the latest or most recent and a cited chunk's filing is not the company's latest filing of that form in the corpus, append "Note: this comes from the {form} filed {date}; a newer {form} (filed {latest}) exists in the dataset." This is deterministic and was checked on the baseline: all 3 wrong answers cite stale 10-Ks (GOOGL 2024-01-31 vs latest 2026-02-05; AMZN 2025-02-07 vs 2026-02-06; AVGO 2023-12-14 vs 2025-12-18).
- **Hybrid:** the structured paragraph, then the generator's answer to `text_question`. Faithfulness scoring applies to the text paragraph only, and numeric checking to the structured one.

### 4.6 Trace

Every answer carries its trace: guard verdicts, the raw plan and the validated plan, each tool call and its result, the composer inputs, and the rules-vs-planner disagreement. The trace is how a routing decision is explained, and it is the input to the Phase D failure analysis.

## 5. Routing policy: decided by a rule fixed before the measurement

Two policies are run on the 48-question probe set on Day 9, scored by `scripts/eval_routing.py` (the same code that scored the rules):

- **A — rules first.** Use the rules' route; call the planner's route only when the rules fell through to their default (`fired == ["default: filing text"]`) or found no company. The planner still fills the structured plan in both policies.
- **B — planner first.** Use the planner's route, with the guard's vetoes.

**Decision rule (stated now):**
- Adopt B only if it has **fewer critical misses than A** and **no more over-refusals than A** on the probe set.
- Otherwise adopt A. On a tie A wins: it is deterministic, cheaper and explainable.
- Rules-only (45/48, 3 critical) is the floor that both must meet.

Honesty notes for the write-up:
- The designer has now seen the probe set. The planner prompt is therefore written only from the route definitions, with examples taken from the eval set and none from the probe.
- Any later change to the rules or the prompt needs a fresh probe set (probe v2), because this one is now single-use. That is the same exhaustion problem as the 10 out-of-scope questions.

## 6. How the agent is evaluated (Day 9)

| measure | set | pass condition (stated now) |
|---|---|---|
| route accuracy + critical/over-refusal counts, paired A vs B | probe (48) | Section 5 rule |
| numeric correctness | 20 `multi_hop_numeric` | value matches the reference to 2 dp **and** every period end date matches the reference inputs; status matches (`ok` vs `caveat`), and each caveat is present in the answer |
| confidently wrong | 20 `multi_hop_numeric` | 0, against the RAG-only baseline's 3 |
| single-hop regression | 20 `single_hop` | answers byte-identical to `day7_answers_reranked_k5_retrieved_pv2.json` |
| refusals retained | 10 `out_of_scope` | 10/10 (a seen set, so this is a regression check, not evidence) |

The period check is required, not optional: mhn_007 shows a wrong answer that matches the right number by coincidence.

## 7. Known gaps

- n=48 and n=20. The probe CI is 0.854–1.000, so routers that differ by one or two questions cannot be ranked. The decision rule counts severities rather than chasing accuracy differences.
- The AVGO net-income extraction gap makes mhn_014/018 `caveat`. Fixing `xbrl_extractor.py` changes the reference file; regenerate it and log the change in CHANGES.md.
- The recency note covers "latest" questions only. A retrieve answer about a named past year is not checked against the period it names (prompt v2's citation date makes that checkable later).
- Hybrid is in the schema, but neither the eval set nor the RAG baseline exercises it. Its only evidence is 8 probe questions.

## 8. Build order (Day 9, Sonnet 5 standard; extended thinking for the planner prompt)

1. **(DONE Oct 5, see CHANGES.md "Day 9 build, step 1")** `FactsDB`: add `status`/`caveats` to lookups (staleness, consecutive-year, same-end-date checks), moved out of `build_mhn_reference.py`; tests.
2. `src/agent/plan.py`: Plan dataclass + validator (schema, whitelist, ticker check); tests with hand-written good and bad plans.
3. `src/agent/executor.py`: operation recipes; tests reproducing every reference value through the executor (must equal `day8_mhn_reference.json`).
4. `src/agent/compose.py`: templates, recency note; tests including the 3 baseline answers → recency note fires.
5. `src/agent/planner.py`: prompt from the route definitions (no probe examples), JSON parse, one re-plan; offline tests with a fake client.
6. `src/agent/agent.py`: the bounded loop + trace; `scripts/run_day9_agent.py`; `ROUTERS["planner_a"]`, `["planner_b"]` in `scripts/eval_routing.py`.
7. Run: policies A and B on the probe, the agent on all 50 eval questions, scoring per Section 6. The planner and generator need Groq and Qdrant, so these run on the user's machine.
