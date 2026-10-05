"""Tests for src/agent/route_rules.py and scripts/eval_routing.py."""
import json
from pathlib import Path

import pytest

from src.agent.route_rules import TICKERS, classify, find_tickers, find_uncovered, hard_refusal
from scripts.eval_routing import EVAL_ROUTE, SEVERITY, evaluate

REPO = Path(__file__).resolve().parent.parent
EVAL = REPO / "data" / "eval_set" / "eval_questions.json"


# ---------------------------------------------------------------- entities

def test_tickers_in_order_of_mention_and_deduplicated():
    assert find_tickers("Between Nvidia and Broadcom, and NVDA again") == ["NVDA", "AVGO"]
    assert find_tickers("How is AWS doing versus Azure?") == ["AMZN", "MSFT"]
    assert find_tickers("Instagram and Facebook growth at Meta") == ["META"]


def test_aliases_need_word_boundaries():
    assert find_tickers("the metadata of pineapple orchards") == []
    assert find_uncovered("Is armour a word? and the hpc market") == []


def test_uncovered_companies_detected():
    assert find_uncovered("What did Tesla report?") == ["tesla"]


# ---------------------------------------------------------------- hard refusals (the guard)

@pytest.mark.parametrize("q,reason", [
    ("What is Apple's stock price?", "real_time"),
    ("Should I buy Nvidia shares?", "advice"),
    ("What will Meta's revenue be next fiscal year?", "forecast"),
    ("What did Amazon's CEO say on the earnings call?", "not_in_filings"),
    ("What did Tesla report as revenue?", "uncovered_company"),
    ("Compare Apple's margins to Samsung's.", "uncovered_company"),
])
def test_hard_refusals(q, reason):
    assert hard_refusal(q)[0] == reason
    v = classify(q)
    assert v.route == "refuse" and v.hard and v.refuse_reason == reason


@pytest.mark.parametrize("q", [
    "What does Oracle's MD&A say about its outlook for next year?",        # outlook stated in a filing
    "What does Nvidia's 10-K say will be the impact of export controls?",  # future tense inside a text question
    "What does Apple say about competition from Samsung?",                 # uncovered company as context only
])
def test_text_questions_with_trigger_words_are_not_refused(q):
    assert hard_refusal(q) is None
    assert classify(q).route == "retrieve"


def test_no_company_is_a_soft_refusal_the_planner_may_overrule():
    v = classify("What is the weather like?")
    assert v.route == "refuse" and v.hard  # weather is real_time, hard
    v = classify("What was the revenue growth rate?")
    assert v.route == "refuse" and v.refuse_reason == "no_covered_company" and not v.hard


# ---------------------------------------------------------------- content routes

@pytest.mark.parametrize("q,route", [
    ("What was Apple's revenue growth between its two most recent fiscal years?", "structured"),
    ("How much net income did Oracle report last year?", "structured"),
    ("What was Amazon's net income margin for its latest fiscal year?", "structured"),
    ("What was AWS revenue growth last year?", "retrieve"),                      # segment
    ("What was Microsoft's operating income in fiscal 2025?", "retrieve"),       # not in the facts table
    ("What was Apple's gross margin?", "retrieve"),
    ("What drove Nvidia's revenue growth?", "retrieve"),                         # text question about a metric
    ("What was Nvidia's revenue growth and what drove it?", "hybrid"),
])
def test_content_routes(q, route):
    assert classify(q).route == route


def test_corpus_wide_question_targets_all_eight():
    v = classify("Which company had the highest net income in its latest fiscal year?")
    assert v.route == "structured" and v.tickers == list(TICKERS)


# ---------------------------------------------------------------- known misses (blind probe set, Oct 5)
# Pinned as strict xfails so a change that fixes one is noticed and re-measured on fresh questions
# rather than silently claimed. Do not tune the rules to these: the probe set is now seen.

@pytest.mark.xfail(strict=True, reason="rp_024: total-revenue question with no metric word ('brought in')")
def test_known_miss_metric_without_vocabulary():
    assert classify("If you compare how much Nvidia brought in during its latest fiscal year with the year "
                    "before, what's the percent difference?").route == "structured"


@pytest.mark.xfail(strict=True, reason="rp_045: comparison idiom 'stack up against' not in COMPARE")
def test_known_miss_comparison_idiom():
    assert classify("How does Intel's data center revenue stack up against Nvidia's?").route == "refuse"


@pytest.mark.xfail(strict=True, reason="rp_043: second-sentence instruction ('Include the figures') not a hybrid join")
def test_known_miss_hybrid_in_second_sentence():
    assert classify("Why did Alphabet's net income change in its latest fiscal year? Include the actual net "
                    "income figures for both years.").route == "hybrid"


# ---------------------------------------------------------------- regression on the eval set

@pytest.mark.skipif(not EVAL.exists(), reason="eval set not present")
def test_rules_route_all_fifty_eval_questions_with_exact_tickers():
    for q in json.loads(EVAL.read_text(encoding="utf-8")):
        v = classify(q["question"])
        assert v.route == EVAL_ROUTE[q["gold_answer_type"]], q["id"]
        if v.route != "refuse":
            assert v.tickers == q["ticker"].split(","), q["id"]


# ---------------------------------------------------------------- the evaluator

def _q(i, gold, ambiguous=False, tickers=("AAPL",), reason=None):
    return {"id": f"q{i}", "question": f"question {i}", "gold_route": gold, "gold_tickers": list(tickers),
            "refuse_reason": reason, "difficulty": "plain", "ambiguous": ambiguous}


def test_evaluator_grades_misses_by_cost_and_excludes_ambiguous():
    qs = [_q(1, "structured"), _q(2, "refuse", reason="advice"), _q(3, "retrieve"),
          _q(4, "hybrid"), _q(5, "retrieve", ambiguous=True)]
    answers = {"question 1": "retrieve", "question 2": "refuse", "question 3": "refuse",
               "question 4": "structured", "question 5": "structured"}
    router = lambda text: {"route": answers[text], "tickers": ["AAPL"], "refuse_reason": "advice"}
    res = evaluate(qs, router)
    assert res["route_accuracy"]["n"] == 4 and res["route_accuracy"]["correct"] == 1
    assert res["severity_counts"] == {"critical": 1, "correct": 1, "over_refusal": 1, "partial": 1}
    assert res["n_ambiguous_excluded"] == 1
    assert res["refuse_reason_when_route_correct"] == {"n": 1, "correct": 1}


def test_every_unsafe_answer_to_a_refuse_question_is_critical():
    for pred in ("structured", "retrieve", "hybrid"):
        assert SEVERITY[("refuse", pred)] == "critical"
    assert SEVERITY[("structured", "retrieve")] == "critical"

# ---------------------------------------------------------------- revision 2 (audit, Oct 5)

@pytest.mark.parametrize("q", [
    "How much did Microsoft invest in OpenAI according to its 10-K?",
    "What does Meta say about how much it plans to invest in AI infrastructure?",
    "Did Amazon's cloud arm grow faster than its retail business?",
])
def test_ordinary_filing_questions_are_not_hard_refused(q):
    assert hard_refusal(q) is None and classify(q).route == "retrieve"


@pytest.mark.parametrize("q", ["Should I invest in Nvidia?", "We want to invest in Apple, good idea?"])
def test_investor_advice_is_still_refused(q):
    assert classify(q).refuse_reason == "advice"


def test_named_uncovered_companies_still_vetoed_after_narrowing_the_list():
    assert classify("Compare Apple to Arm Holdings revenue").refuse_reason == "uncovered_company"


def test_sales_and_marketing_is_an_expense_line_not_revenue():
    assert classify("How much did Meta spend on sales and marketing in fiscal 2024?").route == "retrieve"


def test_owned_brands_remain_company_aliases_by_design():
    # "VMware's parent company ... total revenue" is a company-total question; rules cannot separate it from
    # "revenue from VMware" (a segment), so brands are deliberately NOT segment words (left to the planner)
    assert classify("VMware's parent: what was its total revenue last fiscal year?").route == "structured"
