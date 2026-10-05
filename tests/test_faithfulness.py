"""Tests for src/eval/faithfulness.py. A fake NLI model stands in for deberta (huggingface.co
is not reachable from the dev shell); the real model is covered by
`python -m src.eval.faithfulness --sanity-check`, which you run in your own venv."""
import math
from types import SimpleNamespace

import numpy as np
import pytest

from src.eval.faithfulness import (
    NLIScorer, aggregate_by, aggregate_faithfulness, build_premise_windows,
    extract_claims, recount_at_threshold, resolve_label_order, score_answer, split_sentences,
    _to_probabilities, normalize_text, premise_header,
    extract_filing_dates, strip_attribution,
)

MAX_WORDS = 140


# ---------------------------------------------------------------- fakes

class FakeNLI:
    """Entails a hypothesis iff it appears (case-insensitive, minus final period) inside the
    premise; contradicts iff the hypothesis is in `contradicts`; else neutral. Emits LOGITS in
    `columns` order and asserts no premise exceeds the window limit (a real CrossEncoder would
    silently truncate it)."""

    def __init__(self, contradicts=(), columns=("contradiction", "entailment", "neutral")):
        self.contradicts = set(contradicts)
        self.columns = list(columns)
        self.calls = 0
        self.pairs_seen = 0
        self.model = SimpleNamespace(config=SimpleNamespace(
            id2label={i: lab for i, lab in enumerate(self.columns)}))

    def predict(self, pairs, **kwargs):
        self.calls += 1
        self.pairs_seen += len(pairs)
        rows = []
        for premise, hyp in pairs:
            assert len(premise.split()) <= MAX_WORDS + 5, "premise exceeds window limit"
            if hyp.rstrip(".").lower() in premise.lower():
                label = "entailment"
            elif hyp in self.contradicts:
                label = "contradiction"
            else:
                label = "neutral"
            logits = np.full(3, -4.0)
            logits[self.columns.index(label)] = 6.0
            rows.append(logits)
        return np.array(rows)


def chunk(cid, text):
    return {"chunk_id": cid, "text": text}


# ---------------------------------------------------------------- sentence splitting

def test_split_strips_citations_and_keeps_numbers_intact():
    out = split_sentences("Apple reported net sales of $383.3 billion in fiscal 2023 [1]. "
                          "Services grew 9.1% year over year [2][3].")
    assert out == ["Apple reported net sales of $383.3 billion in fiscal 2023.",
                   "Services grew 9.1% year over year."]


def test_split_does_not_break_on_abbreviations():
    out = split_sentences("Apple Inc. sells products in the U.S. and Europe. It also licenses software.")
    assert len(out) == 2
    assert out[0].startswith("Apple Inc. sells products in the U.S. and Europe")


def test_split_handles_markdown_bullets_and_emphasis():
    out = split_sentences("- **Apple** faces supply risk [1]\n- Microsoft faces cyber risk too [2]\n1. Nvidia depends on TSMC [1]")
    assert out == ["Apple faces supply risk", "Microsoft faces cyber risk too", "Nvidia depends on TSMC"]


def test_extract_claims_drops_refusals_and_short_fragments():
    assert extract_claims("INSUFFICIENT_CONTEXT") == []
    assert extract_claims("Yes. Apple discloses supply chain concentration risk in Asia.") == [
        "Apple discloses supply chain concentration risk in Asia."]


# ---------------------------------------------------------------- premise windows

def test_windows_respect_limit_cover_every_sentence_and_overlap():
    sents = [f"Sentence number {i} " + " ".join(f"w{i}x{j}" for j in range(21)) + "." for i in range(10)]
    windows = build_premise_windows(" ".join(sents), max_words=MAX_WORDS)
    assert len(windows) > 1
    assert all(len(w.split()) <= MAX_WORDS for w in windows)
    for s in sents:
        assert any(s in w for w in windows)
    assert any(sents[i] in windows[0] and sents[i] in windows[1] for i in range(10))  # one-sentence overlap


def test_one_oversized_sentence_is_hard_split_without_losing_words():
    long_sentence = " ".join(f"tok{i}" for i in range(300)) + "."
    windows = build_premise_windows(long_sentence, max_words=MAX_WORDS)
    assert all(len(w.split()) <= MAX_WORDS for w in windows)
    assert "tok0" in windows[0] and "tok299" in windows[-1].replace(".", "")


def test_empty_text_gives_no_windows():
    assert build_premise_windows("   \n ") == []


# ---------------------------------------------------------------- label order / probabilities

def test_label_order_read_from_config_including_permutations_and_string_keys():
    def m(id2label):
        return SimpleNamespace(model=SimpleNamespace(config=SimpleNamespace(id2label=id2label)))
    assert resolve_label_order(m({0: "contradiction", 1: "entailment", 2: "neutral"})) == \
        ["contradiction", "entailment", "neutral"]
    assert resolve_label_order(m({0: "entailment", 1: "neutral", 2: "contradiction"})) == \
        ["entailment", "neutral", "contradiction"]
    assert resolve_label_order(m({"0": "Entailment", "1": "Neutral", "2": "Contradiction"})) == \
        ["entailment", "neutral", "contradiction"]


def test_label_order_falls_back_only_for_generic_names_and_raises_on_unknown_names():
    def m(id2label):
        return SimpleNamespace(model=SimpleNamespace(config=SimpleNamespace(id2label=id2label)))
    assert resolve_label_order(m({0: "LABEL_0", 1: "LABEL_1", 2: "LABEL_2"})) == \
        ["contradiction", "entailment", "neutral"]
    assert resolve_label_order(SimpleNamespace()) == ["contradiction", "entailment", "neutral"]
    with pytest.raises(ValueError):
        resolve_label_order(m({0: "positive", 1: "negative", 2: "neutral"}))


def test_probabilities_softmax_logits_but_do_not_double_softmax_probabilities():
    p = _to_probabilities(np.array([[2.0, 0.0, -1.0]]))
    assert p.sum() == pytest.approx(1.0) and p.argmax() == 0
    already = np.array([[0.7, 0.2, 0.1]])
    assert np.allclose(_to_probabilities(already), already)
    assert _to_probabilities(np.array([3.0, 1.0, 0.0])).shape == (1, 3)


def test_scorer_maps_columns_using_the_models_own_label_order():
    fake = FakeNLI(columns=("entailment", "neutral", "contradiction"))  # non-default order
    probs = NLIScorer(model=fake).score_pairs([("Apple sells phones", "Apple sells phones.")])[0]
    assert max(probs, key=probs.get) == "entailment"
    assert sum(probs.values()) == pytest.approx(1.0)


# ---------------------------------------------------------------- score_answer

def test_supported_unsupported_and_contradicted_verdicts_and_best_chunk():
    fake = FakeNLI(contradicts={"Zorblax revenue fell sharply in 2023."})
    chunks = [chunk("c1", "Filler about nothing relevant. Other filler sentence here."),
              chunk("c2", "Zorblax revenue was 5 billion in 2023. More filler text follows.")]
    answer = ("Zorblax revenue was 5 billion in 2023 [2]. Zorblax revenue fell sharply in 2023 [2]. "
              "Zorblax employs many people worldwide [1].")
    r = score_answer(answer, chunks, NLIScorer(model=fake))
    verdicts = [s["verdict"] for s in r["sentences"]]
    assert verdicts == ["supported", "contradicted", "unsupported"]
    assert r["sentences"][0]["best_matching_chunk_id"] == "c2"
    assert (r["n_supported"], r["n_contradicted"], r["n_unsupported"]) == (1, 1, 1)


def test_supported_wins_when_one_window_entails_and_another_contradicts():
    claim = "Zorblax revenue was 5 billion in 2023."
    fake = FakeNLI(contradicts={claim})
    chunks = [chunk("c1", "Zorblax revenue was 5 billion in 2023. Extra."), chunk("c2", "Unrelated filler text here.")]
    r = score_answer(claim, chunks, NLIScorer(model=fake))
    assert r["sentences"][0]["verdict"] == "supported"


def test_late_sentence_in_a_full_250_word_chunk_is_found_via_windowing():
    filler = " ".join(f"Filler sentence {i} has several ordinary words in it." for i in range(24))
    target = "Zorblax disclosed a unique concentration risk in its supply chain."
    text = filler + " " + target
    assert len(text.split()) > 200
    fake = FakeNLI()  # asserts no premise > window limit
    r = score_answer(target, [chunk("c1", text)], NLIScorer(model=fake))
    assert r["sentences"][0]["verdict"] == "supported"
    assert r["n_windows"] >= 2


def test_pair_count_is_claims_times_windows():
    fake = FakeNLI()
    chunks = [chunk("a", "Alpha text sentence one is here. Alpha two follows right after."),
              chunk("b", "Beta text sentence is entirely different content.")]
    score_answer("Alpha has one claim written out fully. Beta has another claim written out.", chunks, NLIScorer(model=fake))
    assert fake.pairs_seen == 2 * 2  # 2 claims x 2 windows (one per short chunk)


def test_refusal_and_empty_context_behaviour():
    fake = FakeNLI()
    refusal = score_answer("INSUFFICIENT_CONTEXT", [chunk("c1", "Some text here for context.")], NLIScorer(model=fake))
    assert refusal["n_sentences"] == 0 and fake.calls == 0

    no_ctx = score_answer("Zorblax revenue was 5 billion in 2023.", [], NLIScorer(model=fake))
    assert no_ctx["n_unsupported"] == 1 and no_ctx["n_windows"] == 0 and fake.calls == 0


# ---------------------------------------------------------------- aggregation

def _ans(n, s, c=0):
    return {"n_sentences": n, "n_supported": s, "n_contradicted": c, "n_unsupported": n - s - c}


def test_aggregate_matches_hand_computation_and_excludes_refusals():
    agg = aggregate_faithfulness([_ans(4, 3), _ans(2, 2), _ans(0, 0)], n_resamples=500)
    assert agg["n_answers_scored"] == 2 and agg["n_sentences_total"] == 6
    assert agg["mean_answer_faithfulness"]["estimate"] == pytest.approx((0.75 + 1.0) / 2)
    assert agg["sentence_support_rate"]["estimate"] == pytest.approx(5 / 6)   # ratio of sums, not 0.875
    assert agg["answer_any_unsupported_rate"]["estimate"] == pytest.approx(0.5)
    assert agg["sentence_contradiction_rate"]["estimate"] == pytest.approx(0.0)


def test_aggregate_counts_contradictions():
    agg = aggregate_faithfulness([_ans(4, 2, 1), _ans(4, 4)], n_resamples=200)
    assert agg["sentence_contradiction_rate"]["estimate"] == pytest.approx(1 / 8)


def test_aggregate_with_nothing_scoreable_is_nan_not_a_crash():
    agg = aggregate_faithfulness([_ans(0, 0)], n_resamples=100)
    assert agg["n_answers_scored"] == 0
    assert math.isnan(agg["mean_answer_faithfulness"]["estimate"])


def test_aggregate_by_groups_and_flags_small_n():
    items = [dict(_ans(2, 2), gold=True), dict(_ans(2, 1), gold=False), dict(_ans(2, 0), gold=False)]
    out = aggregate_by(items, lambda a: "gold_in_context" if a["gold"] else "gold_missing", n_resamples=200)
    assert set(out) == {"gold_in_context", "gold_missing"}
    assert out["gold_in_context"]["n_answers_scored"] == 1
    assert out["gold_missing"]["mean_answer_faithfulness"]["estimate"] == pytest.approx(0.25)
    assert out["gold_missing"]["mean_answer_faithfulness"]["small_n"] is True


def test_recount_at_threshold_uses_stored_probabilities_without_rerunning_nli():
    result = {"n_sentences": 3, "n_supported": 2, "n_contradicted": 0, "n_unsupported": 1, "sentences": [
        {"max_entailment": 0.95, "max_contradiction": 0.01},
        {"max_entailment": 0.60, "max_contradiction": 0.20},
        {"max_entailment": 0.10, "max_contradiction": 0.80}]}
    strict = recount_at_threshold(result, 0.9)
    assert (strict["n_supported"], strict["n_contradicted"], strict["n_unsupported"]) == (1, 1, 1)
    lax = recount_at_threshold(result, 0.5)
    assert (lax["n_supported"], lax["n_contradicted"], lax["n_unsupported"]) == (2, 1, 0)


def test_fullwidth_citation_markers_are_stripped_from_claims():
    out = split_sentences("Apple relies on outsourcing partners in China mainland\u30104\u3011. Apple also sources components from Japan\u30102\u2020L1-L3\u3011.")
    assert out == ["Apple relies on outsourcing partners in China mainland.",
                   "Apple also sources components from Japan."]


def test_normalize_text_unifies_dashes_quotes_and_spaces():
    assert normalize_text("Apple 10\u2011K filing\u2014single\u2011source\u00a0partners") == "Apple 10-K filing-single-source partners"
    assert normalize_text("the Company\u2019s \u201cdirect\u201d control") == "the Company's \"direct\" control"


def test_non_breaking_hyphen_in_claim_still_matches_ordinary_hyphen_in_premise():
    fake = FakeNLI()  # entails iff the claim text appears inside the premise text
    chunks = [chunk("c1", "Zorblax filed a 10-K with single-source suppliers. Filler sentence follows here.")]
    r = score_answer("Zorblax filed a 10\u2011K with single\u2011source suppliers.", chunks, NLIScorer(model=fake))
    assert r["sentences"][0]["verdict"] == "supported"


# ---------------------------------------------------------------- premise provenance header

def prov_chunk(cid, text, ticker="AAPL", form="10-K", date="2025-10-31", item="item_1a"):
    return {"chunk_id": cid, "text": text, "ticker": ticker, "form": form,
            "filing_date": date, "item": item}


def test_premise_header_is_natural_language_with_company_name_and_item():
    assert premise_header(prov_chunk("c", "x")) == "Apple Inc. (AAPL) Form 10-K filed 2025-10-31, Item 1A."
    assert premise_header(prov_chunk("c", "x", ticker="MSFT", form="10-Q", date="2025-04-30", item="item_2")) \
        == "Microsoft Corporation (MSFT) Form 10-Q filed 2025-04-30, Item 2."


def test_premise_header_empty_without_provenance_fields():
    assert premise_header(chunk("c1", "bare text")) == ""


def test_header_lets_filing_framed_claim_be_supported_and_ablation_shows_the_artifact():
    body = "Primary exposure relates to non-U.S. dollar-denominated sales and operating expenses."
    claim = "Apple Inc. (AAPL) Form 10-K filed 2025-10-31, Item 1A. " + body
    chunks = [prov_chunk("c1", body)]
    on = score_answer(claim, chunks, NLIScorer(model=FakeNLI()), include_header=True)
    off = score_answer(claim, chunks, NLIScorer(model=FakeNLI()), include_header=False)
    assert on["sentences"][0]["verdict"] == "supported"
    assert off["sentences"][0]["verdict"] != "supported"


def test_header_does_not_rescue_a_claim_naming_the_wrong_filing():
    body = "Primary exposure relates to non-U.S. dollar-denominated sales and operating expenses."
    wrong = "Apple Inc. (AAPL) Form 10-K filed 2022-10-28, Item 1A. " + body
    r = score_answer(wrong, [prov_chunk("c1", body)], NLIScorer(model=FakeNLI()), include_header=True)
    assert r["sentences"][0]["verdict"] != "supported"


def test_header_keeps_bare_text_chunks_scoring_identically():
    text = "Zorblax revenue was 5 billion in 2023. More filler text follows."
    a = score_answer("Zorblax revenue was 5 billion in 2023 [1].", [chunk("c1", text)], NLIScorer(model=FakeNLI()), include_header=True)
    b = score_answer("Zorblax revenue was 5 billion in 2023 [1].", [chunk("c1", text)], NLIScorer(model=FakeNLI()), include_header=False)
    assert a["sentences"] == b["sentences"]


def test_header_respects_the_premise_length_limit():
    long_text = " ".join(f"Sentence number {i} says something about nothing in particular." for i in range(120))
    fake = FakeNLI()  # asserts no premise exceeds MAX_WORDS + 5
    score_answer("Some unrelated claim about the filing that has enough words.", [prov_chunk("c1", long_text)], NLIScorer(model=fake))
    assert fake.pairs_seen > 0


# ---------------------------------------------------------------- scoring against the named filing

class KeywordNLI:
    """P(entail)=high if the premise says 'grew', P(contradict)=high if it says 'fell'."""
    def __init__(self):
        self.model = SimpleNamespace(config=SimpleNamespace(
            id2label={0: "contradiction", 1: "entailment", 2: "neutral"}))

    def predict(self, pairs, **kwargs):
        rows = []
        for premise, _ in pairs:
            lab = 1 if "grew" in premise else 0 if "fell" in premise else 2
            row = np.full(3, -4.0); row[lab] = 6.0
            rows.append(row)
        return np.array(rows)


def two_filings():
    return [prov_chunk("A", "Revenue grew strongly.", date="2024-01-01", form="10-Q"),
            prov_chunk("B", "Revenue fell sharply.", date="2025-01-01", form="10-Q")]


def test_extract_filing_dates_handles_iso_and_written_dates():
    assert extract_filing_dates("filed 2024-04-26 and also April 26 2024.") == ["2024-04-26"]
    assert extract_filing_dates("Alphabet's 10-Q (filed Jul. 26, 2023) says") == ["2023-07-26"]
    assert extract_filing_dates("The 2022 Form 10-K mentions 383.3 billion") == []


def test_claim_is_judged_against_the_filing_it_names_not_a_sibling():
    claim = "Revenue grew strongly according to the filing dated 2025-01-01."
    scoped = score_answer(claim, two_filings(), NLIScorer(model=KeywordNLI()), scope_to_named_filing=True)
    unscoped = score_answer(claim, two_filings(), NLIScorer(model=KeywordNLI()), scope_to_named_filing=False)
    s = scoped["sentences"][0]
    assert s["scope"] == "named_filing" and s["verdict"] == "contradicted" and s["best_matching_chunk_id"] == "B"
    assert unscoped["sentences"][0]["verdict"] == "supported"  # the mis-attribution the scoping exists to catch


def test_correctly_attributed_claim_is_supported_and_sibling_contradiction_is_ignored():
    claim = "Revenue grew strongly according to the filing dated 2024-01-01."
    r = score_answer(claim, two_filings(), NLIScorer(model=KeywordNLI()))
    s = r["sentences"][0]
    assert s["verdict"] == "supported" and s["scope"] == "named_filing" and s["max_contradiction"] < 0.5


def test_unsupported_claim_is_not_called_contradicted_because_of_a_sibling_filing():
    claim = "Margins were stable according to the filing dated 2024-01-01 and its controls."
    scoped = score_answer(claim, two_filings(), NLIScorer(model=KeywordNLI() ))
    s = scoped["sentences"][0]
    assert s["scope"] == "named_filing"
    assert s["verdict"] == "supported"  # chunk A says 'grew' -> keyword fake entails; contradiction from B excluded
    unscoped = score_answer("Margins were stable overall across the company filings.",
                            [prov_chunk("B", "Revenue fell sharply.", date="2025-01-01")], NLIScorer(model=KeywordNLI()))
    assert unscoped["sentences"][0]["verdict"] == "contradicted"  # same-text control: contradiction path still works


def test_falls_back_to_all_context_when_no_date_or_no_matching_chunk():
    no_date = score_answer("Revenue grew strongly in the most recent filing.", two_filings(), NLIScorer(model=KeywordNLI()))
    assert no_date["sentences"][0]["scope"] == "all_context"
    unmatched = score_answer("Revenue grew strongly according to the filing dated 2030-05-05.", two_filings(), NLIScorer(model=KeywordNLI()))
    assert unmatched["sentences"][0]["scope"] == "all_context"


# ---------------------------------------------------------------- attribution wrapper

@pytest.mark.parametrize("claim,core", [
    ("Microsoft's 2024-04-25 Form 10-Q risk factors repeat that cyberattacks and security vulnerabilities could lead to reduced revenue.",
     "Cyberattacks and security vulnerabilities could lead to reduced revenue."),
    ("Nvidia's 2024-05-29 Form 10-Q MD&A also notes that strong sequential Data Center growth was driven by all customer types.",
     "Strong sequential Data Center growth was driven by all customer types."),
    ("Amazon's 2025-10-31 10-Q item 1a states that competition will intensify as retail grows abroad.",
     "Competition will intensify as retail grows abroad."),
    ("Oracle 10-K filed 2025-06-18 notes that competition from other cloud infrastructure providers is listed among the risks.",
     "Competition from other cloud infrastructure providers is listed among the risks."),
])
def test_strip_attribution_removes_the_filing_wrapper(claim, core):
    assert strip_attribution(claim) == core


@pytest.mark.parametrize("claim", [
    "Microsoft Microsoft 10-Q filed 2025-04-30 attributes Intelligent Cloud revenue growth to Azure.",  # no 'that'
    "The company states that sales rose sharply last year across all regions.",                         # no filing marker
    "Apple's 2023 Form 10-K states that growth.",                                                       # core too short
    "Revenue rose 5 percent, and management said that was due to pricing in the Form 10-K.",           # 'that' not after a framing verb
])
def test_strip_attribution_leaves_everything_else_unchanged(claim):
    assert strip_attribution(claim) == claim


def test_scored_as_records_the_core_and_the_ablation_sends_the_full_claim():
    body = "cyberattacks and security vulnerabilities could lead to reduced revenue"
    claim = f"Microsoft's 2024-01-30 Form 10-Q risk factors state that {body}."
    ch = [prov_chunk("c1", f"Intro words. {body.capitalize()}. Outro words.", ticker="MSFT", form="10-Q", date="2024-01-30")]
    on = score_answer(claim, ch, NLIScorer(model=FakeNLI()), strip_framing=True)
    off = score_answer(claim, ch, NLIScorer(model=FakeNLI()), strip_framing=False)
    assert on["sentences"][0]["scored_as"].lower().startswith("cyberattacks")
    assert on["sentences"][0]["verdict"] == "supported"
    assert off["sentences"][0]["scored_as"] == claim and off["sentences"][0]["verdict"] != "supported"
    assert on["sentences"][0]["scope"] == "named_filing"  # scope still comes from the full claim's date


def test_stripping_does_not_credit_content_the_named_filing_lacks():
    claim = "Microsoft's 2025-01-01 Form 10-Q risk factors state that revenue grew strongly across every region."
    r = score_answer(claim, two_filings(), NLIScorer(model=KeywordNLI()), strip_framing=True)  # 2025-01-01 filing says 'fell'
    assert r["sentences"][0]["verdict"] == "contradicted"


def test_framing_is_not_stripped_by_default():
    claim = "Microsoft's 2024-01-30 Form 10-Q risk factors state that revenue grew strongly across every region."
    r = score_answer(claim, two_filings(), NLIScorer(model=KeywordNLI()))
    assert r["sentences"][0]["scored_as"] == claim
