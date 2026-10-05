"""Tests for src/generation/ -- prompt construction, citation/refusal parsing, oracle context,
and the Groq client's caching / retry behaviour (all against fakes; no network)."""
import json

import pytest
import requests

from src.generation.generator import (
    REFUSAL_SENTINEL, RAGGenerator, SYSTEM_PROMPT, build_messages, build_oracle_context,
    format_context, is_refusal, parse_citations,
)
from src.generation.groq_client import GroqClient, GroqError, ResponseCache, build_payload, DEFAULT_MODEL


# ---------------------------------------------------------------- fakes

class FakeResponse:
    def __init__(self, status=200, content="ok", headers=None, text=""):
        self.status_code = status
        self._content = content
        self.headers = headers or {}
        self.text = text

    def json(self):
        return {"choices": [{"message": {"content": self._content}}],
                "usage": {"total_tokens": 7}, "model": "llama-3.3-70b-versatile"}


class FakeSession:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def post(self, url, headers=None, json=None, timeout=None):
        self.calls.append({"url": url, "headers": headers, "json": json})
        item = self.responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


class FakeLLM:
    def __init__(self, text="Apple faces supply risk [1]."):
        self.text, self.calls = text, []

    def chat(self, messages, model, temperature, max_tokens):
        self.calls.append({"messages": messages, "model": model, "temperature": temperature})
        return {"text": self.text, "usage": {}, "model": model, "cached": False}


class FakeRetriever:
    def __init__(self, ids):
        self.ids, self.calls = ids, []

    def rank(self, query, ticker=None, k=10):
        self.calls.append((query, ticker, k))
        return self.ids[:k]


def make_chunks(n=3):
    return {f"c{i}": {"chunk_id": f"c{i}", "text": f"Text of chunk {i}.", "ticker": "AAPL",
                      "form": "10-K", "filing_date": "2024-11-01", "item": "item_1a"} for i in range(1, n + 1)}


# ---------------------------------------------------------------- prompt / parsing

def test_context_is_numbered_with_provenance_headers():
    ctx = format_context([make_chunks()["c1"], make_chunks()["c2"]])
    assert ctx.startswith("[1] AAPL | 10-K | filed 2024-11-01 | item_1a\nText of chunk 1.")
    assert "[2] AAPL | 10-K | filed 2024-11-01 | item_1a\nText of chunk 2." in ctx


def test_messages_contain_rules_passages_and_question():
    msgs = build_messages("What is the risk?", [make_chunks()["c1"]])
    assert msgs[0] == {"role": "system", "content": SYSTEM_PROMPT}
    assert REFUSAL_SENTINEL in SYSTEM_PROMPT
    assert "Question: What is the risk?" in msgs[1]["content"]
    assert "Text of chunk 1." in msgs[1]["content"]


def test_refusal_detection_variants_and_non_refusals():
    for text in ("INSUFFICIENT_CONTEXT", "insufficient_context.", "`INSUFFICIENT_CONTEXT`", "  Insufficient context"):
        assert is_refusal(text), text
    assert not is_refusal("Apple faces supply risk [1].")
    assert not is_refusal("The passages say there is INSUFFICIENT_CONTEXT, but Apple reported X.")


def test_citations_map_to_chunk_ids_and_invalid_ones_are_reported():
    ids = ["c1", "c2", "c3"]
    out = parse_citations("A [1]. B [2][3]. C [1, 3]. D [9].", ids)
    assert out["cited_chunk_ids"] == ["c1", "c2", "c3"]
    assert out["invalid_citations"] == [9]


def test_oracle_context_puts_gold_first_dedupes_and_respects_k():
    assert build_oracle_context(["g1"], ["r1", "g1", "r2", "r3"], k=3) == ["g1", "r1", "r2"]
    assert build_oracle_context(["g1", "g2", "g1"], ["r1"], k=5) == ["g1", "g2", "r1"]
    assert build_oracle_context(["g1", "g2", "g3"], ["r1"], k=2) == ["g1", "g2"]


# ---------------------------------------------------------------- generator

def test_generator_retrieves_with_ticker_and_k_and_builds_result():
    retr, llm = FakeRetriever(["c1", "c2", "c3"]), FakeLLM("Apple faces supply risk [1][2].")
    gen = RAGGenerator(retr, make_chunks(), llm, k=2)
    out = gen.answer("Q?", ticker="AAPL", qid="sh_001")
    assert retr.calls == [("Q?", "AAPL", 2)]
    assert out["context_chunk_ids"] == ["c1", "c2"]
    assert out["cited_chunk_ids"] == ["c1", "c2"]
    assert out["is_refusal"] is False and out["id"] == "sh_001"
    assert llm.calls[0]["temperature"] == 0.0


def test_generator_skips_unknown_chunk_ids_and_honours_explicit_context():
    gen = RAGGenerator(FakeRetriever([]), make_chunks(), FakeLLM(), k=3)
    out = gen.answer("Q?", context_chunk_ids=["nope", "c3", "c1"])
    assert out["context_chunk_ids"] == ["c3", "c1"]


def test_generator_does_not_call_the_llm_when_nothing_was_retrieved():
    llm = FakeLLM()
    out = RAGGenerator(FakeRetriever([]), make_chunks(), llm, k=3).answer("Q?")
    assert llm.calls == [] and out["is_refusal"] is True and out["no_context"] is True


def test_generator_flags_refusals():
    out = RAGGenerator(FakeRetriever(["c1"]), make_chunks(), FakeLLM("INSUFFICIENT_CONTEXT"), k=1).answer("Q?")
    assert out["is_refusal"] is True and out["no_context"] is False


# ---------------------------------------------------------------- groq client

def test_success_parses_text_sends_auth_and_does_not_leak_key_into_cache(tmp_path):
    session = FakeSession([FakeResponse(content="hello")])
    client = GroqClient(api_key="secret-key", cache=ResponseCache(tmp_path), session=session)
    out = client.chat([{"role": "user", "content": "hi"}])
    assert out["text"] == "hello" and out["cached"] is False
    assert session.calls[0]["headers"]["Authorization"] == "Bearer secret-key"
    assert session.calls[0]["json"]["temperature"] == 0.0
    assert "secret-key" not in "".join(p.read_text() for p in tmp_path.glob("*.json"))


def test_second_identical_call_is_a_cache_hit_and_needs_no_key_or_network(tmp_path):
    msgs = [{"role": "user", "content": "hi"}]
    GroqClient(api_key="k", cache=ResponseCache(tmp_path), session=FakeSession([FakeResponse(content="hello")])).chat(msgs)
    offline = GroqClient(api_key=None, cache=ResponseCache(tmp_path), session=FakeSession([]))
    out = offline.chat(msgs)
    assert out["text"] == "hello" and out["cached"] is True


def test_cache_key_changes_with_messages_temperature_and_model(tmp_path):
    base = {"model": "m", "messages": [{"role": "user", "content": "a"}], "temperature": 0.0, "max_tokens": 600}
    keys = {ResponseCache.make_key(base),
            ResponseCache.make_key({**base, "messages": [{"role": "user", "content": "b"}]}),
            ResponseCache.make_key({**base, "temperature": 0.7}),
            ResponseCache.make_key({**base, "model": "m2"})}
    assert len(keys) == 4


def test_missing_key_on_cache_miss_raises_clear_error(monkeypatch, tmp_path):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    with pytest.raises(GroqError, match="GROQ_API_KEY"):
        GroqClient(cache=ResponseCache(tmp_path), session=FakeSession([])).chat([{"role": "user", "content": "x"}])


def test_429_honours_retry_after_then_succeeds():
    sleeps = []
    session = FakeSession([FakeResponse(429, headers={"retry-after": "7"}), FakeResponse(content="ok")])
    out = GroqClient(api_key="k", session=session, sleep_fn=sleeps.append).chat([{"role": "user", "content": "x"}])
    assert out["text"] == "ok" and sleeps == [7.0] and len(session.calls) == 2


def test_5xx_and_network_errors_back_off_exponentially_then_succeed():
    sleeps = []
    session = FakeSession([FakeResponse(503), requests.ConnectionError("boom"), FakeResponse(content="ok")])
    out = GroqClient(api_key="k", session=session, sleep_fn=sleeps.append).chat([{"role": "user", "content": "x"}])
    assert out["text"] == "ok" and sleeps == [1, 2]


def test_client_error_fails_immediately_without_retry():
    session = FakeSession([FakeResponse(401, text="invalid api key")])
    with pytest.raises(GroqError, match="401"):
        GroqClient(api_key="k", session=session, sleep_fn=lambda s: None).chat([{"role": "user", "content": "x"}])
    assert len(session.calls) == 1


def test_retries_exhaust_and_failures_are_not_cached(tmp_path):
    session = FakeSession([FakeResponse(500)] * 3)
    client = GroqClient(api_key="k", cache=ResponseCache(tmp_path), session=session,
                        max_retries=2, sleep_fn=lambda s: None)
    with pytest.raises(GroqError, match="3 attempts"):
        client.chat([{"role": "user", "content": "x"}])
    assert list(tmp_path.glob("*.json")) == []


def test_corrupt_cache_entry_is_a_miss_not_a_crash(tmp_path):
    cache = ResponseCache(tmp_path)
    msgs = [{"role": "user", "content": "x"}]
    payload = build_payload(msgs, DEFAULT_MODEL, 0.0, 1500)
    (tmp_path / f"{ResponseCache.make_key(payload)}.json").write_text("{not json")
    out = GroqClient(api_key="k", cache=cache, session=FakeSession([FakeResponse(content="fresh")])).chat(msgs)
    assert out["text"] == "fresh" and out["cached"] is False


def test_gpt_oss_gets_low_reasoning_effort_and_hidden_reasoning_but_llama_does_not():
    msgs = [{"role": "user", "content": "x"}]
    oss = build_payload(msgs, "openai/gpt-oss-120b", 0.0, 1500)
    assert oss["reasoning_effort"] == "low" and oss["include_reasoning"] is False
    llama = build_payload(msgs, "llama-3.3-70b-versatile", 0.0, 600)
    assert "reasoning_effort" not in llama and "include_reasoning" not in llama


def test_request_actually_sent_carries_the_reasoning_params(tmp_path):
    session = FakeSession([FakeResponse(content="ok")])
    GroqClient(api_key="k", session=session).chat([{"role": "user", "content": "x"}], model="openai/gpt-oss-120b")
    sent = session.calls[0]["json"]
    assert sent["reasoning_effort"] == "low" and sent["max_tokens"] == 1500


def test_empty_answer_from_a_reasoning_model_raises_and_is_never_cached(tmp_path):
    client = GroqClient(api_key="k", cache=ResponseCache(tmp_path),
                        session=FakeSession([FakeResponse(content="")]), sleep_fn=lambda s: None)
    with pytest.raises(GroqError, match="empty answer") as exc:
        client.chat([{"role": "user", "content": "x"}])
    assert exc.value.status_code == 400
    assert list(tmp_path.glob("*.json")) == []


def test_permanent_http_errors_carry_status_code_but_exhausted_retries_do_not():
    with pytest.raises(GroqError) as perm:
        GroqClient(api_key="k", session=FakeSession([FakeResponse(404, text="model_not_found")]),
                   sleep_fn=lambda s: None).chat([{"role": "user", "content": "x"}])
    assert perm.value.status_code == 404
    with pytest.raises(GroqError) as transient:
        GroqClient(api_key="k", session=FakeSession([FakeResponse(500)] * 2), max_retries=1,
                   sleep_fn=lambda s: None).chat([{"role": "user", "content": "x"}])
    assert transient.value.status_code is None


def test_fullwidth_and_annotated_citation_styles_from_gpt_oss_are_parsed():
    ids = ["c1", "c2", "c3", "c4"]
    out = parse_citations("A\u3010 4 \u3011. B\u30102\u2020L10-L12\u3011. C [1][3].", ids)
    assert out["cited_chunk_ids"] == ["c4", "c2", "c1", "c3"] and out["invalid_citations"] == []


def test_prompt_demands_one_short_fact_per_sentence_and_plain_brackets():
    assert "ONE fact" in SYSTEM_PROMPT and "plain square brackets" in SYSTEM_PROMPT
    assert "filing the fact comes from" in SYSTEM_PROMPT


# ---------------------------------------------------------------- prompt versions

def test_default_prompt_is_the_frozen_v1_baseline():
    from src.generation.generator import PROMPTS, SYSTEM_PROMPT, build_messages
    msgs = build_messages("Q?", [{"chunk_id": "c", "text": "t", "ticker": "AAPL", "form": "10-K",
                                  "filing_date": "2024-11-01", "item": "item_1"}])
    assert msgs[0]["content"] == SYSTEM_PROMPT == PROMPTS["v1"]  # unchanged -> cached v1 answers stay valid


def test_v2_prompt_asks_for_exact_dates_and_single_statement_and_keeps_the_refusal_rule():
    from src.generation.generator import PROMPTS, REFUSAL_SENTINEL
    v1, v2 = PROMPTS["v1"], PROMPTS["v2"]
    assert "YYYY-MM-DD" in v2 and "filed 2024-11-01" in v2 and "alone" in v2
    assert "State each distinct fact once" in v2
    assert "separately" not in v2 and "separately" in v1  # the rule that caused the duplication is gone
    assert f"exactly {REFUSAL_SENTINEL}" in v2
    assert "advice" not in v2.lower()  # refusal rule deliberately untouched (held-out contamination)


def test_unknown_prompt_version_is_rejected():
    from src.generation.generator import RAGGenerator, build_messages
    with pytest.raises(ValueError):
        build_messages("Q?", [], "v9")
    with pytest.raises(ValueError):
        RAGGenerator(None, {}, None, prompt_version="v9")


def test_generator_sends_the_selected_prompt_to_the_llm():
    from src.generation.generator import PROMPTS
    chunks = {"c1": {"chunk_id": "c1", "text": "Apple text.", "ticker": "AAPL", "form": "10-K",
                     "filing_date": "2024-11-01", "item": "item_1"}}
    seen = []

    class SpyLLM:
        def chat(self, messages, **kw):
            seen.append(messages[0]["content"])
            return {"text": "Apple 10-K filed 2024-11-01 states a fact [1].", "cached": False, "usage": {}}

    class R:
        def rank(self, q, ticker=None, k=5): return ["c1"]

    for v in ("v1", "v2"):
        RAGGenerator(R(), chunks, SpyLLM(), prompt_version=v).answer("Q?", ticker="AAPL", qid="x")
    assert seen == [PROMPTS["v1"], PROMPTS["v2"]]

# ---------------------------------------------------------------- audit fixes (Oct 5)

from src.generation.generator import is_refusal, parse_citations, refusal_with_text  # noqa: E402


@pytest.mark.parametrize("text,refusal,mixed", [
    ("INSUFFICIENT_CONTEXT", True, False),
    ("**INSUFFICIENT_CONTEXT**", True, False),          # bolded: used to be missed and scored as 0 claims
    ("`INSUFFICIENT_CONTEXT`.", True, False),
    ("insufficient context", True, False),
    ("INSUFFICIENT_CONTEXT. However Apple sold 5 units [1].", False, True),   # content no longer discarded
    ("INSUFFICIENT_CONTEXTUAL data says", False, True),
    ("Apple faces supply risk [1].", False, False),
])
def test_refusal_is_the_sentinel_alone_and_mixed_answers_are_flagged(text, refusal, mixed):
    assert is_refusal(text) is refusal and refusal_with_text(text) is mixed


def test_citation_ranges_expand_and_bad_ranges_do_not_explode():
    ids = ["a", "b", "c", "d", "e"]
    assert parse_citations("x [1-3] y", ids)["cited_chunk_ids"] == ["a", "b", "c"]
    assert parse_citations("x [2\u20134] y", ids)["cited_chunk_ids"] == ["b", "c", "d"]   # en dash
    r = parse_citations("x [1-999] y", ids)            # absurd range: not expanded, ends reported
    assert r["cited_chunk_ids"] == ["a"] and r["invalid_citations"] == [999]


def test_citation_stripping_span_is_unchanged_from_the_day7_regex():
    import re
    from src.generation.generator import CITATION_RE
    old = re.compile(r"[\[\u3010]\s*(\d+(?:\s*[,;]\s*\d+)*)[^\]\u3011]*[\]\u3011]")
    for s in ["a [1-3] b", "c [1, 2] d", "e \u30104\u2020L10-L12\u3011 f", "g [2024 10-K] h", "i [3][4]"]:
        assert CITATION_RE.sub("", s) == old.sub("", s)


def test_malformed_200_body_is_a_groq_error():
    class Bad(FakeResponse):
        def json(self):
            return {"unexpected": True}
    client = GroqClient(api_key="k", session=FakeSession([Bad()]), sleep_fn=lambda s: None)
    with pytest.raises(GroqError, match="malformed"):
        client.chat([{"role": "user", "content": "q"}])


def test_no_sleep_after_the_final_failed_attempt():
    slept = []
    client = GroqClient(api_key="k", session=FakeSession([FakeResponse(status=503)] * 3), max_retries=2,
                        sleep_fn=slept.append)
    with pytest.raises(GroqError):
        client.chat([{"role": "user", "content": "q"}])
    assert len(slept) == 2
