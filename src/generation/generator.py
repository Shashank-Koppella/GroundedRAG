"""
RAG answer generation over retrieved chunks -- Phase B, Day 7 (Groq, `openai/gpt-oss-120b`; Llama 3.3 70B was
the plan but is not available to this API key, see CHANGES.md).

Takes any retriever exposing `.rank(query, ticker=None, k=int) -> List[chunk_id]` (the
same interface every retriever in src/retrieval/ uses), builds a numbered-passage prompt,
and returns the model's answer plus everything needed to score it later.

PROMPT DESIGN (each rule exists for a measured reason)
  * Passages carry ticker / form / filing date / item. Day 6 found the retriever's dominant
    failure is confusing WHICH FILING (cross_filing_confusion) or WHICH SECTION (topic_drift)
    a passage came from -- so the generator is shown that metadata and told to say which
    filing a fact comes from when passages disagree, instead of blending years silently.
  * Every sentence must be self-contained (name the company) and cite passage numbers.
    NLI scores one sentence at a time with no surrounding text, so a pronoun-only sentence
    ("It grew 9%") is unscoreable noise, not a hallucination.
  * Figures are copied, never recomputed -- arithmetic belongs to the Day 8 calculator/SQL
    tool, and NLI cannot verify it anyway.
  * Refusal is a fixed sentinel (INSUFFICIENT_CONTEXT), not free text, so refusals are
    detected deterministically instead of by guessing at phrasing.
  * The refusal rule is deliberately generic (no examples of the out-of-scope categories in
    the eval set) so the out-of-scope refusal rate measures the model's judgment, not a
    prompt written to match the test set.
"""
import re
from typing import Dict, List, Optional

REFUSAL_SENTINEL = "INSUFFICIENT_CONTEXT"

SYSTEM_PROMPT = f"""You answer questions about SEC 10-K and 10-Q filings using ONLY the numbered context passages provided.

Rules:
1. Use only information stated in the passages. Do not use outside knowledge and do not infer beyond what the text says.
2. Write 1 to 5 short sentences. Each sentence states ONE fact in at most 30 words. Do not merge facts from different sentences or different passages into one sentence, and do not add causes or links the passages do not state.
3. Each sentence must stand on its own: name the company (for example "Apple", never "it" or "the company") and the filing the fact comes from (form type and fiscal year or filing date, as shown in the passage header).
4. End every sentence with the number(s) of the supporting passage(s) in plain square brackets, like [1] or [2][3]. Use only [ ] brackets.
5. Copy figures and names exactly as written in the passages. Never round, convert, or calculate new numbers. If passages come from different filings (different years or quarters) and the question does not name one, give the facts for each filing separately.
6. If the passages do not contain what is needed to answer, or the question cannot be answered from the content of SEC filings, reply with exactly {REFUSAL_SENTINEL} and nothing else."""

# PROMPT V2 (Day 7 ablation; v1 above stays frozen as the baseline and is the default, so every cached
# v1 answer and every reported v1 number remains reproducible). Two changes only, each from a measured
# Day 7 failure:
#   * rule 3: cite the filing as "<form> filed YYYY-MM-DD", the exact date from the passage header.
#     v1 let the model write "Microsoft's 2026 Form 10-K"; year-only claims cannot be scored against
#     the filing they name (17/52 retrieved and 21/49 oracle claims), and the same fact scored 1.00 or
#     0.00 depending on citation style (sh_005).
#   * rule 5: state each distinct fact once. v1's "give the facts for each filing separately" made the
#     model copy one sentence onto four 10-Q filings of which only one contains it (sh_004).
# Deliberately NOT changed: the refusal rule. oos_003 (an investment-advice question answered instead of
# refused) is a known miss, but all 10 out-of-scope questions have been seen, so tuning the refusal rule
# against them would contaminate the out-of-scope metric. That needs a fresh held-out set.
SYSTEM_PROMPT_V2 = f"""You answer questions about SEC 10-K and 10-Q filings using ONLY the numbered context passages provided.

Rules:
1. Use only information stated in the passages. Do not use outside knowledge and do not infer beyond what the text says.
2. Write 1 to 5 short sentences. Each sentence states ONE fact in at most 30 words. Do not merge facts from different sentences or different passages into one sentence, and do not add causes or links the passages do not state.
3. Each sentence must stand on its own: name the company (for example "Apple", never "it" or "the company"), the form type, and the filing date written exactly as in the passage header, in the form "Apple 10-K filed 2024-11-01". Always use the full YYYY-MM-DD date. Never refer to a filing by year or quarter alone.
4. End every sentence with the number(s) of the supporting passage(s) in plain square brackets, like [1] or [2][3]. Use only [ ] brackets.
5. Copy figures and names exactly as written in the passages. Never round, convert, or calculate new numbers.
6. State each distinct fact once. If several passages from different filings state the same fact, name only the most recent of those filings and do not repeat the fact for the others. If the filings state different facts, write one sentence per fact, each naming its own filing.
7. If the passages do not contain what is needed to answer, or the question cannot be answered from the content of SEC filings, reply with exactly {REFUSAL_SENTINEL} and nothing else."""

PROMPTS = {"v1": SYSTEM_PROMPT, "v2": SYSTEM_PROMPT_V2}

# Matches [4], [1, 3], [1-3], [1–3] and the fullwidth 【4】 / 【4†L10-L12】 style that gpt-oss models emit despite
# instructions. Group 1 is the passage number(s) incl. ranges; anything after them inside the brackets (e.g. a line
# range after a dagger) is ignored. The WHOLE match span is unchanged from the Oct 4 version (the trailing
# [^\]]* consumed "-3" before), so citation stripping in faithfulness scoring is byte-identical; only parsing improved.
CITATION_RE = re.compile(r"[\[\u3010]\s*(\d+(?:\s*[-\u2013,;]\s*\d+)*)[^\]\u3011]*[\]\u3011]")
MAX_CITATION_RANGE = 20


def format_context(chunks: List[Dict]) -> str:
    """Numbered passages with provenance headers, 1-indexed to match the [n] citations."""
    blocks = []
    for i, c in enumerate(chunks, start=1):
        header = " | ".join(str(x) for x in (
            c.get("ticker", "?"), c.get("form", "?"), f"filed {c.get('filing_date', '?')}", c.get("item", "?")))
        blocks.append(f"[{i}] {header}\n{c.get('text', '').strip()}")
    return "\n\n".join(blocks)


def build_messages(question: str, chunks: List[Dict], prompt_version: str = "v1") -> List[Dict]:
    if prompt_version not in PROMPTS:
        raise ValueError(f"unknown prompt_version {prompt_version!r}; choose from {sorted(PROMPTS)}")
    user = f"Context passages:\n\n{format_context(chunks)}\n\nQuestion: {question}"
    return [{"role": "system", "content": PROMPTS[prompt_version]}, {"role": "user", "content": user}]


_SENTINEL_ONLY_RE = re.compile(r"^" + REFUSAL_SENTINEL + r"[.!]?$")


def _normalize_for_refusal(answer: str) -> str:
    core = answer.strip().strip("`'\"*_.! \t\n")    # markdown emphasis, quotes, final punctuation
    return re.sub(r"\s+", "_", core.upper())


def is_refusal(answer: str) -> bool:
    """True only when the answer IS the sentinel (optionally quoted/bolded, optional final period).
    "INSUFFICIENT_CONTEXT. However Apple sold 5" is NOT a refusal: it carries content, so it is scored
    as an answer and flagged by `refusal_with_text` instead of having its content silently discarded."""
    return bool(_SENTINEL_ONLY_RE.match(_normalize_for_refusal(answer)))


def refusal_with_text(answer: str) -> bool:
    """The sentinel appears, but alongside other content: a malformed answer worth flagging."""
    return REFUSAL_SENTINEL in _normalize_for_refusal(answer) and not is_refusal(answer)


def parse_citations(answer: str, context_chunk_ids: List[str]) -> Dict:
    """Map [n] markers back to chunk ids. Markers outside 1..len(context) are reported, not
    dropped silently -- a citation to a passage that was never shown is itself a finding."""
    cited, invalid = [], []
    for group in CITATION_RE.findall(answer):
        for piece in re.split(r"[,;]", group):
            bounds = [int(x) for x in re.split(r"[-\u2013]", piece.strip()) if x.strip()]
            if len(bounds) == 2 and bounds[0] <= bounds[1] and bounds[1] - bounds[0] <= MAX_CITATION_RANGE:
                nums = list(range(bounds[0], bounds[1] + 1))   # [1-3] -> 1, 2, 3
            else:
                nums = bounds
            for n in nums:
                if 1 <= n <= len(context_chunk_ids):
                    cid = context_chunk_ids[n - 1]
                    if cid not in cited:
                        cited.append(cid)
                elif n not in invalid:
                    invalid.append(n)
    return {"cited_chunk_ids": cited, "invalid_citations": invalid}


def build_oracle_context(gold_ids: List[str], retrieved_ids: List[str], k: int) -> List[str]:
    """Gold chunks first, then retrieved non-gold chunks up to k. This is an optimistic UPPER
    BOUND on generator faithfulness (gold is guaranteed present, and placed first, where models
    attend best) -- it isolates the generator from retrieval failures, the same move as Day 5's
    recall-ceiling diagnostic applied one stage later."""
    ordered = list(dict.fromkeys(gold_ids))
    for cid in retrieved_ids:
        if len(ordered) >= k:
            break
        if cid not in ordered:
            ordered.append(cid)
    return ordered[:k]


class RAGGenerator:
    def __init__(self, retriever, chunks_by_id: Dict[str, Dict], llm, k: int = 5,
                 model: str = "openai/gpt-oss-120b", max_tokens: int = 1500, prompt_version: str = "v1"):
        if prompt_version not in PROMPTS:
            raise ValueError(f"unknown prompt_version {prompt_version!r}; choose from {sorted(PROMPTS)}")
        self.prompt_version = prompt_version
        self.retriever = retriever
        self.chunks_by_id = chunks_by_id
        self.llm = llm
        self.k = k
        self.model = model
        self.max_tokens = max_tokens

    def answer(self, question: str, ticker: Optional[str] = None, qid: Optional[str] = None,
               context_chunk_ids: Optional[List[str]] = None) -> Dict:
        """context_chunk_ids=None -> retrieve k chunks; pass an explicit list to override
        (oracle-context runs)."""
        if context_chunk_ids is None:
            context_chunk_ids = self.retriever.rank(question, ticker=ticker, k=self.k)
        ids = [cid for cid in context_chunk_ids if cid in self.chunks_by_id][: self.k]

        base = {"id": qid, "question": question, "ticker": ticker, "context_chunk_ids": ids}
        if not ids:  # nothing retrieved: don't spend an API call to learn there is no evidence
            return {**base, "answer": REFUSAL_SENTINEL, "is_refusal": True, "no_context": True,
                    "cited_chunk_ids": [], "invalid_citations": [], "model": self.model,
                    "cached": None, "usage": {}}

        messages = build_messages(question, [self.chunks_by_id[c] for c in ids], self.prompt_version)
        resp = self.llm.chat(messages, model=self.model, temperature=0.0, max_tokens=self.max_tokens)
        text = resp["text"].strip()
        return {**base, "answer": text, "is_refusal": is_refusal(text), "refusal_with_text": refusal_with_text(text),
                "no_context": False,
                **parse_citations(text, ids), "model": resp.get("model", self.model),
                "cached": resp.get("cached"), "usage": resp.get("usage", {})}
