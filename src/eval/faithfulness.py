"""
NLI-based faithfulness scoring -- Phase B, Day 7 (Section 5/6 of the plan).

Question answered: does each claim in a generated answer actually follow from the
chunks the generator was shown? Uses `cross-encoder/nli-deberta-v3-base` (a local NLI
model, not an LLM-judge: free, deterministic, and it doesn't share the generator's blind
spots -- Section 5's stated tradeoff).

PIPELINE
    answer text --split--> claim sentences (citations/markdown stripped, refusals dropped)
    context chunks --window--> premise windows (<= ~140 words each)
    every (window, claim) pair --NLI--> P(contradiction), P(entailment), P(neutral)
    claim verdict = best window's entailment probability vs a threshold
    per-answer counts --cluster bootstrap over QUESTIONS--> rates with 95% CIs

DESIGN DECISIONS (and what each one trades away)
  * Premise windows, not whole chunks. Day 1's chunker emits exactly 250-word chunks;
    number-heavy financial text can tokenize to well over 512 DeBERTa tokens once the
    claim is added, and a CrossEncoder silently TRUNCATES. A truncated premise turns a
    true claim into "neutral" -- i.e. an invented hallucination. Windows of <= 140 words
    (sentence-aligned, one-sentence overlap) keep every pair inside the model's limit.
  * Max over windows. A claim counts as supported if ANY window entails it. Cost: a claim
    that needs two different chunks combined (multi-chunk synthesis) is under-credited.
    NLI over concatenated evidence would fix that but reintroduces the length problem.
  * Label order is read from the model's own config, never assumed. Mapping the three
    logits to the wrong labels would flip "supported" and "contradicted" with no error.
  * NLI is weak on numbers ("$383.3B" vs "$393.3B"). It is a faithfulness proxy, not
    ground truth -- which is the interview-defensible reason the agent routes numeric
    questions to SQL/calculator (Day 8). `--sanity-check` includes a numeric probe so
    that weakness is measured on the real model rather than assumed.
  * "Not entailed" != "hallucinated". Neutral can be a genuine NLI miss. Metrics are
    named accordingly (`answer_any_unsupported_rate`, not "hallucination rate").

Needs huggingface.co to load the NLI model -- run from your own venv, not a sandbox.
"""
import re
import sys
import unicodedata
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from src.eval.bootstrap import cluster_mean_bootstrap, cluster_ratio_bootstrap
from src.generation.generator import CITATION_RE, REFUSAL_SENTINEL  # single source of truth for both

NLI_MODEL_NAME = "cross-encoder/nli-deberta-v3-base"
CANONICAL_LABELS = ("contradiction", "entailment", "neutral")

_DASHES = dict.fromkeys(map(ord, "\u2010\u2011\u2012\u2013\u2014\u2015\u2212"), "-")
_QUOTES = {ord("\u2018"): "'", ord("\u2019"): "'", ord("\u201c"): '"', ord("\u201d"): '"'}


def normalize_text(text: str) -> str:
    """Make claim and premise use the same characters before NLI. LLMs emit non-breaking hyphens
    (10\u2011K), en/em dashes and non-breaking spaces; SEC text uses curly quotes and ordinary hyphens. A
    tokenizer can split these differently, so a character-level mismatch could lower an entailment score for
    reasons unrelated to meaning. Applied to BOTH sides, so it cannot create a match that is not there."""
    text = unicodedata.normalize("NFKC", text).translate(_DASHES).translate(_QUOTES)
    return re.sub(r"[ \t\u00a0]+", " ", text).strip()


# --------------------------------------------------------------------------------------
# Sentence / claim extraction
# --------------------------------------------------------------------------------------

# Abbreviations whose trailing period must not end a sentence. Corporate suffixes are
# protected on purpose: wrongly MERGING two sentences keeps all content in the claim,
# whereas wrongly SPLITTING "Alphabet Inc. | Google Cloud grew..." strands the subject.
_ABBREVIATIONS = [
    "U.S.", "U.K.", "E.U.", "e.g.", "i.e.", "vs.", "No.", "Nos.", "Inc.", "Corp.",
    "Co.", "Ltd.", "L.P.", "Mr.", "Ms.", "Dr.", "St.", "approx.", "etc.", "Fig.",
]
_DOT = "․"  # one-dot leader: visually a period, but not matched by the splitter

_BULLET_RE = re.compile(r"^\s*(?:[-*•]|\d+[.)]|#{1,6})\s+")
_SPLIT_RE = re.compile(r'(?<=[.!?])\s+(?=[A-Z0-9"“(\[])')


def _protect(text: str) -> str:
    for abbr in _ABBREVIATIONS:
        text = text.replace(abbr, abbr.replace(".", _DOT))
    return text


def _split_basic(text: str) -> List[str]:
    """Light splitter used for premises (chunk text): newline- and punctuation-based,
    no filtering. Table-ish SEC text often has no sentence punctuation; a long unsplit
    run is handled downstream by the window packer's hard word-split."""
    out: List[str] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        for piece in _SPLIT_RE.split(_protect(line)):
            piece = piece.replace(_DOT, ".").strip()
            if piece:
                out.append(piece)
    return out


def split_sentences(answer: str) -> List[str]:
    """Split a generated answer into sentences: strips markdown bullets/emphasis and
    [n]-style citation markers, splits on sentence punctuation (decimals like 3.5% and
    '$1.2 billion' survive), and cleans spacing left behind by removed citations."""
    sentences: List[str] = []
    for line in answer.splitlines():
        line = _BULLET_RE.sub("", line)
        line = line.replace("**", "").replace("__", "")
        line = CITATION_RE.sub("", line)
        line = re.sub(r"\s+([.,;:!?])", r"\1", line)
        line = re.sub(r"\s{2,}", " ", line).strip()
        if not line:
            continue
        for piece in _SPLIT_RE.split(_protect(line)):
            piece = piece.replace(_DOT, ".").strip()
            if piece:
                sentences.append(piece)
    return sentences


def extract_claims(answer: str, min_words: int = 4) -> List[str]:
    """Sentences worth NLI-scoring: drops refusals and fragments shorter than
    `min_words` (headings, 'Yes.', dangling 'Inc.') that are not checkable claims."""
    claims = []
    for s in split_sentences(answer):
        if REFUSAL_SENTINEL in s.upper().replace(" ", "_"):
            continue
        if len(s.split()) < min_words:
            continue
        claims.append(s)
    return claims


COMPANY_NAMES = {
    "AAPL": "Apple Inc.", "MSFT": "Microsoft Corporation", "GOOGL": "Alphabet Inc.",
    "AMZN": "Amazon.com, Inc.", "META": "Meta Platforms, Inc.", "NVDA": "NVIDIA Corporation",
    "ORCL": "Oracle Corporation", "AVGO": "Broadcom Inc.",
}


def premise_header(chunk: Dict) -> str:
    """Provenance line for a chunk, in natural language: 'Apple Inc. (AAPL) Form 10-K filed
    2025-10-31, Item 1A.'

    The generator was shown every passage under a header carrying ticker, form type and filing
    date, and was told to name the filing each fact comes from. A claim such as 'Apple's 2025
    Form 10-K states ...' therefore asserts provenance as well as content, and the chunk text
    alone cannot entail the provenance half: chunk bodies contain no filing metadata. Scoring
    against the bare text penalised the model for following the prompt (Day 7 follow-up 5).
    The header is real provenance (it comes from the chunk record, not from the model), so a
    claim that names the wrong filing still fails: it contradicts the header. Returns '' when
    the chunk has no provenance fields, so bare-text chunks score exactly as before."""
    ticker, form, date, item = (chunk.get(k) for k in ("ticker", "form", "filing_date", "item"))
    if not (ticker or form or date):
        return ""
    name = COMPANY_NAMES.get(ticker, ticker or "")
    head = f"{name} ({ticker})" if ticker and name != ticker else (name or "")
    parts = [head, f"Form {form}" if form else "", f"filed {date}" if date else ""]
    line = " ".join(p for p in parts if p)
    if item:
        m = re.fullmatch(r"item_(\w+)", str(item))
        line += f", Item {m.group(1).upper()}" if m else f", {item}"
    return line + "."


_ISO_DATE_RE = re.compile(r"\b(\d{4})-(\d{2})-(\d{2})\b")
_MONTHS = {m: i for i, m in enumerate(
    ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"], start=1)}
_TEXT_DATE_RE = re.compile(
    r"\b(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+(\d{1,2})(?:st|nd|rd|th)?,?\s+(\d{4})\b")


def extract_filing_dates(text: str) -> List[str]:
    """Calendar dates named in a claim, as ISO strings ('2024-04-26' and 'April 26, 2024' both
    -> '2024-04-26'). Used to find which filing a claim says it comes from."""
    found = ["-".join(m.groups()) for m in _ISO_DATE_RE.finditer(text)]
    for m in _TEXT_DATE_RE.finditer(text):
        found.append(f"{m.group(3)}-{_MONTHS[m.group(1).lower()]:02d}-{int(m.group(2)):02d}")
    return list(dict.fromkeys(found))


_FRAMING_VERBS = (r"(?:state|note|say|describe|report|indicate|identify|mention|explain|warn|cite|repeat|disclose|"
                  r"highlight|acknowledge|show|confirm|reiterate)s?")
_FRAMING_RE = re.compile(
    r"^(?P<pre>.{0,160}?)\b(?:also |similarly |further |additionally )?" + _FRAMING_VERBS + r"\s+that\s+(?P<core>.+)$",
    re.IGNORECASE | re.DOTALL)
_PROVENANCE_MARKER_RE = re.compile(
    r"(10-?[KQ]|\bForm\b|\bfiling|\bfiled\b|MD&A|risk factors?|\bitem\s+\w+|\bannual\b|\bquarterly\b|\b(?:19|20)\d{2}\b)",
    re.IGNORECASE)


def strip_attribution(claim: str, min_core_words: int = 6) -> str:
    """'Microsoft's 2024-04-25 Form 10-Q risk factors repeat that cyberattacks could X.' ->
    'Cyberattacks could X.'

    The generator is required to name the filing each fact comes from, so every claim opens
    with a provenance wrapper. Provenance is checked deterministically elsewhere (the named
    date selects which filing's windows are scored), so the NLI model only needs to judge the
    content after 'that'. Left in, the wrapper made NLI mark near-verbatim paraphrases as
    neutral. NEGATIVE RESULT (Day 7 follow-up 8): it made scores slightly worse, so it is
    opt-in. For year-only claims the wrapper is what lets NLI pick the right filing, and a stripped
    claim can leave a dangling pronoun. Conservative on purpose: strips only when the preamble
    contains a filing marker, a framing verb is followed by 'that', and at least
    `min_core_words` remain; otherwise the claim is returned unchanged."""
    m = _FRAMING_RE.match(claim.strip())
    if not m or not _PROVENANCE_MARKER_RE.search(m.group("pre")):
        return claim
    core = m.group("core").strip()
    if len(core.split()) < min_core_words:
        return claim
    return core[0].upper() + core[1:]


def build_premise_windows(text: str, max_words: int = 140) -> List[str]:
    """Pack a chunk's sentences into windows of <= max_words words, sentence-aligned,
    carrying the last sentence of each window into the next (so a claim spanning a window
    boundary is seen whole by at least one window). A single sentence longer than
    max_words is hard-split into overlapping word slices."""
    units: List[str] = []
    for s in _split_basic(text):
        words = s.split()
        if len(words) <= max_words:
            units.append(s)
            continue
        step = max(1, int(max_words * 0.75))
        for i in range(0, len(words), step):
            units.append(" ".join(words[i:i + max_words]))
            if i + max_words >= len(words):
                break

    windows: List[str] = []
    cur: List[str] = []
    cur_words = 0
    for u in units:
        uw = len(u.split())
        if cur and cur_words + uw > max_words:
            windows.append(" ".join(cur))
            carry = cur[-1]
            cw = len(carry.split())
            if cw + uw <= max_words:
                cur, cur_words = [carry], cw
            else:
                cur, cur_words = [], 0
        cur.append(u)
        cur_words += uw
    if cur:
        windows.append(" ".join(cur))
    return windows


# --------------------------------------------------------------------------------------
# NLI scorer
# --------------------------------------------------------------------------------------

def resolve_label_order(model) -> List[str]:
    """Return the model's output-column label order from its own config. Falls back to the
    documented order for cross-encoder/nli-* ('contradiction', 'entailment', 'neutral') only
    when the config carries no real label names (e.g. 'LABEL_0'); raises if names are present
    but aren't the three NLI labels, rather than guessing."""
    cfg = getattr(getattr(model, "model", None), "config", None) or getattr(model, "config", None)
    id2label = getattr(cfg, "id2label", None)
    if id2label:
        order = [str(v).lower() for _, v in sorted(((int(k), v) for k, v in id2label.items()))]
        if set(order) == set(CANONICAL_LABELS):
            return order
        if not all(o.startswith("label_") for o in order):
            raise ValueError(f"Unexpected NLI label names in model config: {order}")
    return list(CANONICAL_LABELS)


def _to_probabilities(raw: np.ndarray) -> np.ndarray:
    """CrossEncoder.predict returns logits for multi-class models in the sentence-transformers
    versions we support. If it already returned probabilities (rows in [0,1] summing to ~1),
    don't softmax a second time."""
    raw = np.asarray(raw, dtype=float)
    if raw.ndim == 1:
        raw = raw[None, :]
    looks_like_probs = (
        raw.min() >= 0.0 and raw.max() <= 1.0 and np.allclose(raw.sum(axis=1), 1.0, atol=1e-3)
    )
    if looks_like_probs:
        return raw
    shifted = raw - raw.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)


class NLIScorer:
    """Thin wrapper around a CrossEncoder NLI model. `model` is injectable so tests (and
    anyone with a different NLI model) can pass anything exposing `.predict(pairs, ...)`."""

    def __init__(self, model=None, batch_size: int = 16, max_length: int = 512):
        self._model = model
        self.batch_size = batch_size
        self.max_length = max_length
        self._label_order: Optional[List[str]] = None

    def _get_model(self):
        if self._model is None:
            from sentence_transformers import CrossEncoder
            self._model = CrossEncoder(NLI_MODEL_NAME, max_length=self.max_length)
        return self._model

    @property
    def label_order(self) -> List[str]:
        if self._label_order is None:
            self._label_order = resolve_label_order(self._get_model())
        return self._label_order

    def score_pairs(self, pairs: Sequence[Tuple[str, str]]) -> List[Dict[str, float]]:
        """pairs: (premise, hypothesis). Returns one {label: probability} dict per pair."""
        if not pairs:
            return []
        model = self._get_model()
        raw = model.predict(list(pairs), batch_size=self.batch_size, show_progress_bar=False)
        probs = _to_probabilities(raw)
        order = self.label_order
        return [{label: float(row[i]) for i, label in enumerate(order)} for row in probs]


# --------------------------------------------------------------------------------------
# Per-answer scoring
# --------------------------------------------------------------------------------------

def score_answer(
    answer: str,
    context_chunks: List[Dict],
    scorer: NLIScorer,
    entail_threshold: float = 0.5,
    contradiction_threshold: float = 0.5,
    window_words: int = 140,
    include_header: bool = True,
    scope_to_named_filing: bool = True,
    strip_framing: bool = False,
) -> Dict:
    """
    Score one generated answer against the chunks it was generated from.

    Verdict per claim: 'supported' if the best window's P(entailment) >= entail_threshold;
    else 'contradicted' if some window's P(contradiction) >= contradiction_threshold;
    else 'unsupported' (neutral). 'supported' wins if both hold (one window backs it).
    """
    claims = [normalize_text(c) for c in extract_claims(answer)]
    nli_claims = [strip_attribution(c) if strip_framing else c for c in claims]
    windows: List[Tuple[str, str]] = []  # (chunk_id, window_text)
    chunk_date = {c["chunk_id"]: c.get("filing_date") for c in context_chunks}
    for chunk in context_chunks:
        header = premise_header(chunk) if include_header else ""
        for w in build_premise_windows(chunk.get("text", ""), window_words):
            windows.append((chunk["chunk_id"], normalize_text(f"{header} {w}" if header else w)))

    result = {
        "n_sentences": len(claims), "n_supported": 0, "n_contradicted": 0,
        "n_unsupported": 0, "n_windows": len(windows), "sentences": [],
    }
    if not claims:
        return result

    if windows:
        pairs = [(w, c) for c in nli_claims for _, w in windows]
        probs = scorer.score_pairs(pairs)
    else:
        probs = []

    for ci, claim in enumerate(claims):
        scope, named = "all_context", extract_filing_dates(claim)
        if windows:
            block = probs[ci * len(windows):(ci + 1) * len(windows)]
            idx = list(range(len(windows)))
            if scope_to_named_filing and named:
                # Judge the claim against the filing it says it comes from. Without this, every
                # unsupported claim also 'contradicts' the headers of the sibling filings in
                # context, and a claim stamped with one filing's date could be credited by
                # another filing's text.
                matched = [j for j, (cid, _) in enumerate(windows) if chunk_date.get(cid) in named]
                if matched:
                    idx, scope = matched, "named_filing"
            best_e = max(idx, key=lambda j: block[j]["entailment"])
            max_ent = block[best_e]["entailment"]
            max_con = max(block[j]["contradiction"] for j in idx)
            best_chunk = windows[best_e][0]
        else:  # no context at all -> nothing can support the claim
            max_ent, max_con, best_chunk = 0.0, 0.0, None

        if max_ent >= entail_threshold:
            verdict = "supported"
        elif max_con >= contradiction_threshold:
            verdict = "contradicted"
        else:
            verdict = "unsupported"
        result[f"n_{verdict}"] += 1
        result["sentences"].append({
            "sentence": claim, "scored_as": nli_claims[ci], "verdict": verdict,
            "max_entailment": max_ent, "max_contradiction": max_con,
            "best_matching_chunk_id": best_chunk, "scope": scope,
        })
    return result


def recount_at_threshold(result: Dict, entail_threshold: float, contradiction_threshold: float = 0.5) -> Dict:
    """Recompute a score_answer() result's verdict counts at a different entailment threshold,
    from the per-sentence probabilities it already stored -- no NLI re-run. Lets a report show
    how sensitive the headline number is to the (arbitrary) 0.5 cutoff."""
    sup = con = uns = 0
    for s in result["sentences"]:
        if s["max_entailment"] >= entail_threshold:
            sup += 1
        elif s["max_contradiction"] >= contradiction_threshold:
            con += 1
        else:
            uns += 1
    return {**result, "n_supported": sup, "n_contradicted": con, "n_unsupported": uns}


# --------------------------------------------------------------------------------------
# Aggregation with cluster-bootstrap CIs
# --------------------------------------------------------------------------------------

def aggregate_faithfulness(per_answer: List[Dict], n_resamples: int = 10000, seed: int = 42) -> Dict:
    """
    per_answer: score_answer() outputs, one per QUESTION. Answers with zero scoreable
    sentences (refusals) are excluded here and reported by the caller as refusals.

    Headline = mean_answer_faithfulness (macro: every question weighs the same, matching
    the unit being resampled and how Recall@k/MRR are reported). The pooled sentence rate
    is reported alongside it; long answers weigh more there.
    """
    scored = [a for a in per_answer if a["n_sentences"] > 0]
    n_sent = [a["n_sentences"] for a in scored]
    supported = [a["n_supported"] for a in scored]
    contradicted = [a["n_contradicted"] for a in scored]

    return {
        "n_answers_scored": len(scored),
        "n_sentences_total": int(sum(n_sent)),
        "mean_answer_faithfulness": cluster_mean_bootstrap(
            [s / n for s, n in zip(supported, n_sent)], n_resamples, seed=seed),
        "sentence_support_rate": cluster_ratio_bootstrap(supported, n_sent, n_resamples, seed=seed),
        "sentence_contradiction_rate": cluster_ratio_bootstrap(contradicted, n_sent, n_resamples, seed=seed),
        "answer_any_unsupported_rate": cluster_mean_bootstrap(
            [1.0 if s < n else 0.0 for s, n in zip(supported, n_sent)], n_resamples, seed=seed),
    }


def aggregate_by(per_answer: List[Dict], key_fn: Callable[[Dict], str],
                 n_resamples: int = 10000, seed: int = 42) -> Dict[str, Dict]:
    """aggregate_faithfulness() within groups, e.g. gold-in-context vs not. Small groups
    are returned with their (wide) intervals and `small_n` flags, never hidden."""
    groups: Dict[str, List[Dict]] = {}
    for a in per_answer:
        groups.setdefault(key_fn(a), []).append(a)
    return {g: aggregate_faithfulness(items, n_resamples, seed) for g, items in sorted(groups.items())}


# --------------------------------------------------------------------------------------
# Real-model sanity check (run this once before trusting any number)
# --------------------------------------------------------------------------------------

_SANITY_PREMISE = ("Apple designs, manufactures and markets smartphones, personal computers, "
                   "tablets, wearables and accessories.")
_SANITY_PROBES = [
    # (premise, hypothesis, expected label or None for informational-only, description)
    (_SANITY_PREMISE, "Apple sells smartphones.", "entailment", "clear entailment"),
    (_SANITY_PREMISE, "Apple does not sell any smartphones.", "contradiction", "clear contradiction"),
    (_SANITY_PREMISE, "Apple's headquarters are in Cupertino, California.", "neutral", "unstated fact"),
    ("Total net sales were $383.3 billion in fiscal 2023, a decrease of 3% compared to fiscal 2022.",
     "Net sales increased by 10% in fiscal 2023.", None,
     "numeric contradiction (informational: NLI is known to be weak on numbers)"),
]


def run_sanity_check() -> int:
    """Loads the real NLI model and checks its resolved label order against hand-built probes.
    Exit code 0 only if the three hard probes come out as expected."""
    scorer = NLIScorer()
    print(f"Model: {NLI_MODEL_NAME}\nResolved label order: {scorer.label_order}\n")
    results = scorer.score_pairs([(p, h) for p, h, _, _ in _SANITY_PROBES])
    failures = 0
    for (premise, hyp, expected, desc), probs in zip(_SANITY_PROBES, results):
        got = max(probs, key=probs.get)
        status = "info" if expected is None else ("ok" if got == expected else "FAIL")
        failures += status == "FAIL"
        print(f"[{status:>4}] {desc}\n       hypothesis: {hyp}\n"
              f"       got={got}  expected={expected or '(n/a)'}  "
              f"probs={{{', '.join(f'{k}: {v:.3f}' for k, v in probs.items())}}}")
    print("\nSANITY CHECK PASSED" if not failures else "\nSANITY CHECK FAILED -- do not trust faithfulness numbers")
    return 1 if failures else 0


if __name__ == "__main__":
    if "--sanity-check" in sys.argv:
        sys.exit(run_sanity_check())
    print("Usage: python -m src.eval.faithfulness --sanity-check")
