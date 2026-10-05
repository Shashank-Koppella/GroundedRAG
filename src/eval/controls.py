"""
Validity controls for the NLI faithfulness scorer (Day 7, strengthened Oct 5 after an audit).

A faithfulness number is only meaningful if the scorer gives LOW support when the context cannot
support the answer. Three controls, from weakest to strongest:

  other_company   score each answer against a context from a DIFFERENT company. The original Day 7
                  control (0.000). Weak on its own: the provenance header names the wrong company, so
                  the scorer can reject every claim on the company name alone without reading content.
  same_company    score each answer against another question's context from the SAME company (no
                  shared chunks, no gold chunk of this question). The header now matches, so a low
                  score here means the scorer is reading content. Some support is legitimate (same
                  company, overlapping topics), so read it relative to the real run, not against 0.
  header_only     keep each real chunk's provenance header but replace its text with a content-free
                  sentence. Measures how much "support" the header alone produces; should be ~0.

Pairing rules fix an audit finding: the Day 7 pairing compared question tickers, and out-of-scope
answers have ticker None, so a None-ticker answer could be "swapped" onto the same company's chunks.
Pairing now compares the companies actually present in the two contexts, and uses single-hop
answers only.
"""
from typing import Dict, List, Optional

HEADER_ONLY_FILLER = "This passage is part of the filing."
MODES = ("other_company", "same_company")


def context_tickers(context_ids: List[str], chunks_by_id: Dict[str, Dict]) -> set:
    return {chunks_by_id[c]["ticker"] for c in context_ids if c in chunks_by_id}


def live_answers(results: List[Dict]) -> List[Dict]:
    """Answers a control can be built from: single-hop, not refused, with a context."""
    return [r for r in results if r.get("type") == "single_hop" and not r["is_refusal"] and r.get("context_chunk_ids")]


def pair_contexts(results: List[Dict], chunks_by_id: Dict[str, Dict], mode: str,
                  relevant_by_id: Optional[Dict[str, List[str]]] = None) -> Dict[str, List[str]]:
    """answer id -> the context ids it should be scored against under `mode`. Answers with no valid
    partner are left out (and reported by the caller), never paired with an invalid context.
    relevant_by_id: question id -> gold + equivalent chunk ids (src/eval/equivalence.relevant_ids); a
    same-company partner context may contain none of them (an answer-bearing duplicate would inflate the
    control). Defaults to each answer's own gold_chunk_ids."""
    if mode not in MODES:
        raise ValueError(f"unknown negative-control mode {mode!r}; choose from {MODES}")
    live = live_answers(results)
    out: Dict[str, List[str]] = {}
    for i, r in enumerate(live):
        mine = context_tickers(r["context_chunk_ids"], chunks_by_id)
        own_q = {r.get("ticker")} - {None}
        gold = set((relevant_by_id or {}).get(r["id"]) or r.get("gold_chunk_ids") or [])
        for j in range(1, len(live)):
            other = live[(i + j) % len(live)]
            theirs = context_tickers(other["context_chunk_ids"], chunks_by_id)
            if mode == "other_company":
                ok = not (mine | own_q) & theirs
            else:
                ok = (other["id"] != r["id"] and theirs == mine == own_q
                      and not set(other["context_chunk_ids"]) & set(r["context_chunk_ids"])
                      and not gold & set(other["context_chunk_ids"]))
            if ok:
                out[r["id"]] = list(other["context_chunk_ids"])
                break
    return out


def header_only(chunk: Dict) -> Dict:
    """The chunk with its provenance fields intact and its content removed."""
    return {**chunk, "text": HEADER_ONLY_FILLER}


def control_suffix(mode: Optional[str], header_only_flag: bool) -> str:
    s = ""
    if mode == "other_company":
        s += "_negctl"
    elif mode == "same_company":
        s += "_negctl_same"
    if header_only_flag:
        s += "_hdronly"
    return s
