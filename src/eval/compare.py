"""
Paired comparison of two faithfulness result files (Day 7 ablations: prompt v1 vs v2, retrieved vs oracle).

Why paired: two runs usually answer different numbers of questions (a refusal drops a question
from the faithfulness denominator), so comparing the two headline means mixes "the answers
changed" with "different questions were scored". The comparison here uses only questions scored
in BOTH runs, takes the per-question difference in the fraction of supported claims, and bootstraps
over questions (the same cluster unit as everywhere else in this repo).
"""
from typing import Dict, List

from src.eval.bootstrap import cluster_mean_bootstrap


def per_answer_fraction(answer: Dict) -> float:
    return answer["n_supported"] / answer["n_sentences"]


def paired_difference(a: List[Dict], b: List[Dict], n_resamples: int = 10000, seed: int = 42) -> Dict:
    """a, b: `per_answer_single_hop` lists from two faithfulness output files. Difference = b - a."""
    amap = {r["id"]: r for r in a if r["n_sentences"] > 0}
    bmap = {r["id"]: r for r in b if r["n_sentences"] > 0}
    shared = sorted(set(amap) & set(bmap))
    diffs = [per_answer_fraction(bmap[q]) - per_answer_fraction(amap[q]) for q in shared]
    out = {
        "n_shared": len(shared),
        "only_in_a": sorted(set(amap) - set(bmap)),
        "only_in_b": sorted(set(bmap) - set(amap)),
        "per_question": {q: {"a": per_answer_fraction(amap[q]), "b": per_answer_fraction(bmap[q])} for q in shared},
        "wins_ties_losses_for_b": [sum(d > 1e-9 for d in diffs), sum(abs(d) <= 1e-9 for d in diffs),
                                   sum(d < -1e-9 for d in diffs)],
    }
    out["mean_difference"] = (cluster_mean_bootstrap(diffs, n_resamples, seed=seed) if diffs else None)
    return out


def scoped_fraction(answers: List[Dict]) -> Dict:
    claims = [s for a in answers for s in a["sentences"]]
    scoped = sum(1 for s in claims if s.get("scope") == "named_filing")
    return {"scoped": scoped, "claims": len(claims)}
