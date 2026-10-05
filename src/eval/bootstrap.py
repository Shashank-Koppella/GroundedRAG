"""
Cluster (question-level) bootstrap confidence intervals -- Phase B, Day 7.

WHY THIS EXISTS ALONGSIDE scoring.bootstrap_ci
----------------------------------------------
scoring.bootstrap_ci resamples a flat list of per-question values (one number per
question), which is correct for Recall@k / MRR where each question contributes exactly
one value. Faithfulness is different: each generated answer contains SEVERAL sentences,
each independently scored by the NLI model, and a sentence-level rate such as
"supported sentences / total sentences" is a RATIO of two sums over questions.

Two ways to get that wrong, both of which produce confident-looking but meaningless
intervals (the reason the plan flags this code for extended thinking):

  1. Resampling SENTENCES instead of QUESTIONS. Sentences from the same answer share
     one retrieved context and one generation, so they are strongly correlated.
     Treating them as independent draws makes the effective sample size look like
     ~100 sentences when it is really 20 questions, and the interval comes out far too
     narrow. The independent unit is the question (the "cluster"), so the cluster is
     what gets resampled, with all of its sentences riding along together.

  2. Averaging per-question ratios when the headline is a pooled ratio (or vice
     versa). The pooled rate is sum(supported)/sum(sentences) recomputed on every
     resample -- a "ratio of sums" -- NOT the mean of resampled ratios. Resampling
     (numerator, denominator) PAIRS keeps both sums consistent within a resample.

Both are handled by one primitive, `cluster_ratio_bootstrap`: it resamples cluster
indices with replacement and recomputes sum(num)/sum(den) on each resample. A plain
mean over questions is the special case den == 1 for every question
(`cluster_mean_bootstrap`).

Percentile method, numpy generator seeded for reproducibility. Known limitation, stated
rather than hidden: with ~20 clusters and a rate near 0 or 1, the percentile bootstrap
is optimistic (the interval can collapse to a point). Results carry a `small_n` flag
for n < 30 so reports can say so.
"""
from typing import Dict, Sequence

import numpy as np

SMALL_N_THRESHOLD = 30


def _empty_result(n_resamples: int, ci: float, seed: int) -> Dict:
    return {
        "estimate": float("nan"),
        "ci_95": [float("nan"), float("nan")],
        "ci_level": ci,
        "n_clusters": 0,
        "n_resamples": n_resamples,
        "n_resamples_dropped": 0,
        "small_n": True,
        "seed": seed,
    }


def cluster_ratio_bootstrap(
    numerators: Sequence[float],
    denominators: Sequence[float],
    n_resamples: int = 10000,
    ci: float = 0.95,
    seed: int = 42,
) -> Dict:
    """
    Percentile-bootstrap CI for  sum(numerators) / sum(denominators),  resampling
    CLUSTERS (questions) with replacement. numerators[i] / denominators[i] are the
    cluster-i totals (e.g. supported sentences / total sentences for answer i).

    A resample whose denominators sum to 0 (e.g. it drew only clusters with no
    scoreable sentences) has an undefined ratio; it is excluded from the percentile
    computation and counted in `n_resamples_dropped` rather than silently treated as 0.
    """
    num = np.asarray(numerators, dtype=float)
    den = np.asarray(denominators, dtype=float)
    if num.ndim != 1 or num.shape != den.shape:
        raise ValueError("numerators and denominators must be 1-D and the same length")
    if not 0.0 < ci < 1.0:
        raise ValueError("ci must be in (0, 1)")
    n = len(num)
    if n == 0:
        return _empty_result(n_resamples, ci, seed)

    total_den = den.sum()
    estimate = float(num.sum() / total_den) if total_den > 0 else float("nan")

    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_resamples, n))  # each row = one resample of clusters
    num_s = num[idx].sum(axis=1)
    den_s = den[idx].sum(axis=1)
    valid = den_s > 0
    stats = num_s[valid] / den_s[valid]
    n_dropped = int((~valid).sum())

    if stats.size == 0:
        lo = hi = float("nan")
    else:
        alpha = (1.0 - ci) / 2.0 * 100.0
        lo, hi = (float(x) for x in np.percentile(stats, [alpha, 100.0 - alpha]))

    return {
        "estimate": estimate,
        "ci_95": [lo, hi],
        "ci_level": ci,
        "n_clusters": n,
        "n_resamples": n_resamples,
        "n_resamples_dropped": n_dropped,
        "small_n": n < SMALL_N_THRESHOLD,
        "seed": seed,
    }


def cluster_mean_bootstrap(
    values: Sequence[float],
    n_resamples: int = 10000,
    ci: float = 0.95,
    seed: int = 42,
) -> Dict:
    """Percentile-bootstrap CI for the mean of one value per cluster (question)."""
    values = list(values)
    return cluster_ratio_bootstrap(values, [1.0] * len(values), n_resamples, ci, seed)
