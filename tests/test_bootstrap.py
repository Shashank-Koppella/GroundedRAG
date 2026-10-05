"""Tests for src/eval/bootstrap.py -- the cluster bootstrap behind the faithfulness CIs.

These pin the properties that make the intervals meaningful, not just that the code
runs: ratio-of-sums semantics, clusters (not sentences) as the resampling unit, honest
treatment of undefined resamples, and an empirical coverage check.
"""
import math

import numpy as np
import pytest

from src.eval.bootstrap import cluster_ratio_bootstrap, cluster_mean_bootstrap


def test_ratio_estimate_is_ratio_of_sums_not_mean_of_ratios():
    # per-cluster ratios are 0.5 and 0.9 (mean 0.7), but the pooled rate is 10/12.
    r = cluster_ratio_bootstrap([1, 9], [2, 10], n_resamples=500)
    assert r["estimate"] == pytest.approx(10 / 12)
    assert r["estimate"] != pytest.approx(0.7)


def test_mean_is_ratio_with_unit_denominators():
    vals = [1.0, 0.0, 0.5, 1.0]
    r = cluster_mean_bootstrap(vals, n_resamples=500)
    assert r["estimate"] == pytest.approx(0.625)
    assert r["n_clusters"] == 4


def test_deterministic_for_same_seed_and_different_across_seeds():
    vals = list(np.random.default_rng(0).random(25))
    a = cluster_mean_bootstrap(vals, n_resamples=1000, seed=1)
    b = cluster_mean_bootstrap(vals, n_resamples=1000, seed=1)
    c = cluster_mean_bootstrap(vals, n_resamples=1000, seed=2)
    assert a["ci_95"] == b["ci_95"]
    assert a["ci_95"] != c["ci_95"]


def test_ci_brackets_estimate_and_matches_analytic_width():
    # 100 Bernoulli draws, 50 ones: SE = sqrt(.25/100) = .05 -> 95% CI ~ 0.5 +/- 0.098
    vals = [1.0] * 50 + [0.0] * 50
    r = cluster_mean_bootstrap(vals, n_resamples=10000)
    lo, hi = r["ci_95"]
    assert lo < r["estimate"] < hi
    assert lo == pytest.approx(0.402, abs=0.02)
    assert hi == pytest.approx(0.598, abs=0.02)


def test_resampling_clusters_gives_wider_interval_than_resampling_sentences():
    """The reason this module exists. 20 answers x 8 sentences; within an answer every
    sentence has the same verdict (perfect within-cluster correlation, the extreme of
    'sentences share one context'). Resampling sentences as if independent pretends we
    have 160 observations; the honest unit is 20 answers."""
    n_clusters, per = 20, 8
    supported = [per] * 10 + [0] * 10
    sentences = [per] * n_clusters
    cluster_ci = cluster_ratio_bootstrap(supported, sentences, n_resamples=5000)["ci_95"]

    flat = [1.0] * (10 * per) + [0.0] * (10 * per)  # same data, each sentence its own "cluster"
    naive_ci = cluster_mean_bootstrap(flat, n_resamples=5000)["ci_95"]

    cluster_width = cluster_ci[1] - cluster_ci[0]
    naive_width = naive_ci[1] - naive_ci[0]
    assert cluster_width > 2.0 * naive_width


def test_empirical_coverage_is_close_to_nominal():
    """Simulate many datasets from a known truth and check the 95% interval contains it
    roughly 95% of the time. A resampling bug (wrong unit, off-by-one percentile index,
    double-counting) shows up here as coverage far from 0.95. Seeded -> deterministic."""
    true_p, n_q, n_sims = 0.6, 50, 400
    data_rng = np.random.default_rng(123)
    hits = 0
    for s in range(n_sims):
        vals = (data_rng.random(n_q) < true_p).astype(float)
        lo, hi = cluster_mean_bootstrap(vals, n_resamples=1000, seed=s)["ci_95"]
        hits += (lo <= true_p <= hi)
    coverage = hits / n_sims
    assert 0.90 <= coverage <= 0.99, f"coverage {coverage:.3f} far from nominal 0.95"


def test_empty_input_returns_nan_not_error():
    r = cluster_ratio_bootstrap([], [])
    assert math.isnan(r["estimate"]) and r["n_clusters"] == 0
    assert all(math.isnan(x) for x in r["ci_95"])


def test_single_cluster_collapses_to_point_and_is_flagged_small():
    r = cluster_ratio_bootstrap([3], [4], n_resamples=200)
    assert r["estimate"] == pytest.approx(0.75)
    assert r["ci_95"] == [pytest.approx(0.75), pytest.approx(0.75)]
    assert r["small_n"] is True


def test_zero_denominator_resamples_are_dropped_and_counted_not_zero_filled():
    # cluster 0 has data, cluster 1 has none; ~25% of resamples draw only cluster 1.
    r = cluster_ratio_bootstrap([1, 0], [1, 0], n_resamples=2000)
    assert 0 < r["n_resamples_dropped"] < 2000
    assert r["ci_95"] == [pytest.approx(1.0), pytest.approx(1.0)]  # NOT dragged toward 0


def test_all_zero_denominators_gives_nan_estimate():
    r = cluster_ratio_bootstrap([0, 0], [0, 0], n_resamples=100)
    assert math.isnan(r["estimate"])
    assert all(math.isnan(x) for x in r["ci_95"])


def test_small_n_flag_threshold():
    assert cluster_mean_bootstrap([1.0] * 29, n_resamples=50)["small_n"] is True
    assert cluster_mean_bootstrap([1.0] * 30, n_resamples=50)["small_n"] is False


def test_input_validation():
    with pytest.raises(ValueError):
        cluster_ratio_bootstrap([1, 2], [1])
    with pytest.raises(ValueError):
        cluster_ratio_bootstrap([1], [1], ci=1.5)


# ---------------------------------------------------------------- paired comparison

def _ans(i, sup, n):
    return {"id": i, "n_supported": sup, "n_sentences": n}


def test_paired_difference_uses_only_shared_questions_and_signs_b_minus_a():
    from src.eval.compare import paired_difference
    a = [_ans("q1", 1, 2), _ans("q2", 0, 2), _ans("q3", 2, 2)]
    b = [_ans("q1", 2, 2), _ans("q2", 0, 2), _ans("q4", 1, 1)]
    r = paired_difference(a, b, n_resamples=500)
    assert r["n_shared"] == 2 and r["only_in_a"] == ["q3"] and r["only_in_b"] == ["q4"]
    assert r["mean_difference"]["estimate"] == pytest.approx(0.25)  # (+0.5 and 0.0) / 2
    assert r["wins_ties_losses_for_b"] == [1, 1, 0]


def test_paired_difference_ignores_answers_with_no_claims():
    from src.eval.compare import paired_difference
    r = paired_difference([_ans("q1", 0, 0)], [_ans("q1", 1, 1)], n_resamples=100)
    assert r["n_shared"] == 0 and r["mean_difference"] is None
