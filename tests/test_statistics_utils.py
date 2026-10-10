import numpy as np

from src import statistics_utils as S


def test_paired_differences():
    a = np.array([0.5, 0.6, 0.7])
    b = np.array([0.4, 0.5, 0.6])
    d = S.paired_differences(a, b)
    assert np.allclose(d, [0.1, 0.1, 0.1])


def test_permutation_test_identical_series_gives_high_p():
    diffs = np.zeros(10)
    p, observed = S.permutation_test(diffs, n_permutations=500, seed=0)
    assert observed == 0.0
    assert p > 0.5


def test_paired_comparison_runs(cfg):
    metric_direct = [0.5, 0.52, 0.49, 0.51, 0.50, 0.53]
    metric_lw = [0.6, 0.58, 0.61, 0.59, 0.62, 0.60]
    result = S.paired_comparison(metric_direct, metric_lw, cfg)
    assert result["n_paired_seeds"] == 6
    assert result["p_value"] is not None
    assert result["mean_paired_difference_layerwise_minus_direct"] > 0
