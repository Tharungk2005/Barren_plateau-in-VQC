"""Predeclared paired statistical comparison between Direct and Layer-wise
training, matched by seed. Test choice (Wilcoxon signed-rank, with a
permutation-test fallback when the sample is too small / has ties that
make Wilcoxon degenerate) is fixed BEFORE looking at results, per
config.json -> statistics.
"""
import numpy as np
from scipy import stats


def paired_differences(metric_a, metric_b):
    a = np.asarray(metric_a, dtype=float)
    b = np.asarray(metric_b, dtype=float)
    return a - b


def bootstrap_ci_mean(diffs, confidence=0.95, n_boot=10000, seed=0):
    rng = np.random.default_rng(seed)
    n = len(diffs)
    if n == 0:
        return (None, None)
    boots = np.empty(n_boot)
    for i in range(n_boot):
        resample = rng.choice(diffs, size=n, replace=True)
        boots[i] = np.mean(resample)
    lo = np.percentile(boots, (1 - confidence) / 2 * 100)
    hi = np.percentile(boots, (1 + confidence) / 2 * 100)
    return float(lo), float(hi)


def permutation_test(diffs, n_permutations=10000, seed=0):
    """Two-sided paired permutation test on the mean difference, using
    random sign flips (valid under the null of exchangeable paired
    differences).
    """
    rng = np.random.default_rng(seed)
    diffs = np.asarray(diffs, dtype=float)
    observed = np.mean(diffs)
    n = len(diffs)
    if n == 0:
        return None, observed
    count = 0
    for _ in range(n_permutations):
        signs = rng.choice([-1, 1], size=n)
        permuted_mean = np.mean(diffs * signs)
        if abs(permuted_mean) >= abs(observed):
            count += 1
    p_value = (count + 1) / (n_permutations + 1)
    return float(p_value), float(observed)


def paired_comparison(metric_direct, metric_layerwise, cfg):
    """Returns a dict with mean paired difference (layerwise - direct),
    CI, and the predeclared statistical test result. Honestly reports
    when the sample size (number of paired seeds) is too small for a
    reliable Wilcoxon result, falling back to the permutation test.
    """
    diffs = paired_differences(metric_layerwise, metric_direct)
    n = len(diffs)
    mean_diff = float(np.mean(diffs)) if n > 0 else None
    ci_lo, ci_hi = bootstrap_ci_mean(diffs, confidence=cfg["statistics"]["confidence_level"])

    test_used = cfg["statistics"]["paired_test"]
    p_value = None
    stat = None

    if n >= 6 and np.any(diffs != 0):
        try:
            stat, p_value = stats.wilcoxon(diffs)
            test_used = "wilcoxon"
        except ValueError:
            test_used = cfg["statistics"]["fallback_test"]
    else:
        test_used = cfg["statistics"]["fallback_test"]

    if test_used == "permutation" or p_value is None:
        p_value, stat = permutation_test(diffs, n_permutations=cfg["statistics"]["n_permutations"])
        test_used = "permutation"

    return {
        "n_paired_seeds": n,
        "mean_paired_difference_layerwise_minus_direct": mean_diff,
        "ci_lower": ci_lo, "ci_upper": ci_hi,
        "confidence_level": cfg["statistics"]["confidence_level"],
        "test_used": test_used,
        "statistic": float(stat) if stat is not None else None,
        "p_value": float(p_value) if p_value is not None else None,
        "note": "n is the number of paired random seeds; with few seeds, statistical power is limited and a non-significant result must not be read as proof of equivalence.",
    }
