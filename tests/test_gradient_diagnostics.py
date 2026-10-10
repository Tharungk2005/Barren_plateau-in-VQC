import numpy as np

from src import gradient_diagnostics as G


def test_sample_gradients_shapes():
    rng = np.random.default_rng(0)
    comp, norms = G.sample_gradients(4, 2, n_samples=5, rng=rng)
    assert comp.shape == (5,)
    assert norms.shape == (5,)
    assert np.all(np.isfinite(comp))
    assert np.all(norms >= 0)


def test_run_gradient_diagnostic_rows():
    rows = G.run_gradient_diagnostic([4], [1, 2], n_samples=5, seed=0)
    assert len(rows) == 2
    for r in rows:
        assert r["n_samples"] == 5
        assert "gradient_component_variance_ci_lo" in r


def test_fit_log_linear_trend_rejects_nonpositive():
    result = G.fit_log_linear_trend([1, 2, 3], [0.5, 0.0, 0.1])
    assert result["slope"] is None


def test_fit_log_linear_trend_decreasing():
    depths = [1, 2, 3, 4]
    variances = [1.0, 0.5, 0.25, 0.125]  # exact exponential decay
    result = G.fit_log_linear_trend(depths, variances)
    assert result["slope"] < 0
    assert result["r_squared"] > 0.99
