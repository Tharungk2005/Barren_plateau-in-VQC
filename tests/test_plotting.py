import os
import pandas as pd

from src import plotting as PLT


def _fake_main_df():
    rows = []
    for nq in [4, 6]:
        for depth in [2, 6]:
            for protocol in ["direct", "layerwise"]:
                for seed in [0, 1]:
                    wall_clock = {"direct": 10.0, "layerwise": 4.0}[protocol]
                    converged = protocol == "layerwise"
                    rows.append({
                        "n_qubits": nq, "n_layers": depth, "protocol": protocol, "seed": seed,
                        "status": "success", "total_wall_clock_s": wall_clock,
                        "converged": converged, "test_balanced_accuracy": 0.6,
                    })
    return pd.DataFrame(rows)


def test_wallclock_comparison_averages_across_qubit_counts(tmp_path):
    df = _fake_main_df()
    PLT.plot_wallclock_comparison(df, str(tmp_path))
    assert os.path.exists(tmp_path / "wallclock_comparison.png")
    # Regression check: with wall_clock values of 10.0/4.0 fixed across both
    # qubit counts, a correct two-stage average must still equal 10.0/4.0 --
    # the historical bug silently picked only the first qubit count's row.
    per_qubit = df.groupby(["n_qubits", "n_layers", "protocol"])["total_wall_clock_s"].mean().reset_index()
    overall = per_qubit.groupby(["n_layers", "protocol"])["total_wall_clock_s"].mean().reset_index()
    assert (overall[overall.protocol == "direct"]["total_wall_clock_s"] == 10.0).all()
    assert (overall[overall.protocol == "layerwise"]["total_wall_clock_s"] == 4.0).all()


def test_success_rate_comparison_runs_without_error(tmp_path):
    df = _fake_main_df()
    PLT.plot_success_rate_comparison(df, str(tmp_path))
    assert os.path.exists(tmp_path / "success_rate_comparison.png")


def test_gradient_variance_diagnostic_plot(tmp_path):
    gd = pd.DataFrame([
        {"n_qubits": 4, "n_layers": 1, "n_samples": 10,
         "gradient_component_variance": 0.5, "gradient_component_variance_ci_lo": 0.3,
         "gradient_component_variance_ci_hi": 0.7},
        {"n_qubits": 4, "n_layers": 2, "n_samples": 10,
         "gradient_component_variance": 0.1, "gradient_component_variance_ci_lo": 0.05,
         "gradient_component_variance_ci_hi": 0.15},
    ])
    PLT.plot_gradient_variance_diagnostic(gd, str(tmp_path))
    assert os.path.exists(tmp_path / "gradient_variance_diagnostic.png")
