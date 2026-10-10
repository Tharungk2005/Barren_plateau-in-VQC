import os
import pandas as pd

from src import export as EXP


def _fake_main_df():
    rows = []
    for protocol in ["direct", "layerwise"]:
        for seed in [0, 1, 2]:
            rows.append({
                "n_qubits": 4, "n_layers": 2, "protocol": protocol, "seed": seed,
                "status": "success", "val_accuracy": 0.6, "val_balanced_accuracy": 0.55,
                "test_accuracy": 0.6, "test_balanced_accuracy": 0.58, "test_precision": 0.5,
                "test_recall": 0.5, "test_f1": 0.5, "test_roc_auc": 0.6, "test_loss_mse": 0.9,
                "test_mean_cost": 1.2, "grad_norm_at_init": 0.8, "grad_norm_at_init_normalized": 0.1,
                "plateau_fraction": 0.1, "convergence_update": None, "converged": False,
                "total_wall_clock_s": 2.0, "total_circuit_evaluations": 100,
            })
    return pd.DataFrame(rows)


def test_table4_main_vqc_results_groups_by_protocol():
    df = _fake_main_df()
    t4 = EXP.table4_main_vqc_results(df)
    assert set(t4["protocol"]) == {"direct", "layerwise"}
    assert (t4["n_seeds"] == 3).all()


def test_table9_statistical_comparison_runs_without_error(cfg):
    df = _fake_main_df()
    t9 = EXP.table9_statistical_comparison(df, cfg)
    assert "p_value" in t9.columns
    assert len(t9) > 0


def test_table7_shots_results_includes_analytic_rows():
    import numpy as np
    rows = []
    for shots in [None, 1000, 100]:
        for protocol in ["direct", "layerwise"]:
            rows.append({
                "n_qubits": 4, "n_layers": 2, "protocol": protocol, "seed": 0,
                "shots": shots, "status": "success",
                "test_balanced_accuracy": 0.6, "test_accuracy": 0.6, "test_f1": 0.5,
            })
    df = pd.DataFrame(rows)
    t7 = EXP.table7_shots_results(df)
    assert "analytic" in set(t7["shots"])
    assert len(t7) == 6  # 3 shot settings x 2 protocols, none dropped


def test_save_all_tables_and_excel_workbook(tmp_path):
    df = _fake_main_df()
    tables = {"table4_main_vqc_results": EXP.table4_main_vqc_results(df)}
    EXP.save_all_tables(str(tmp_path), tables)
    assert os.path.exists(tmp_path / "table4_main_vqc_results.csv")
    workbook_path = EXP.save_excel_workbook(str(tmp_path), tables)
    assert os.path.exists(workbook_path)
