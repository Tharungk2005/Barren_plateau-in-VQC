"""Builds the mandatory tables (1-9) from saved run-level CSVs and exports
CSV + a single Excel workbook with named sheets. Nothing here invents a
number: every value is read from a results CSV/JSON produced by
experiment_runner / gradient_diagnostics / baselines.
"""
import os

import numpy as np
import pandas as pd

from src.utils import ensure_dir
from src import statistics_utils as S


def table1_dataset_summary(raw_metadata, prepared_meta_by_qubits):
    rows = []
    for nq, meta in prepared_meta_by_qubits.items():
        rows.append({
            "dataset_source": raw_metadata["source_url"],
            "n_instances": raw_metadata["n_instances"],
            "n_raw_features": raw_metadata["n_features"],
            "class_counts": str(raw_metadata["class_counts"]),
            "n_qubits_representation": nq,
            "n_train": meta["n_train"], "n_val": meta["n_val"], "n_test": meta["n_test"],
            "pca_explained_variance_ratio_sum": round(sum(meta["pca_explained_variance_ratio"]), 4),
            "missing_values": "none (verified during schema validation)",
        })
    return pd.DataFrame(rows)


def table2_experimental_configs(cfg):
    grid = cfg["experiment_grid"]["quick"] if cfg["quick_mode"] else cfg["experiment_grid"]["full"]
    from src import circuits as C
    rows = []
    for nq in grid["qubit_counts"]:
        for depth in grid["depths"]:
            rows.append({
                "n_qubits": nq, "depth": depth,
                "n_trainable_params": C.n_trainable_params(nq, depth),
                "ansatz": "RY+RZ per qubit per layer, CZ ring entangler",
                "optimizer": cfg["training"]["optimizer"], "learning_rate": cfg["training"]["learning_rate"],
                "batch_size": cfg["training"]["batch_size"],
                "seeds": str(grid["seeds"]),
                "total_update_budget": cfg["training"]["quick_mode_updates_per_run"] if cfg["quick_mode"] else cfg["training"]["full_mode_updates_per_run"],
                "init_scheme_direct": str(cfg["circuit"]["layer_init_range_direct"]),
                "init_scheme_layerwise_new_layer": f"uniform(-{cfg['circuit']['layer_init_near_zero_scale']}, {cfg['circuit']['layer_init_near_zero_scale']})",
                "gradient_method": cfg["circuit"]["diff_method_exact"],
                "shots": "analytic (exact)",
                "noise": "noiseless",
            })
    return pd.DataFrame(rows)


def table3_classical_baselines(baselines_df):
    return baselines_df.groupby(["model_name", "feature_representation"]).agg(
        val_accuracy_mean=("val_accuracy", "mean"), val_balanced_accuracy_mean=("val_balanced_accuracy", "mean"),
        test_accuracy_mean=("test_accuracy", "mean"), test_balanced_accuracy_mean=("test_balanced_accuracy", "mean"),
        test_precision_mean=("test_precision", "mean"), test_recall_mean=("test_recall", "mean"),
        test_f1_mean=("test_f1", "mean"), test_roc_auc_mean=("test_roc_auc", "mean"),
        test_mean_cost_mean=("test_mean_cost", "mean"), n_seeds=("seed", "nunique"),
    ).reset_index()


def table4_main_vqc_results(main_df):
    success = main_df[main_df.status == "success"]
    agg = success.groupby(["n_qubits", "n_layers", "protocol"]).agg(
        n_seeds=("seed", "nunique"),
        val_accuracy_mean=("val_accuracy", "mean"), val_accuracy_std=("val_accuracy", "std"),
        val_balanced_accuracy_mean=("val_balanced_accuracy", "mean"), val_balanced_accuracy_std=("val_balanced_accuracy", "std"),
        test_accuracy_mean=("test_accuracy", "mean"), test_accuracy_std=("test_accuracy", "std"),
        test_balanced_accuracy_mean=("test_balanced_accuracy", "mean"), test_balanced_accuracy_std=("test_balanced_accuracy", "std"),
        test_precision_mean=("test_precision", "mean"), test_recall_mean=("test_recall", "mean"),
        test_f1_mean=("test_f1", "mean"), test_f1_std=("test_f1", "std"),
        test_roc_auc_mean=("test_roc_auc", "mean"),
        test_loss_mse_mean=("test_loss_mse", "mean"),
        test_mean_cost_mean=("test_mean_cost", "mean"),
    ).reset_index()
    failed_counts = main_df[main_df.status == "failed"].groupby(["n_qubits", "n_layers", "protocol"]).size().reset_index(name="n_failed")
    agg = agg.merge(failed_counts, on=["n_qubits", "n_layers", "protocol"], how="left")
    agg["n_failed"] = agg["n_failed"].fillna(0).astype(int)
    return agg


def table5_convergence_trainability(main_df):
    success = main_df[main_df.status == "success"]
    agg = success.groupby(["n_qubits", "n_layers", "protocol"]).agg(
        n_seeds=("seed", "nunique"),
        success_rate=("converged", "mean"),
        mean_convergence_update=("convergence_update", "mean"),
        grad_norm_at_init_mean=("grad_norm_at_init", "mean"),
        grad_norm_at_init_normalized_mean=("grad_norm_at_init_normalized", "mean"),
        plateau_fraction_mean=("plateau_fraction", "mean"),
        mean_wall_clock_s=("total_wall_clock_s", "mean"),
        mean_circuit_evaluations=("total_circuit_evaluations", "mean"),
    ).reset_index()
    return agg


def table6_seed_level_results(main_df):
    success = main_df[main_df.status == "success"].copy()
    key_cols = ["n_qubits", "n_layers", "protocol"]
    metric_cols = ["test_balanced_accuracy", "test_f1", "test_roc_auc", "total_wall_clock_s"]
    summary = success.groupby(key_cols)[metric_cols].agg(["mean", "std"]).reset_index()
    summary.columns = [
        "_".join(c for c in col if c) if isinstance(col, tuple) else col
        for col in summary.columns
    ]
    detail = success[key_cols + ["seed"] + metric_cols]
    return detail, summary


def table7_shots_results(shots_df):
    if shots_df is None or len(shots_df) == 0:
        return pd.DataFrame()
    success = shots_df[shots_df.status == "success"].copy()
    # "analytic" (exact expectation value) runs have shots == NaN; pandas
    # groupby drops NaN keys by default, which would silently drop every
    # analytic row from this table, so they are labeled explicitly first.
    success["shots"] = success["shots"].apply(lambda v: "analytic" if pd.isna(v) else int(v))
    agg = success.groupby(["n_qubits", "n_layers", "protocol", "shots"]).agg(
        n_seeds=("seed", "nunique"),
        test_balanced_accuracy_mean=("test_balanced_accuracy", "mean"),
        test_balanced_accuracy_std=("test_balanced_accuracy", "std"),
        test_accuracy_mean=("test_accuracy", "mean"),
        test_f1_mean=("test_f1", "mean"),
    ).reset_index()
    return agg


def table8_noise_results(noise_df):
    if noise_df is None or len(noise_df) == 0:
        return pd.DataFrame()
    success = noise_df[noise_df.status == "success"]
    agg = success.groupby(["n_qubits", "n_layers", "protocol", "noise_prob"]).agg(
        n_seeds=("seed", "nunique"),
        test_balanced_accuracy_mean=("test_balanced_accuracy", "mean"),
        test_balanced_accuracy_std=("test_balanced_accuracy", "std"),
        test_accuracy_mean=("test_accuracy", "mean"),
        test_f1_mean=("test_f1", "mean"),
    ).reset_index()
    return agg


def table9_statistical_comparison(main_df, cfg):
    success = main_df[main_df.status == "success"]
    rows = []
    for (nq, depth), group in success.groupby(["n_qubits", "n_layers"]):
        direct = group[group.protocol == "direct"].set_index("seed")
        lw = group[group.protocol == "layerwise"].set_index("seed")
        common_seeds = sorted(set(direct.index) & set(lw.index))
        if len(common_seeds) == 0:
            continue
        for metric in ["test_balanced_accuracy", "test_f1", "test_roc_auc"]:
            d_vals = direct.loc[common_seeds, metric].values
            l_vals = lw.loc[common_seeds, metric].values
            result = S.paired_comparison(d_vals, l_vals, cfg)
            rows.append({
                "n_qubits": nq, "n_layers": depth, "metric": metric,
                **result,
            })
    return pd.DataFrame(rows)


def save_all_tables(tables_dir, tables: dict):
    ensure_dir(tables_dir)
    for name, df in tables.items():
        if df is None:
            continue
        df.to_csv(os.path.join(tables_dir, f"{name}.csv"), index=False)


def save_excel_workbook(tables_dir, tables: dict, workbook_name="results_summary.xlsx"):
    path = os.path.join(tables_dir, workbook_name)
    ensure_dir(tables_dir)
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        for name, df in tables.items():
            if df is None or len(df) == 0:
                continue
            sheet_name = name[:31]
            df_rounded = df.copy()
            for col in df_rounded.select_dtypes(include=[float]).columns:
                df_rounded[col] = df_rounded[col].round(4)
            df_rounded.to_excel(writer, sheet_name=sheet_name, index=False)
    return path
