"""Orchestrates the experiment grid with checkpointing/resume, run-level
CSV logging, a result manifest, and safe handling of failed runs.

Resumability: before running any (experiment_type, n_qubits, n_layers,
protocol, seed, shots, noise_prob) combination, the existing run-level
CSV is checked; if a row with matching keys and status == 'success'
already exists, the run is skipped. Failed rows are kept (never deleted)
and are not silently retried as successes.
"""
import os
import csv
import time
import traceback

import numpy as np
import pandas as pd

from src.utils import project_path, ensure_dir, save_json, load_json, now_iso, environment_snapshot
from src import data_retrieval as dr
from src import preprocessing as pp
from src import training as T
from src import metrics as M
from src import baselines as B
from src import gradient_diagnostics as G

MAIN_RUN_FIELDS = [
    "run_id", "experiment_type", "n_qubits", "n_layers", "protocol", "seed",
    "shots", "noise_prob", "n_updates", "n_trainable_params",
    "val_accuracy", "val_balanced_accuracy", "val_loss",
    "test_accuracy", "test_balanced_accuracy", "test_precision", "test_recall",
    "test_f1", "test_roc_auc", "test_loss_mse", "test_mean_cost", "test_total_cost",
    "grad_norm_at_init", "grad_norm_at_init_normalized",
    "plateau_fraction", "convergence_update", "converged",
    "total_circuit_evaluations", "total_wall_clock_s", "diff_method",
    "status", "error_message", "start_time", "end_time", "elapsed_seconds",
]


def _csv_path(cfg, name):
    base = project_path(cfg["paths"]["results_quick"] if cfg["quick_mode"] else cfg["paths"]["results_full"])
    ensure_dir(base)
    return os.path.join(base, name)


def _history_dir(cfg):
    base = project_path(cfg["paths"]["results_quick"] if cfg["quick_mode"] else cfg["paths"]["results_full"])
    return ensure_dir(os.path.join(base, "histories"))


def history_dir(cfg):
    return _history_dir(cfg)


def _load_existing_run_ids(csv_path):
    if not os.path.exists(csv_path):
        return set()
    df = pd.read_csv(csv_path)
    if "run_id" not in df.columns:
        return set()
    successful = df[df["status"] == "success"]["run_id"].tolist()
    return set(successful)


def _append_row(csv_path, row, fieldnames):
    write_header = not os.path.exists(csv_path)
    ensure_dir(os.path.dirname(csv_path))
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def _compute_convergence(history, target, cfg):
    patience = cfg["training"]["convergence_patience_checkpoints"]
    update_idx = M.first_update_reaching_target(history["val_balanced_accuracy"], target, patience=patience)
    converged = update_idx is not None
    return update_idx, converged


def _run_one(n_qubits, n_layers, protocol, seed, data, cfg, convergence_target,
             shots=None, noise_prob=0.0, experiment_type="main", n_updates=None):
    run_id = f"{experiment_type}_nq{n_qubits}_d{n_layers}_{protocol}_seed{seed}_shots{shots}_noise{noise_prob}"
    start = now_iso()
    t0 = time.perf_counter()
    try:
        if protocol == "direct":
            res = T.direct_training_run(n_qubits, n_layers, data, cfg, seed=seed,
                                         shots=shots, noise_prob=noise_prob, n_updates=n_updates)
        elif protocol == "layerwise":
            res = T.layerwise_training_run(n_qubits, n_layers, data, cfg, seed=seed,
                                            shots=shots, noise_prob=noise_prob, n_updates_total=n_updates)
        else:
            raise ValueError(f"Unknown protocol {protocol}")

        test_metrics = T.evaluate_on_split(res, data["X_test_angles"], data["y_test"], cfg)
        plateau = M.plateau_fraction(res["history"]["grad_norm"], cfg["training"]["plateau_grad_norm_threshold"])
        conv_update, converged = _compute_convergence(res["history"], convergence_target, cfg)

        elapsed = time.perf_counter() - t0
        row = {
            "run_id": run_id, "experiment_type": experiment_type,
            "n_qubits": n_qubits, "n_layers": n_layers, "protocol": protocol, "seed": seed,
            "shots": shots, "noise_prob": noise_prob,
            "n_updates": res["n_updates"], "n_trainable_params": res["n_trainable_params"],
            "val_accuracy": res["history"]["val_accuracy"][-1] if res["history"]["val_accuracy"] else None,
            "val_balanced_accuracy": res["history"]["val_balanced_accuracy"][-1] if res["history"]["val_balanced_accuracy"] else None,
            "val_loss": res["history"]["val_loss"][-1] if res["history"]["val_loss"] else None,
            "test_accuracy": test_metrics["accuracy"], "test_balanced_accuracy": test_metrics["balanced_accuracy"],
            "test_precision": test_metrics["precision"], "test_recall": test_metrics["recall"],
            "test_f1": test_metrics["f1"], "test_roc_auc": test_metrics["roc_auc"],
            "test_loss_mse": test_metrics["loss_mse"],
            "test_mean_cost": test_metrics.get("mean_cost"), "test_total_cost": test_metrics.get("total_cost"),
            "grad_norm_at_init": res["grad_norm_at_init"], "grad_norm_at_init_normalized": res["grad_norm_at_init_normalized"],
            "plateau_fraction": plateau, "convergence_update": conv_update, "converged": converged,
            "total_circuit_evaluations": res["total_circuit_evaluations"], "total_wall_clock_s": res["total_wall_clock_s"],
            "diff_method": res["diff_method"],
            "status": "success", "error_message": "",
            "start_time": start, "end_time": now_iso(), "elapsed_seconds": elapsed,
        }

        hist_path = os.path.join(_history_dir(cfg), f"{run_id}.json")
        save_json({
            "history": res["history"],
            "stage_logs": res.get("stage_logs"),
            "final_weights": res["final_weights"],
        }, hist_path)

        return row, True
    except Exception as exc:
        elapsed = time.perf_counter() - t0
        row = {k: None for k in MAIN_RUN_FIELDS}
        row.update({
            "run_id": run_id, "experiment_type": experiment_type,
            "n_qubits": n_qubits, "n_layers": n_layers, "protocol": protocol, "seed": seed,
            "shots": shots, "noise_prob": noise_prob,
            "status": "failed", "error_message": f"{exc}\n{traceback.format_exc()}",
            "start_time": start, "end_time": now_iso(), "elapsed_seconds": elapsed,
        })
        return row, False


def get_convergence_target(baseline_results):
    return baseline_results["logreg_full"]["val_metrics"]["balanced_accuracy"]


def run_main_grid(cfg, df, y, convergence_target, grid_override=None):
    grid = grid_override or (cfg["experiment_grid"]["quick"] if cfg["quick_mode"] else cfg["experiment_grid"]["full"])
    csv_path = _csv_path(cfg, "main_grid_results.csv")
    existing = _load_existing_run_ids(csv_path)

    data_cache = {}
    n_updates = cfg["training"]["quick_mode_updates_per_run"] if cfg["quick_mode"] else cfg["training"]["full_mode_updates_per_run"]

    for n_qubits in grid["qubit_counts"]:
        if n_qubits not in data_cache:
            data_cache[n_qubits] = pp.prepare_dataset_for_qubits(df, y, n_qubits, cfg, seed=0)
        data = data_cache[n_qubits]
        for n_layers in grid["depths"]:
            for protocol in grid["protocols"]:
                for seed in grid["seeds"]:
                    run_id = f"main_nq{n_qubits}_d{n_layers}_{protocol}_seed{seed}_shotsNone_noise0.0"
                    if run_id in existing:
                        continue
                    row, ok = _run_one(n_qubits, n_layers, protocol, seed, data, cfg,
                                        convergence_target, shots=None, noise_prob=0.0,
                                        experiment_type="main", n_updates=n_updates)
                    _append_row(csv_path, row, MAIN_RUN_FIELDS)
    return csv_path


def run_shots_experiment(cfg, df, y, convergence_target):
    sub = cfg["shots_experiment"]["quick_subset"] if cfg["quick_mode"] else cfg["shots_experiment"]["full_subset"]
    settings = cfg["shots_experiment"]["quick_settings"] if cfg["quick_mode"] else cfg["shots_experiment"]["full_settings"]
    csv_path = _csv_path(cfg, "shots_experiment_results.csv")
    existing = _load_existing_run_ids(csv_path)

    n_updates = cfg["training"]["quick_mode_updates_per_run"] if cfg["quick_mode"] else cfg["training"]["full_mode_updates_per_run"]
    data_cache = {}

    for n_qubits in sub["qubit_counts"]:
        if n_qubits not in data_cache:
            data_cache[n_qubits] = pp.prepare_dataset_for_qubits(df, y, n_qubits, cfg, seed=0)
        data = data_cache[n_qubits]
        for n_layers in sub["depths"]:
            for shot_setting in settings:
                shots = None if shot_setting == "analytic" else int(shot_setting)
                for protocol in ["direct", "layerwise"]:
                    for seed in sub["seeds"]:
                        run_id = f"shots_nq{n_qubits}_d{n_layers}_{protocol}_seed{seed}_shots{shots}_noise0.0"
                        if run_id in existing:
                            continue
                        row, ok = _run_one(n_qubits, n_layers, protocol, seed, data, cfg,
                                            convergence_target, shots=shots, noise_prob=0.0,
                                            experiment_type="shots", n_updates=n_updates)
                        _append_row(csv_path, row, MAIN_RUN_FIELDS)
    return csv_path


def run_noise_experiment(cfg, df, y, convergence_target):
    sub = cfg["noise_experiment"]["quick_subset"] if cfg["quick_mode"] else cfg["noise_experiment"]["full_subset"]
    probs = cfg["noise_experiment"]["depolarizing_probabilities"]
    csv_path = _csv_path(cfg, "noise_experiment_results.csv")
    existing = _load_existing_run_ids(csv_path)

    n_updates = cfg["training"]["quick_mode_updates_per_run"] if cfg["quick_mode"] else cfg["training"]["full_mode_updates_per_run"]
    data_cache = {}

    for n_qubits in sub["qubit_counts"]:
        if n_qubits not in data_cache:
            data_cache[n_qubits] = pp.prepare_dataset_for_qubits(df, y, n_qubits, cfg, seed=0)
        data = data_cache[n_qubits]
        for n_layers in sub["depths"]:
            for p in probs:
                for protocol in ["direct", "layerwise"]:
                    for seed in sub["seeds"]:
                        run_id = f"noise_nq{n_qubits}_d{n_layers}_{protocol}_seed{seed}_shotsNone_noise{p}"
                        if run_id in existing:
                            continue
                        row, ok = _run_one(n_qubits, n_layers, protocol, seed, data, cfg,
                                            convergence_target, shots=None, noise_prob=p,
                                            experiment_type="noise", n_updates=n_updates)
                        _append_row(csv_path, row, MAIN_RUN_FIELDS)
    return csv_path


def run_classical_baselines(cfg, df, y):
    csv_path = _csv_path(cfg, "classical_baselines_results.csv")
    seeds = cfg["experiment_grid"]["quick" if cfg["quick_mode"] else "full"]["seeds"]
    rows = []
    for seed in seeds:
        results = B.train_and_evaluate_baselines(df, y, cfg, n_qubits_for_pca=4, seed=seed)
        for key, res in results.items():
            row = {
                "seed": seed, "model_key": key, "model_name": res["model_name"],
                "feature_representation": res["feature_representation"],
                "val_accuracy": res["val_metrics"]["accuracy"],
                "val_balanced_accuracy": res["val_metrics"]["balanced_accuracy"],
                "test_accuracy": res["test_metrics"]["accuracy"],
                "test_balanced_accuracy": res["test_metrics"]["balanced_accuracy"],
                "test_precision": res["test_metrics"]["precision"],
                "test_recall": res["test_metrics"]["recall"],
                "test_f1": res["test_metrics"]["f1"],
                "test_roc_auc": res["test_metrics"]["roc_auc"],
                "test_mean_cost": res["test_metrics"].get("mean_cost"),
            }
            rows.append(row)
    out_df = pd.DataFrame(rows)
    ensure_dir(os.path.dirname(csv_path))
    out_df.to_csv(csv_path, index=False)
    return csv_path, out_df


def run_gradient_diagnostic_experiment(cfg):
    gd = cfg["gradient_diagnostic"]
    n_samples = gd["n_samples_quick"] if cfg["quick_mode"] else gd["n_samples_full"]
    rows = G.run_gradient_diagnostic(gd["qubit_counts"], gd["depths"], n_samples, seed=cfg["random_seed_global"])
    csv_path = _csv_path(cfg, "gradient_diagnostic_results.csv")
    out_df = pd.DataFrame(rows)
    ensure_dir(os.path.dirname(csv_path))
    out_df.to_csv(csv_path, index=False)
    return csv_path, out_df


def write_manifest(cfg):
    base = project_path(cfg["paths"]["results_quick"] if cfg["quick_mode"] else cfg["paths"]["results_full"])
    manifest = {"generated_at": now_iso(), "mode": "quick" if cfg["quick_mode"] else "full",
                "environment": environment_snapshot(), "files": {}}
    for name in ["main_grid_results.csv", "shots_experiment_results.csv",
                 "noise_experiment_results.csv", "classical_baselines_results.csv",
                 "gradient_diagnostic_results.csv"]:
        path = os.path.join(base, name)
        if os.path.exists(path):
            d = pd.read_csv(path)
            info = {"n_rows": len(d)}
            if "status" in d.columns:
                info["n_success"] = int((d["status"] == "success").sum())
                info["n_failed"] = int((d["status"] == "failed").sum())
            manifest["files"][name] = info
        else:
            manifest["files"][name] = {"n_rows": 0, "note": "not generated"}
    save_json(manifest, os.path.join(base, "result_manifest.json"))
    return manifest
