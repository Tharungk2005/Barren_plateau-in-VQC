"""Single entry point that runs the complete, reproducible experiment
pipeline end to end: dataset retrieval -> preprocessing -> classical
baselines -> main VQC grid -> shots experiment -> depolarizing-noise
experiment -> gradient-variance diagnostic -> circuit diagrams -> tables
-> figures -> result manifest.

Usage (PowerShell, from the project root, with the venv activated):
    python run_project.py --mode quick
    python run_project.py --mode full

Re-running with the same --mode resumes: already-completed (success) runs
recorded in results/<mode>/*.csv are skipped, not re-executed.
"""
import argparse
import os
import time

from src.utils import load_config, project_path, save_json, now_iso, ensure_dir
from src import data_retrieval as dr
from src import preprocessing as pp
from src import experiment_runner as ER
from src import circuits as C
from src import plotting as PLT
from src import export as EXP

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["quick", "full"], default="quick")
    parser.add_argument("--skip-circuit-diagrams", action="store_true")
    args = parser.parse_args()

    cfg = load_config()
    cfg["quick_mode"] = (args.mode == "quick")
    cfg["full_mode"] = (args.mode == "full")

    results_dir = project_path(cfg["paths"]["results_quick"] if cfg["quick_mode"] else cfg["paths"]["results_full"])
    figures_dir = project_path(cfg["paths"]["figures"])
    tables_dir = project_path(cfg["paths"]["tables"])
    ensure_dir(results_dir)
    ensure_dir(figures_dir)
    ensure_dir(tables_dir)

    t_start = time.perf_counter()
    print(f"[{now_iso()}] Mode = {args.mode}. Results will be written to {results_dir}")

    print("Step 1/9: Fetching and validating the real UCI dataset ...")
    df = dr.fetch_raw_dataset(cfg)
    y = pp.encode_labels(df, cfg)
    raw_meta = dr.load_cached_metadata(cfg)
    print(f"  -> {df.shape[0]} instances, {df.shape[1] - 1} features, class counts={raw_meta['class_counts']}")

    print("Step 2/9: Preprocessing (leakage-safe split + PCA angle encoding) for each qubit count ...")
    grid = cfg["experiment_grid"]["quick"] if cfg["quick_mode"] else cfg["experiment_grid"]["full"]
    prepared_by_qubits = {}
    for nq in grid["qubit_counts"]:
        prepared = pp.prepare_dataset_for_qubits(df, y, nq, cfg, seed=0)
        pp.save_pipeline_artifacts(prepared, cfg)
        prepared_by_qubits[nq] = prepared
        print(f"  -> n_qubits={nq}: train={prepared['n_train']} val={prepared['n_val']} test={prepared['n_test']}")

    print("Step 3/9: Classical baselines (Logistic Regression, MLP) ...")
    baselines_csv, baselines_df = ER.run_classical_baselines(cfg, df, y)
    convergence_target = ER.get_convergence_target(
        {"logreg_full": {"val_metrics": {"balanced_accuracy":
            baselines_df[(baselines_df.model_key == "logreg_full")]["val_balanced_accuracy"].mean()}}}
    )
    print(f"  -> convergence target (mean logreg-full val balanced accuracy) = {convergence_target:.4f}")

    print("Step 4/9: Main VQC grid (Direct vs. Layer-wise) ...")
    main_csv = ER.run_main_grid(cfg, df, y, convergence_target)
    main_df = pd.read_csv(main_csv)
    print(f"  -> {len(main_df)} total run rows ({(main_df.status == 'success').sum()} success, {(main_df.status == 'failed').sum()} failed)")

    print("Step 5/9: Sampling-noise (finite-shots) experiment ...")
    shots_csv = ER.run_shots_experiment(cfg, df, y, convergence_target)
    shots_df = pd.read_csv(shots_csv) if os.path.exists(shots_csv) else None

    print("Step 6/9: Simulated depolarizing-noise experiment ...")
    noise_csv = ER.run_noise_experiment(cfg, df, y, convergence_target)
    noise_df = pd.read_csv(noise_csv) if os.path.exists(noise_csv) else None

    print("Step 7/9: Barren-plateau gradient-variance diagnostic ...")
    gd_csv, gd_df = ER.run_gradient_diagnostic_experiment(cfg)

    if not args.skip_circuit_diagrams:
        print("Step 8/9: Circuit diagrams for each unique (qubits, depth) configuration ...")
        unique_configs = sorted(set((nq, d) for nq in grid["qubit_counts"] for d in grid["depths"]))
        for nq, depth in unique_configs:
            fig, ax = C.circuit_drawing_mpl(nq, min(depth, 3) if depth > 6 else depth)
            label_depth = min(depth, 3) if depth > 6 else depth
            suffix = f"_first{label_depth}layers_of_{depth}" if label_depth != depth else ""
            fig.suptitle(f"VQC architecture: {nq} qubits, {label_depth} layer(s) shown{(' of ' + str(depth) + ' total') if label_depth != depth else ''}")
            PLT.savefig(fig, os.path.join(figures_dir, f"circuit_diagram_nq{nq}_d{depth}{suffix}"))
        print(f"  -> saved {len(unique_configs)} circuit diagrams")
    else:
        print("Step 8/9: Skipped (--skip-circuit-diagrams)")

    print("Step 9/9: Building tables, Excel workbook, and figures from saved results ...")
    history_dir = ER.history_dir(cfg)

    prepared_meta_for_table = {nq: {
        "n_train": prepared_by_qubits[nq]["n_train"],
        "n_val": prepared_by_qubits[nq]["n_val"],
        "n_test": prepared_by_qubits[nq]["n_test"],
        "pca_explained_variance_ratio": prepared_by_qubits[nq]["pca_explained_variance_ratio"],
    } for nq in prepared_by_qubits}

    tables = {
        "table1_dataset_summary": EXP.table1_dataset_summary(raw_meta, prepared_meta_for_table),
        "table2_experimental_configs": EXP.table2_experimental_configs(cfg),
        "table3_classical_baselines": EXP.table3_classical_baselines(baselines_df),
        "table4_main_vqc_results": EXP.table4_main_vqc_results(main_df),
        "table5_convergence_trainability": EXP.table5_convergence_trainability(main_df),
        "table7_shots_results": EXP.table7_shots_results(shots_df),
        "table8_noise_results": EXP.table8_noise_results(noise_df),
        "table9_statistical_comparison": EXP.table9_statistical_comparison(main_df, cfg),
        "gradient_diagnostic_summary": gd_df,
        "classical_baselines_raw": baselines_df,
    }
    detail6, summary6 = EXP.table6_seed_level_results(main_df)
    tables["table6_seed_level_detail"] = detail6
    tables["table6_seed_level_summary"] = summary6

    EXP.save_all_tables(tables_dir, tables)
    workbook_path = EXP.save_excel_workbook(tables_dir, tables)
    print(f"  -> Excel workbook: {workbook_path}")

    for nq in grid["qubit_counts"]:
        PLT.plot_val_accuracy_vs_updates(history_dir, main_df, figures_dir, nq, grid["depths"])
        for depth in grid["depths"]:
            PLT.plot_loss_curves(history_dir, main_df, figures_dir, nq, depth, "train_loss", "Training loss (MSE)")
            PLT.plot_loss_curves(history_dir, main_df, figures_dir, nq, depth, "val_loss", "Validation loss (MSE)")
            PLT.plot_gradient_norm_traces(history_dir, main_df, figures_dir, nq, depth)
            for seed in grid["seeds"]:
                PLT.plot_layerwise_stage_performance(history_dir, main_df, figures_dir, nq, depth, seed)

    PLT.plot_gradient_variance_diagnostic(gd_df, figures_dir)
    PLT.plot_success_rate_comparison(main_df, figures_dir)
    PLT.plot_final_metric_comparison(main_df, figures_dir, metric="test_balanced_accuracy")
    PLT.plot_final_metric_comparison(main_df, figures_dir, metric="test_f1")
    PLT.plot_wallclock_comparison(main_df, figures_dir)
    if shots_df is not None:
        PLT.plot_shots_sensitivity(shots_df, figures_dir)
    if noise_df is not None:
        PLT.plot_noise_sensitivity(noise_df, figures_dir)

    manifest = ER.write_manifest(cfg)
    save_json({"mode": args.mode, "convergence_target": convergence_target,
               "total_wall_clock_s": time.perf_counter() - t_start, "completed_at": now_iso()},
              os.path.join(results_dir, "run_summary.json"))

    print(f"[{now_iso()}] DONE in {time.perf_counter() - t_start:.1f}s. Manifest: {manifest['files']}")


if __name__ == "__main__":
    main()
