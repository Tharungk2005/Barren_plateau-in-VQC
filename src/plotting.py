"""All figures, regenerated purely from saved CSV/JSON results (never from
hand-typed numbers). Every figure states n (seeds/samples), protocol,
qubit count, depth, and experimental condition in its title or legend.
"""
import os
import json

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.utils import ensure_dir

PROTOCOL_COLORS = {"direct": "#d62728", "layerwise": "#1f77b4"}
PROTOCOL_LABELS = {"direct": "Direct Training", "layerwise": "Layer-wise Training"}


def savefig(fig, path_no_ext, dpi=150):
    return _savefig(fig, path_no_ext, dpi=dpi)


def _savefig(fig, path_no_ext, dpi=150):
    ensure_dir(os.path.dirname(path_no_ext))
    fig.savefig(path_no_ext + ".png", dpi=dpi, bbox_inches="tight")
    try:
        fig.savefig(path_no_ext + ".pdf", bbox_inches="tight")
    except Exception:
        pass
    plt.close(fig)


def plot_val_accuracy_vs_updates(history_dir, main_df, figures_dir, n_qubits, depths):
    for depth in depths:
        fig, ax = plt.subplots(figsize=(7, 5))
        n_seeds_used = {}
        for protocol in ["direct", "layerwise"]:
            rows = main_df[(main_df.n_qubits == n_qubits) & (main_df.n_layers == depth) &
                            (main_df.protocol == protocol) & (main_df.status == "success")]
            if len(rows) == 0:
                continue
            curves = []
            for _, r in rows.iterrows():
                hist_path = os.path.join(history_dir, f"{r['run_id']}.json")
                if not os.path.exists(hist_path):
                    continue
                with open(hist_path) as f:
                    h = json.load(f)["history"]
                curves.append((h["update"], h["val_accuracy"]))
            if not curves:
                continue
            n_seeds_used[protocol] = len(curves)
            common_len = min(len(c[1]) for c in curves)
            updates = curves[0][0][:common_len]
            vals = np.array([c[1][:common_len] for c in curves])
            mean_v = vals.mean(axis=0)
            std_v = vals.std(axis=0)
            ax.plot(updates, mean_v, label=f"{PROTOCOL_LABELS[protocol]} (n={len(curves)} seeds)",
                    color=PROTOCOL_COLORS[protocol])
            ax.fill_between(updates, mean_v - std_v, mean_v + std_v, alpha=0.2, color=PROTOCOL_COLORS[protocol])
        ax.set_xlabel("Optimizer update")
        ax.set_ylabel("Validation accuracy")
        ax.set_title(f"Validation accuracy vs. updates (n_qubits={n_qubits}, depth={depth})")
        ax.legend()
        ax.grid(alpha=0.3)
        _savefig(fig, os.path.join(figures_dir, f"val_accuracy_vs_updates_nq{n_qubits}_d{depth}"))


def plot_loss_curves(history_dir, main_df, figures_dir, n_qubits, depth, loss_key, title_prefix):
    fig, ax = plt.subplots(figsize=(7, 5))
    for protocol in ["direct", "layerwise"]:
        rows = main_df[(main_df.n_qubits == n_qubits) & (main_df.n_layers == depth) &
                        (main_df.protocol == protocol) & (main_df.status == "success")]
        curves = []
        for _, r in rows.iterrows():
            hist_path = os.path.join(history_dir, f"{r['run_id']}.json")
            if not os.path.exists(hist_path):
                continue
            with open(hist_path) as f:
                h = json.load(f)["history"]
            curves.append((h["update"], h[loss_key]))
        if not curves:
            continue
        common_len = min(len(c[1]) for c in curves)
        updates = curves[0][0][:common_len]
        vals = np.array([c[1][:common_len] for c in curves])
        mean_v, std_v = vals.mean(axis=0), vals.std(axis=0)
        ax.plot(updates, mean_v, label=f"{PROTOCOL_LABELS[protocol]} (n={len(curves)} seeds)",
                color=PROTOCOL_COLORS[protocol])
        ax.fill_between(updates, mean_v - std_v, mean_v + std_v, alpha=0.2, color=PROTOCOL_COLORS[protocol])
    ax.set_xlabel("Optimizer update")
    ax.set_ylabel(title_prefix)
    ax.set_title(f"{title_prefix} (n_qubits={n_qubits}, depth={depth})")
    ax.legend()
    ax.grid(alpha=0.3)
    _savefig(fig, os.path.join(figures_dir, f"{loss_key}_nq{n_qubits}_d{depth}"))


def plot_gradient_norm_traces(history_dir, main_df, figures_dir, n_qubits, depth):
    plot_loss_curves(history_dir, main_df, figures_dir, n_qubits, depth, "grad_norm", "Gradient norm")


def plot_gradient_variance_diagnostic(gd_df, figures_dir):
    fig, ax = plt.subplots(figsize=(7, 5))
    for nq in sorted(gd_df.n_qubits.unique()):
        sub = gd_df[gd_df.n_qubits == nq].sort_values("n_layers")
        ax.errorbar(sub.n_layers, sub.gradient_component_variance,
                    yerr=[sub.gradient_component_variance - sub.gradient_component_variance_ci_lo,
                          sub.gradient_component_variance_ci_hi - sub.gradient_component_variance],
                    marker="o", label=f"{nq} qubits")
    ax.set_yscale("log")
    ax.set_xlabel("Circuit depth (layers)")
    ax.set_ylabel("Gradient component variance (log scale)")
    n_samples = gd_df["n_samples"].iloc[0] if len(gd_df) else "?"
    ax.set_title(f"Gradient-variance diagnostic vs. depth (n_samples={n_samples} per point, 95% bootstrap CI)")
    ax.legend()
    ax.grid(alpha=0.3, which="both")
    _savefig(fig, os.path.join(figures_dir, "gradient_variance_diagnostic"))


def plot_layerwise_stage_performance(history_dir, main_df, figures_dir, n_qubits, depth, seed):
    row = main_df[(main_df.n_qubits == n_qubits) & (main_df.n_layers == depth) &
                   (main_df.protocol == "layerwise") & (main_df.seed == seed) &
                   (main_df.status == "success")]
    if len(row) == 0:
        return
    run_id = row.iloc[0]["run_id"]
    hist_path = os.path.join(history_dir, f"{run_id}.json")
    if not os.path.exists(hist_path):
        return
    with open(hist_path) as f:
        payload = json.load(f)
    stage_logs = payload.get("stage_logs") or []
    if not stage_logs:
        return
    layers = [s["layer_idx"] + 1 for s in stage_logs]
    after = [s["loss_after_stage"] for s in stage_logs]
    after_insertion = [s.get("loss_immediately_after_insertion") for s in stage_logs]

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(layers, after, marker="o", label="Validation loss after stage (trained)", color="#1f77b4")
    valid_ins = [(l, b) for l, b in zip(layers, after_insertion) if b is not None]
    if valid_ins:
        li, bi = zip(*valid_ins)
        ax.scatter(li, bi, marker="x", color="#d62728",
                   label="Validation loss immediately after layer insertion\n(new layer at near-zero angles, before training)")
    ax.set_xlabel("Number of layers active")
    ax.set_ylabel("Validation MSE loss")
    ax.set_title(f"Layer-wise stage performance (n_qubits={n_qubits}, target depth={depth}, seed={seed})")
    ax.legend()
    ax.grid(alpha=0.3)
    _savefig(fig, os.path.join(figures_dir, f"layerwise_stage_performance_nq{n_qubits}_d{depth}_seed{seed}"))


def plot_success_rate_comparison(main_df, figures_dir):
    per_qubit = main_df[main_df.status == "success"].groupby(["n_qubits", "n_layers", "protocol"])["converged"].mean().reset_index()
    agg = per_qubit.groupby(["n_layers", "protocol"])["converged"].mean().reset_index()
    fig, ax = plt.subplots(figsize=(8, 5))
    depths = sorted(agg.n_layers.unique())
    width = 0.35
    for i, protocol in enumerate(["direct", "layerwise"]):
        sub = agg[agg.protocol == protocol]
        xs = np.arange(len(depths)) + (i - 0.5) * width
        ys = [sub[sub.n_layers == d]["converged"].values[0] if d in sub.n_layers.values else 0 for d in depths]
        ax.bar(xs, ys, width=width, label=PROTOCOL_LABELS[protocol], color=PROTOCOL_COLORS[protocol])
    ax.set_xticks(np.arange(len(depths)))
    ax.set_xticklabels(depths)
    ax.set_xlabel("Circuit depth (layers)")
    ax.set_ylabel("Success rate (fraction of seeds reaching convergence target)")
    ax.set_title("Success rate comparison across paired seeds, by depth")
    ax.legend()
    ax.grid(alpha=0.3, axis="y")
    _savefig(fig, os.path.join(figures_dir, "success_rate_comparison"))


def plot_final_metric_comparison(main_df, figures_dir, metric="test_balanced_accuracy"):
    agg = main_df[main_df.status == "success"].groupby(["n_qubits", "n_layers", "protocol"])[metric].agg(["mean", "std", "count"]).reset_index()
    fig, ax = plt.subplots(figsize=(9, 5))
    depths = sorted(agg.n_layers.unique())
    qubit_counts = sorted(agg.n_qubits.unique())
    width = 0.35
    for qi, nq in enumerate(qubit_counts):
        for i, protocol in enumerate(["direct", "layerwise"]):
            sub = agg[(agg.protocol == protocol) & (agg.n_qubits == nq)]
            xs = np.arange(len(depths)) + (i - 0.5) * width + qi * len(depths) * 1.3
            means = [sub[sub.n_layers == d]["mean"].values[0] if d in sub.n_layers.values else 0 for d in depths]
            stds = [sub[sub.n_layers == d]["std"].values[0] if d in sub.n_layers.values else 0 for d in depths]
            ax.bar(xs, means, yerr=stds, width=width, capsize=3,
                   label=f"{PROTOCOL_LABELS[protocol]}" if qi == 0 else None,
                   color=PROTOCOL_COLORS[protocol])
    ax.set_ylabel(metric.replace("_", " "))
    ax.set_title(f"Final {metric.replace('_', ' ')} comparison (mean +/- std across seeds)")
    ax.legend()
    ax.grid(alpha=0.3, axis="y")
    _savefig(fig, os.path.join(figures_dir, f"final_metric_comparison_{metric}"))


def plot_noise_sensitivity(noise_df, figures_dir, metric="test_balanced_accuracy"):
    if noise_df is None or len(noise_df) == 0:
        return
    agg = noise_df[noise_df.status == "success"].groupby(["n_qubits", "n_layers", "protocol", "noise_prob"])[metric].agg(["mean", "std", "count"]).reset_index()
    for (nq, depth) in agg[["n_qubits", "n_layers"]].drop_duplicates().itertuples(index=False):
        fig, ax = plt.subplots(figsize=(7, 5))
        for protocol in ["direct", "layerwise"]:
            sub = agg[(agg.n_qubits == nq) & (agg.n_layers == depth) & (agg.protocol == protocol)].sort_values("noise_prob")
            if len(sub) == 0:
                continue
            ax.errorbar(sub.noise_prob, sub["mean"], yerr=sub["std"], marker="o",
                        label=f"{PROTOCOL_LABELS[protocol]} (n={int(sub['count'].max())} seeds)",
                        color=PROTOCOL_COLORS[protocol])
        ax.set_xlabel("Depolarizing probability per gate")
        ax.set_ylabel(metric.replace("_", " "))
        ax.set_title(f"Depolarizing-noise sensitivity (n_qubits={nq}, depth={depth})")
        ax.legend()
        ax.grid(alpha=0.3)
        _savefig(fig, os.path.join(figures_dir, f"noise_sensitivity_nq{nq}_d{depth}"))


def plot_shots_sensitivity(shots_df, figures_dir, metric="test_balanced_accuracy"):
    if shots_df is None or len(shots_df) == 0:
        return
    df = shots_df.copy()
    df["shots_label"] = df["shots"].fillna("analytic").astype(str)
    agg = df[df.status == "success"].groupby(["n_qubits", "n_layers", "protocol", "shots_label"])[metric].agg(["mean", "std", "count"]).reset_index()
    for (nq, depth) in agg[["n_qubits", "n_layers"]].drop_duplicates().itertuples(index=False):
        fig, ax = plt.subplots(figsize=(7, 5))
        order = ["analytic", "1000", "100"]
        for protocol in ["direct", "layerwise"]:
            sub = agg[(agg.n_qubits == nq) & (agg.n_layers == depth) & (agg.protocol == protocol)]
            sub = sub.set_index("shots_label").reindex([o for o in order if o in sub["shots_label"].values]).reset_index()
            if len(sub) == 0:
                continue
            ax.errorbar(range(len(sub)), sub["mean"], yerr=sub["std"], marker="o",
                        label=f"{PROTOCOL_LABELS[protocol]} (n={int(sub['count'].max())} seeds)",
                        color=PROTOCOL_COLORS[protocol])
            ax.set_xticks(range(len(sub)))
            ax.set_xticklabels(sub["shots_label"])
        ax.set_xlabel("Measurement shots (training + evaluation)")
        ax.set_ylabel(metric.replace("_", " "))
        ax.set_title(f"Sampling-noise sensitivity (n_qubits={nq}, depth={depth})")
        ax.legend()
        ax.grid(alpha=0.3)
        _savefig(fig, os.path.join(figures_dir, f"shots_sensitivity_nq{nq}_d{depth}"))


def plot_wallclock_comparison(main_df, figures_dir):
    per_qubit = main_df[main_df.status == "success"].groupby(["n_qubits", "n_layers", "protocol"])["total_wall_clock_s"].mean().reset_index()
    agg = per_qubit.groupby(["n_layers", "protocol"])["total_wall_clock_s"].mean().reset_index()
    fig, ax = plt.subplots(figsize=(8, 5))
    depths = sorted(agg.n_layers.unique())
    width = 0.35
    for i, protocol in enumerate(["direct", "layerwise"]):
        sub = agg[agg.protocol == protocol]
        xs = np.arange(len(depths)) + (i - 0.5) * width
        ys = [sub[sub.n_layers == d]["total_wall_clock_s"].values[0] if d in sub.n_layers.values else 0 for d in depths]
        ax.bar(xs, ys, width=width, label=PROTOCOL_LABELS[protocol], color=PROTOCOL_COLORS[protocol])
    ax.set_xticks(np.arange(len(depths)))
    ax.set_xticklabels(depths)
    ax.set_xlabel("Circuit depth (layers)")
    ax.set_ylabel("Mean wall-clock training time (s)")
    ax.set_title("Wall-clock time comparison (mean across seeds and qubit counts)")
    ax.legend()
    ax.grid(alpha=0.3, axis="y")
    _savefig(fig, os.path.join(figures_dir, "wallclock_comparison"))


def plot_confusion_matrix(cm, class_names, title, path_no_ext):
    fig, ax = plt.subplots(figsize=(5, 4.5))
    im = ax.imshow(cm, cmap="Blues")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names)
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(title)
    fig.colorbar(im, ax=ax)
    _savefig(fig, path_no_ext)
