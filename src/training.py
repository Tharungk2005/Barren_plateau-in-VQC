"""Direct and Layer-wise incremental training protocols for the VQC.

Both protocols share: architecture (src.circuits), MSE loss, Adam optimizer
(implemented manually below so parameter subsets can be frozen exactly),
batch size, learning rate, and a matched total optimizer-update budget.
"""
import time

import numpy as np
from pennylane import numpy as pnp
import pennylane as qml

from src import circuits as C
from src import metrics as M


class AdamOptimizer:
    """Minimal Adam optimizer operating on a single numpy array."""

    def __init__(self, shape, lr=0.01, beta1=0.9, beta2=0.999, eps=1e-8):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.m = np.zeros(shape)
        self.v = np.zeros(shape)
        self.t = 0

    def step(self, params, grad):
        self.t += 1
        self.m = self.beta1 * self.m + (1 - self.beta1) * grad
        self.v = self.beta2 * self.v + (1 - self.beta2) * (grad ** 2)
        mhat = self.m / (1 - self.beta1 ** self.t)
        vhat = self.v / (1 - self.beta2 ** self.t)
        return params - self.lr * mhat / (np.sqrt(vhat) + self.eps)


def estimate_circuit_evaluations(batch_size, n_active_params, diff_method):
    """Documented ESTIMATE of circuit executions for one optimizer update.
    backprop: analytic differentiation does not literally re-execute the
    circuit once per parameter; we count 1 execution-equivalent per sample
    (forward pass) as a conservative lower-bound estimate, clearly labeled.
    parameter-shift: standard cost of 2 circuit evaluations per trainable
    parameter per sample, plus 1 forward evaluation per sample for the loss.
    """
    if diff_method == "backprop":
        return batch_size  # forward-pass equivalent only (estimate)
    return batch_size * (1 + 2 * n_active_params)


def _batch_indices(n_samples, batch_size, rng):
    idx = rng.permutation(n_samples)
    for start in range(0, n_samples, batch_size):
        yield idx[start:start + batch_size]


def _evaluate_full(qnode, weights, X, y, n_layers_active=None):
    preds = np.array(qnode(X, weights, n_layers_active))
    return preds


def _cost_mean_sq(preds, y):
    return float(np.mean((np.asarray(preds) - np.asarray(y)) ** 2))


def direct_training_run(n_qubits, n_layers, data, cfg, seed, shots=None,
                         noise_prob=0.0, n_updates=None, record_grad_every=1):
    """Train an L-layer circuit from scratch with all parameters trainable
    together. Returns a history dict and final weights.
    """
    rng = np.random.default_rng(seed)
    tr_cfg = cfg["training"]
    lr = tr_cfg["learning_rate"]
    batch_size = tr_cfg["batch_size"]
    n_updates = n_updates or (tr_cfg["quick_mode_updates_per_run"] if cfg["quick_mode"]
                               else tr_cfg["full_mode_updates_per_run"])

    low, high = cfg["circuit"]["layer_init_range_direct"]
    weights = pnp.array(
        C.init_weights_uniform(n_qubits, n_layers, low, high, rng),
        requires_grad=True,
    )

    diff_method = cfg["circuit"]["diff_method_exact"] if (shots is None and noise_prob == 0.0) \
        else cfg["circuit"]["diff_method_shots"]
    qnode = C.make_qnode(n_qubits, shots=shots, noise_prob=noise_prob, diff_method=diff_method)

    X_train, y_train = data["X_train_angles"], data["y_train"]
    X_val, y_val = data["X_val_angles"], data["y_val"]

    opt = AdamOptimizer(weights.shape, lr=lr)
    n_active_params = C.n_trainable_params(n_qubits, n_layers)

    history = {
        "train_loss": [], "val_loss": [], "val_accuracy": [], "val_balanced_accuracy": [],
        "grad_norm": [], "grad_norm_normalized": [], "update": [],
        "circuit_evaluations_cumulative": [], "wall_clock_cumulative_s": [],
    }

    def cost(w, Xb, yb):
        preds = qnode(Xb, w)
        return pnp.mean((preds - yb) ** 2)

    grad_fn = qml.grad(cost, argnums=0)

    batch_gen = _batch_indices(len(y_train), batch_size, rng)
    cum_evals = 0
    t_start = time.perf_counter()

    g0 = grad_fn(weights, X_train[:min(batch_size, len(y_train))],
                 y_train[:min(batch_size, len(y_train))])
    grad_norm_init = M.gradient_norm(g0)
    grad_norm_init_normalized = M.normalized_gradient_norm(g0)

    update = 0
    while update < n_updates:
        try:
            batch_idx = next(batch_gen)
        except StopIteration:
            batch_gen = _batch_indices(len(y_train), batch_size, rng)
            batch_idx = next(batch_gen)

        Xb, yb = X_train[batch_idx], y_train[batch_idx]
        grad = grad_fn(weights, Xb, yb)
        grad_np = np.array(grad)
        weights = pnp.array(opt.step(np.array(weights), grad_np), requires_grad=True)

        update += 1
        cum_evals += estimate_circuit_evaluations(len(batch_idx), n_active_params, diff_method)

        if update % record_grad_every == 0 or update == n_updates:
            train_preds = _evaluate_full(qnode, weights, Xb, yb)
            train_loss = _cost_mean_sq(train_preds, yb)
            val_preds = _evaluate_full(qnode, weights, X_val, y_val)
            val_metrics = M.compute_classification_metrics(y_val, val_preds)

            history["update"].append(update)
            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_metrics["loss_mse"])
            history["val_accuracy"].append(val_metrics["accuracy"])
            history["val_balanced_accuracy"].append(val_metrics["balanced_accuracy"])
            history["grad_norm"].append(M.gradient_norm(grad_np))
            history["grad_norm_normalized"].append(M.normalized_gradient_norm(grad_np))
            history["circuit_evaluations_cumulative"].append(cum_evals)
            history["wall_clock_cumulative_s"].append(time.perf_counter() - t_start)

    wall_clock_total = time.perf_counter() - t_start

    return {
        "protocol": "direct",
        "n_qubits": n_qubits, "n_layers": n_layers, "seed": seed,
        "final_weights": np.array(weights).tolist(),
        "n_trainable_params": n_active_params,
        "grad_norm_at_init": grad_norm_init,
        "grad_norm_at_init_normalized": grad_norm_init_normalized,
        "history": history,
        "total_circuit_evaluations": cum_evals,
        "total_wall_clock_s": wall_clock_total,
        "n_updates": n_updates,
        "diff_method": diff_method,
        "shots": shots, "noise_prob": noise_prob,
        "qnode": qnode,
        "final_weights_array": weights,
    }


def layerwise_training_run(n_qubits, n_layers_target, data, cfg, seed, shots=None,
                            noise_prob=0.0, n_updates_total=None, record_grad_every=1):
    """Grow the circuit one layer at a time. At each stage only the newest
    layer's parameters are trainable; all previously trained layers are
    frozen (verified by the caller / tests via weight-equality checks).
    """
    rng = np.random.default_rng(seed)
    tr_cfg = cfg["training"]
    lr = tr_cfg["learning_rate"]
    batch_size = tr_cfg["batch_size"]
    n_updates_total = n_updates_total or (
        tr_cfg["quick_mode_updates_per_run"] if cfg["quick_mode"] else tr_cfg["full_mode_updates_per_run"]
    )
    updates_per_stage = max(1, n_updates_total // n_layers_target)

    near_zero_scale = cfg["circuit"]["layer_init_near_zero_scale"]
    low, high = cfg["circuit"]["layer_init_range_direct"]

    diff_method = cfg["circuit"]["diff_method_exact"] if (shots is None and noise_prob == 0.0) \
        else cfg["circuit"]["diff_method_shots"]
    qnode = C.make_qnode(n_qubits, shots=shots, noise_prob=noise_prob, diff_method=diff_method)

    X_train, y_train = data["X_train_angles"], data["y_train"]
    X_val, y_val = data["X_val_angles"], data["y_val"]

    # First layer uses the same wide init range as Direct Training (there
    # is no "previous solution" to disturb yet).
    frozen_weights = np.zeros((0, n_qubits, 2))
    stage_logs = []
    history = {
        "train_loss": [], "val_loss": [], "val_accuracy": [], "val_balanced_accuracy": [],
        "grad_norm": [], "grad_norm_normalized": [], "update": [], "active_layer": [],
        "circuit_evaluations_cumulative": [], "wall_clock_cumulative_s": [],
    }

    cum_evals = 0
    t_start = time.perf_counter()
    grad_norm_init = None
    grad_norm_init_normalized = None
    global_update = 0

    for layer_idx in range(n_layers_target):
        if layer_idx == 0:
            new_layer = C.init_weights_uniform(n_qubits, 1, low, high, rng)
        else:
            new_layer = C.init_new_layer_near_zero(n_qubits, near_zero_scale, rng)
        active = pnp.array(new_layer, requires_grad=True)
        frozen_const = pnp.array(frozen_weights, requires_grad=False)

        loss_prior_stage_end = None
        loss_immediately_after_insertion = None
        if layer_idx > 0:
            full_prior = np.concatenate([frozen_weights, np.array(active)], axis=0)
            preds_prior = _evaluate_full(qnode, full_prior, X_val, y_val, n_layers_active=layer_idx)
            loss_prior_stage_end = _cost_mean_sq(preds_prior, y_val)
            # Loss WITH the new (near-zero-angle) layer active but before any
            # of this stage's optimizer updates -- the direct test of the
            # architectural caveat that a near-zero layer is not an identity
            # operation, because its CZ gates remain active.
            preds_after_insertion = _evaluate_full(qnode, full_prior, X_val, y_val, n_layers_active=layer_idx + 1)
            loss_immediately_after_insertion = _cost_mean_sq(preds_after_insertion, y_val)

        def cost(active_w, frozen_w, Xb, yb):
            full_w = pnp.concatenate([frozen_w, active_w], axis=0)
            preds = qnode(Xb, full_w)
            return pnp.mean((preds - yb) ** 2)

        grad_fn = qml.grad(cost, argnums=0)
        opt = AdamOptimizer(active.shape, lr=lr)
        n_active_params = C.n_trainable_params(n_qubits, 1)

        batch_gen = _batch_indices(len(y_train), batch_size, rng)

        g0 = grad_fn(active, frozen_const,
                     X_train[:min(batch_size, len(y_train))],
                     y_train[:min(batch_size, len(y_train))])
        if layer_idx == 0:
            grad_norm_init = M.gradient_norm(g0)
            grad_norm_init_normalized = M.normalized_gradient_norm(g0)

        frozen_before_stage = np.array(frozen_const).copy()

        for s in range(updates_per_stage):
            try:
                batch_idx = next(batch_gen)
            except StopIteration:
                batch_gen = _batch_indices(len(y_train), batch_size, rng)
                batch_idx = next(batch_gen)
            Xb, yb = X_train[batch_idx], y_train[batch_idx]

            grad = grad_fn(active, frozen_const, Xb, yb)
            grad_np = np.array(grad)
            active = pnp.array(opt.step(np.array(active), grad_np), requires_grad=True)

            global_update += 1
            cum_evals += estimate_circuit_evaluations(len(batch_idx), n_active_params, diff_method)

            if (s + 1) % record_grad_every == 0 or (s + 1) == updates_per_stage:
                full_w = np.concatenate([frozen_weights, np.array(active)], axis=0)
                train_preds = _evaluate_full(qnode, full_w, Xb, yb, n_layers_active=layer_idx + 1)
                train_loss = _cost_mean_sq(train_preds, yb)
                val_preds = _evaluate_full(qnode, full_w, X_val, y_val, n_layers_active=layer_idx + 1)
                val_metrics = M.compute_classification_metrics(y_val, val_preds)

                history["update"].append(global_update)
                history["active_layer"].append(layer_idx)
                history["train_loss"].append(train_loss)
                history["val_loss"].append(val_metrics["loss_mse"])
                history["val_accuracy"].append(val_metrics["accuracy"])
                history["val_balanced_accuracy"].append(val_metrics["balanced_accuracy"])
                history["grad_norm"].append(M.gradient_norm(grad_np))
                history["grad_norm_normalized"].append(M.normalized_gradient_norm(grad_np))
                history["circuit_evaluations_cumulative"].append(cum_evals)
                history["wall_clock_cumulative_s"].append(time.perf_counter() - t_start)

        # Freeze-integrity check: frozen block must be bit-identical to
        # before this stage began.
        frozen_unchanged = bool(np.array_equal(frozen_before_stage, np.array(frozen_const)))

        full_w_after = np.concatenate([frozen_weights, np.array(active)], axis=0)
        preds_after = _evaluate_full(qnode, full_w_after, X_val, y_val, n_layers_active=layer_idx + 1)
        loss_after_stage = _cost_mean_sq(preds_after, y_val)

        stage_logs.append({
            "layer_idx": layer_idx,
            "updates_in_stage": updates_per_stage,
            "loss_prior_stage_end": loss_prior_stage_end,
            "loss_immediately_after_insertion": loss_immediately_after_insertion,
            "loss_after_stage": loss_after_stage,
            "frozen_params_unchanged": frozen_unchanged,
        })

        frozen_weights = full_w_after

    wall_clock_total = time.perf_counter() - t_start
    n_active_params_total = C.n_trainable_params(n_qubits, n_layers_target)

    return {
        "protocol": "layerwise",
        "n_qubits": n_qubits, "n_layers": n_layers_target, "seed": seed,
        "final_weights": frozen_weights.tolist(),
        "n_trainable_params": n_active_params_total,
        "grad_norm_at_init": grad_norm_init,
        "grad_norm_at_init_normalized": grad_norm_init_normalized,
        "history": history,
        "stage_logs": stage_logs,
        "total_circuit_evaluations": cum_evals,
        "total_wall_clock_s": wall_clock_total,
        "n_updates": global_update,
        "updates_per_stage": updates_per_stage,
        "diff_method": diff_method,
        "shots": shots, "noise_prob": noise_prob,
        "qnode": qnode,
        "final_weights_array": pnp.array(frozen_weights),
    }


def evaluate_on_split(run_result, X, y, cfg):
    qnode = run_result["qnode"]
    weights = np.array(run_result["final_weights"])
    preds = _evaluate_full(qnode, weights, X, y)
    cost_matrix = None
    cm_cfg = cfg.get("cost_matrix")
    if cm_cfg:
        cost_matrix = {
            ("good", "bad"): cm_cfg["cost_actual_good_predicted_bad"],
            ("bad", "good"): cm_cfg["cost_actual_bad_predicted_good"],
            ("good", "good"): cm_cfg["cost_actual_good_predicted_good"],
            ("bad", "bad"): cm_cfg["cost_actual_bad_predicted_bad"],
        }
    threshold = cfg["label_convention"]["decision_threshold"]
    return M.compute_classification_metrics(y, preds, threshold=threshold, cost_matrix=cost_matrix)
