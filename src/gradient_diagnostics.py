"""Barren-plateau gradient-variance diagnostic.

This is a CIRCUIT-TRAINABILITY DIAGNOSTIC using random parameter
initializations, independent of the credit-risk dataset and NOT a
substitute for the dataset classification experiment. For each
(n_qubits, depth) combination, `n_samples` independent random circuits
are drawn (uniform[0, 2*pi] initialization, matching the Direct Training
init distribution), the gradient of a fixed observable w.r.t. one
representative parameter is computed, and the sample variance of that
gradient component is reported together with a 95% confidence interval
of the variance (via bootstrap) and the gradient-norm distribution.
"""
import numpy as np
import pennylane as qml
from pennylane import numpy as pnp

from src import circuits as C


def sample_gradients(n_qubits, n_layers, n_samples, rng, readout_qubit=0,
                      fixed_input=True):
    """Draw n_samples random circuits; for each, compute the full gradient
    of expval(PauliZ(readout_qubit)) w.r.t. all trainable parameters at a
    fresh uniform[0,2pi] initialization. Returns:
      grad_component: gradient of the FIRST parameter (theta_{1,1}, the
        conventional single-parameter barren-plateau diagnostic), array
        of shape (n_samples,).
      grad_norms: full-gradient-vector norm per sample, shape (n_samples,).
    """
    qnode = C.make_qnode(n_qubits, readout_qubit=readout_qubit, shots=None)
    x = np.full(n_qubits, np.pi / 4) if fixed_input else rng.uniform(0, np.pi, n_qubits)

    grad_component = np.zeros(n_samples)
    grad_norms = np.zeros(n_samples)

    for s in range(n_samples):
        w = pnp.array(rng.uniform(0, 2 * np.pi, size=(n_layers, n_qubits, 2)), requires_grad=True)
        g = qml.grad(qnode, argnums=1)(x, w)
        g_np = np.array(g)
        grad_component[s] = g_np[0, 0, 0]
        grad_norms[s] = np.linalg.norm(g_np)

    return grad_component, grad_norms


def bootstrap_variance_ci(samples, n_boot=2000, ci=0.95, rng=None):
    rng = rng or np.random.default_rng(0)
    n = len(samples)
    boot_vars = np.empty(n_boot)
    for b in range(n_boot):
        resample = rng.choice(samples, size=n, replace=True)
        boot_vars[b] = np.var(resample)
    lo = np.percentile(boot_vars, (1 - ci) / 2 * 100)
    hi = np.percentile(boot_vars, (1 + ci) / 2 * 100)
    return float(lo), float(hi)


def run_gradient_diagnostic(qubit_counts, depths, n_samples, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for nq in qubit_counts:
        for depth in depths:
            grad_component, grad_norms = sample_gradients(nq, depth, n_samples, rng)
            var = float(np.var(grad_component))
            ci_lo, ci_hi = bootstrap_variance_ci(grad_component, rng=rng)
            rows.append({
                "n_qubits": nq, "n_layers": depth, "n_samples": n_samples,
                "gradient_component_variance": var,
                "gradient_component_variance_ci_lo": ci_lo,
                "gradient_component_variance_ci_hi": ci_hi,
                "gradient_component_mean": float(np.mean(grad_component)),
                "grad_norm_mean": float(np.mean(grad_norms)),
                "grad_norm_std": float(np.std(grad_norms)),
                "init_distribution": "uniform[0, 2*pi)",
                "gradient_method": "backprop (exact, autograd)",
                "fixed_input": "x_i = pi/4 for all qubits (fixed encoding, varying only ansatz params)",
            })
    return rows


def fit_log_linear_trend(depths, variances):
    """Fit log(variance) = a*depth + b via least squares; report slope,
    intercept, and R^2 so any 'exponential decay' claim is evidence-based
    rather than asserted. Returns None values if variances contain
    non-positive entries (log undefined).
    """
    depths = np.asarray(depths, dtype=float)
    variances = np.asarray(variances, dtype=float)
    if np.any(variances <= 0):
        return {"slope": None, "intercept": None, "r_squared": None, "note": "non-positive variance; log-linear fit skipped"}
    log_v = np.log(variances)
    A = np.vstack([depths, np.ones_like(depths)]).T
    coef, residuals, rank, sv = np.linalg.lstsq(A, log_v, rcond=None)
    slope, intercept = coef
    pred = A @ coef
    ss_res = np.sum((log_v - pred) ** 2)
    ss_tot = np.sum((log_v - np.mean(log_v)) ** 2)
    r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else None
    return {"slope": float(slope), "intercept": float(intercept), "r_squared": float(r_squared) if r_squared is not None else None}
