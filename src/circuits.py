"""Variational quantum circuit definition used by BOTH training protocols.

Architecture (fixed, documented):
  Encoding:  RY(x_i) on each qubit i, i = 0..n_qubits-1.
  Per layer: trainable RY(theta) then trainable RZ(phi) on every qubit,
             followed by a ring of fixed CZ entangling gates between
             neighbouring qubits: edges (0,1),(1,2),...,(n-2,n-1),(n-1,0).
             For n_qubits == 2 the "ring" has only ONE distinct edge
             (0,1); it is applied once, not twice.
  Readout:   expval(PauliZ(readout_qubit)), readout_qubit = 0 by default.

Trainable parameter tensor shape: (n_layers, n_qubits, 2), where
index 0 of the last axis is the RY angle and index 1 is the RZ angle.
Parameter count: p = 2 * n_qubits * n_layers (CZ gates are fixed, not
trainable, and contribute 0 parameters).

IMPORTANT CAVEAT (documented per project spec section 6): because the CZ
ring gates are always active, setting a newly inserted layer's RY/RZ
angles to (near) zero does NOT make that layer an identity operation --
the CZ gates still entangle the qubits. Layer-wise training therefore
measures and reports the loss perturbation that each layer insertion
causes, rather than assuming it is negligible.
"""
import pennylane as qml
from pennylane import numpy as pnp
import numpy as np


def entangling_edges(n_qubits):
    """Ring topology edges, de-duplicated for the n_qubits == 2 case."""
    if n_qubits < 2:
        return []
    if n_qubits == 2:
        return [(0, 1)]
    return [(i, (i + 1) % n_qubits) for i in range(n_qubits)]


def n_trainable_params(n_qubits, n_layers):
    return 2 * n_qubits * n_layers


def _apply_encoding(x, n_qubits):
    for i in range(n_qubits):
        qml.RY(x[..., i], wires=i)


def _apply_layer(weights_layer, n_qubits):
    """weights_layer: shape (n_qubits, 2) -> [:,0]=RY angles, [:,1]=RZ angles."""
    for i in range(n_qubits):
        qml.RY(weights_layer[i, 0], wires=i)
        qml.RZ(weights_layer[i, 1], wires=i)
    for (a, b) in entangling_edges(n_qubits):
        qml.CZ(wires=[a, b])


def build_circuit_tape(x, weights, n_qubits, n_layers_active=None):
    """Construct the data-encoding + ansatz tape for `n_layers_active`
    layers (defaults to weights.shape[0]) and return the readout op.
    `weights` has shape (n_layers_total, n_qubits, 2); only the first
    `n_layers_active` layers are applied (used by layer-wise training
    stages where later layers do not exist yet).
    """
    n_layers_active = weights.shape[0] if n_layers_active is None else n_layers_active
    _apply_encoding(x, n_qubits)
    for layer_idx in range(n_layers_active):
        _apply_layer(weights[layer_idx], n_qubits)


def make_qnode(n_qubits, readout_qubit=0, shots=None, noise_prob=0.0,
               diff_method=None):
    """Build a QNode for the architecture above.

    shots: None -> exact expectation values (analytic). int -> finite-shot
           sampling estimate of the expectation value.
    noise_prob: depolarizing-channel probability applied after every gate
           on every qubit (0.0 -> noiseless, uses default.qubit; > 0 uses
           default.mixed density-matrix simulation).
    diff_method: override; otherwise backprop (exact, noiseless),
           parameter-shift (shots or noisy).
    """
    if noise_prob and noise_prob > 0.0:
        dev = qml.device("default.mixed", wires=n_qubits)
        default_diff = "parameter-shift"
    else:
        dev = qml.device("default.qubit", wires=n_qubits)
        default_diff = "backprop" if shots is None else "parameter-shift"

    diff_method = diff_method or default_diff

    def circuit_fn(x, weights, n_layers_active=None):
        n_layers_active = weights.shape[0] if n_layers_active is None else n_layers_active
        for i in range(n_qubits):
            qml.RY(x[..., i], wires=i)
            if noise_prob and noise_prob > 0.0:
                qml.DepolarizingChannel(noise_prob, wires=i)
        for layer_idx in range(n_layers_active):
            wl = weights[layer_idx]
            for i in range(n_qubits):
                qml.RY(wl[i, 0], wires=i)
                qml.RZ(wl[i, 1], wires=i)
                if noise_prob and noise_prob > 0.0:
                    qml.DepolarizingChannel(noise_prob, wires=i)
            for (a, b) in entangling_edges(n_qubits):
                qml.CZ(wires=[a, b])
        return qml.expval(qml.PauliZ(readout_qubit))

    qnode = qml.QNode(circuit_fn, dev, diff_method=diff_method)
    if shots is not None:
        qnode = qml.set_shots(qnode, shots=shots)
    return qnode


def init_weights_uniform(n_qubits, n_layers, low, high, rng):
    return rng.uniform(low, high, size=(n_layers, n_qubits, 2))


def init_new_layer_near_zero(n_qubits, scale, rng):
    return rng.uniform(-scale, scale, size=(1, n_qubits, 2))


def circuit_drawing_text(n_qubits, n_layers, readout_qubit=0):
    qnode = make_qnode(n_qubits, readout_qubit=readout_qubit, shots=None, noise_prob=0.0)
    rng = np.random.default_rng(0)
    x = rng.uniform(0, np.pi, n_qubits)
    w = rng.uniform(0, 2 * np.pi, size=(n_layers, n_qubits, 2))
    return qml.draw(qnode)(x, w)


def circuit_drawing_mpl(n_qubits, n_layers, readout_qubit=0):
    qnode = make_qnode(n_qubits, readout_qubit=readout_qubit, shots=None, noise_prob=0.0)
    rng = np.random.default_rng(0)
    x = rng.uniform(0, np.pi, n_qubits)
    w = rng.uniform(0, 2 * np.pi, size=(n_layers, n_qubits, 2))
    fig, ax = qml.draw_mpl(qnode, decimals=2)(x, w)
    return fig, ax
