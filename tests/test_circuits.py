import numpy as np
import pennylane as qml

from src import circuits as C


def test_entangling_edges_two_qubits_no_duplicate():
    edges = C.entangling_edges(2)
    assert edges == [(0, 1)]


def test_entangling_edges_ring_four_qubits():
    edges = C.entangling_edges(4)
    assert edges == [(0, 1), (1, 2), (2, 3), (3, 0)]
    assert len(set(edges)) == len(edges)


def test_param_count_formula():
    for nq in [4, 6, 8]:
        for depth in [2, 6, 12, 24]:
            assert C.n_trainable_params(nq, depth) == 2 * nq * depth


def test_output_range():
    rng = np.random.default_rng(0)
    qnode = C.make_qnode(4, shots=None)
    x = rng.uniform(0, np.pi, 4)
    w = rng.uniform(0, 2 * np.pi, size=(3, 4, 2))
    out = qnode(x, w)
    assert -1.0 - 1e-9 <= out <= 1.0 + 1e-9


def test_gradient_finite():
    rng = np.random.default_rng(0)
    qnode = C.make_qnode(4, shots=None)
    x = rng.uniform(0, np.pi, 4)
    from pennylane import numpy as pnp
    w = pnp.array(rng.uniform(0, 2 * np.pi, size=(3, 4, 2)), requires_grad=True)
    grad = qml.grad(qnode, argnums=1)(x, w)
    g = np.array(grad)
    assert np.all(np.isfinite(g))
    assert g.shape == (3, 4, 2)


def test_batched_forward_matches_single_sample():
    rng = np.random.default_rng(1)
    qnode = C.make_qnode(4, shots=None)
    X = rng.uniform(0, np.pi, size=(5, 4))
    w = rng.uniform(0, 2 * np.pi, size=(2, 4, 2))
    batched = np.array(qnode(X, w))
    singles = np.array([qnode(X[i], w) for i in range(5)])
    assert np.allclose(batched, singles, atol=1e-8)
