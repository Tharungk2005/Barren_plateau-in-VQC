import numpy as np

from src import metrics as M


def test_predict_labels_threshold():
    outputs = np.array([-0.5, 0.0, 0.1, -0.01, 0.9])
    preds = M.predict_labels(outputs, threshold=0.0)
    assert list(preds) == [-1, -1, 1, -1, 1]


def test_mse_loss_known_value():
    outputs = np.array([1.0, -1.0])
    labels = np.array([1.0, 1.0])
    assert M.mse_loss(outputs, labels) == 2.0  # (0)^2 + (-2)^2 = 4; mean = 2.0


def test_compute_classification_metrics_perfect():
    y_true = np.array([-1, -1, 1, 1])
    outputs = np.array([-1.0, -1.0, 1.0, 1.0])
    m = M.compute_classification_metrics(y_true, outputs)
    assert m["accuracy"] == 1.0
    assert m["balanced_accuracy"] == 1.0
    assert m["precision"] == 1.0
    assert m["recall"] == 1.0


def test_cost_matrix_asymmetry():
    y_true = np.array([1])  # actual bad
    y_pred_good = np.array([-1])  # predicted good -> cost 5
    cost_matrix = {("good", "bad"): 1, ("bad", "good"): 5, ("good", "good"): 0, ("bad", "bad"): 0}
    total, mean = M.misclassification_cost(y_true, y_pred_good, cost_matrix)
    assert total == 5

    y_true2 = np.array([-1])  # actual good
    y_pred_bad = np.array([1])  # predicted bad -> cost 1
    total2, _ = M.misclassification_cost(y_true2, y_pred_bad, cost_matrix)
    assert total2 == 1


def test_normalized_gradient_norm_scales_with_param_count():
    g_small = np.ones(4)
    g_large = np.ones(16)
    norm_small = M.normalized_gradient_norm(g_small)
    norm_large = M.normalized_gradient_norm(g_large)
    assert abs(norm_small - norm_large) < 1e-9  # same per-parameter magnitude


def test_first_update_reaching_target():
    history = [0.5, 0.6, 0.71, 0.72, 0.73]
    idx = M.first_update_reaching_target(history, target=0.7, patience=2)
    assert idx == 3
    idx_none = M.first_update_reaching_target(history, target=0.99, patience=2)
    assert idx_none is None
