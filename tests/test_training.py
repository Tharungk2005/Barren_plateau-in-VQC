import numpy as np

from src import data_retrieval as dr
from src import preprocessing as pp
from src import training as T
from src import circuits as C


def _tiny_data(cfg):
    df = dr.fetch_raw_dataset(cfg)
    y = pp.encode_labels(df, cfg)
    return pp.prepare_dataset_for_qubits(df, y, 4, cfg, seed=0)


def test_direct_training_reproducible(cfg):
    data = _tiny_data(cfg)
    r1 = T.direct_training_run(4, 2, data, cfg, seed=0, n_updates=10, record_grad_every=5)
    r2 = T.direct_training_run(4, 2, data, cfg, seed=0, n_updates=10, record_grad_every=5)
    assert np.allclose(r1["final_weights"], r2["final_weights"])


def test_direct_training_updates_params(cfg):
    data = _tiny_data(cfg)
    r = T.direct_training_run(4, 2, data, cfg, seed=0, n_updates=10, record_grad_every=5)
    initial_range = cfg["circuit"]["layer_init_range_direct"]
    final_w = np.array(r["final_weights"])
    assert final_w.shape == (2, 4, 2)
    # Params must have moved from their (wide-range) random init on average.
    rng = np.random.default_rng(0)
    init_w = C.init_weights_uniform(4, 2, initial_range[0], initial_range[1], rng)
    assert not np.allclose(final_w, init_w)


def test_loss_finite_and_grad_valid(cfg):
    data = _tiny_data(cfg)
    r = T.direct_training_run(4, 2, data, cfg, seed=0, n_updates=5, record_grad_every=1)
    assert all(np.isfinite(v) for v in r["history"]["train_loss"])
    assert all(np.isfinite(v) for v in r["history"]["grad_norm"])


def test_layerwise_freezes_previous_layers(cfg):
    data = _tiny_data(cfg)
    r = T.layerwise_training_run(4, 3, data, cfg, seed=0, n_updates_total=15, record_grad_every=5)
    for stage in r["stage_logs"]:
        assert stage["frozen_params_unchanged"] is True


def test_layerwise_insertion_loss_reflects_new_layer_not_frozen_state(cfg):
    """Regression test: loss_immediately_after_insertion must measure the
    circuit WITH the newly inserted (near-zero-angle) layer active, not
    redundantly re-measure the old frozen-only state (a bug caught during
    report-writing: the old field was byte-identical to the previous
    stage's final loss because it excluded the new layer entirely)."""
    data = _tiny_data(cfg)
    r = T.layerwise_training_run(4, 3, data, cfg, seed=0, n_updates_total=15, record_grad_every=5)
    for stage in r["stage_logs"]:
        if stage["layer_idx"] == 0:
            continue
        assert stage["loss_immediately_after_insertion"] is not None
        # It must differ from the prior stage's final loss (CZ gates are
        # still active even at near-zero rotation angles -> not identity).
        assert stage["loss_immediately_after_insertion"] != stage["loss_prior_stage_end"]


def test_layerwise_param_count_grows(cfg):
    data = _tiny_data(cfg)
    r = T.layerwise_training_run(4, 3, data, cfg, seed=0, n_updates_total=15, record_grad_every=5)
    final_w = np.array(r["final_weights"])
    assert final_w.shape == (3, 4, 2)
    assert r["n_trainable_params"] == C.n_trainable_params(4, 3)


def test_direct_training_shallow_and_deep(cfg):
    data = _tiny_data(cfg)
    for depth in [2, 8]:
        r = T.direct_training_run(4, depth, data, cfg, seed=0, n_updates=5, record_grad_every=5)
        assert len(r["history"]["train_loss"]) >= 1


def test_evaluate_on_split_metrics_in_range(cfg):
    data = _tiny_data(cfg)
    r = T.direct_training_run(4, 2, data, cfg, seed=0, n_updates=10, record_grad_every=5)
    m = T.evaluate_on_split(r, data["X_test_angles"], data["y_test"], cfg)
    assert 0.0 <= m["accuracy"] <= 1.0
    assert 0.0 <= m["balanced_accuracy"] <= 1.0
    assert m["loss_mse"] >= 0.0
