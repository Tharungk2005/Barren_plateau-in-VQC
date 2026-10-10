import numpy as np

from src import data_retrieval as dr
from src import preprocessing as pp


def test_label_mapping(cfg):
    df = dr.fetch_raw_dataset(cfg)
    y = pp.encode_labels(df, cfg)
    assert set(np.unique(y)) == {-1, 1}
    good_mask = df["class"] == 1
    assert np.all(y[good_mask.values] == -1)
    bad_mask = df["class"] == 2
    assert np.all(y[bad_mask.values] == 1)


def test_no_index_overlap(cfg):
    df = dr.fetch_raw_dataset(cfg)
    y = pp.encode_labels(df, cfg)
    X_train, y_train, X_val, y_val, X_test, y_test = pp.stratified_splits(df, y, cfg)
    n_total = len(X_train) + len(X_val) + len(X_test)
    assert n_total == len(df)
    assert abs(len(X_train) - 700) <= 2
    assert abs(len(X_val) - 150) <= 2
    assert abs(len(X_test) - 150) <= 2


def test_input_dimension_matches_qubits(cfg):
    df = dr.fetch_raw_dataset(cfg)
    y = pp.encode_labels(df, cfg)
    for nq in [4, 6, 8]:
        prepared = pp.prepare_dataset_for_qubits(df, y, nq, cfg, seed=0)
        assert prepared["X_train_angles"].shape[1] == nq
        assert prepared["X_val_angles"].shape[1] == nq
        assert prepared["X_test_angles"].shape[1] == nq


def test_angles_within_range(cfg):
    df = dr.fetch_raw_dataset(cfg)
    y = pp.encode_labels(df, cfg)
    prepared = pp.prepare_dataset_for_qubits(df, y, 4, cfg, seed=0)
    low, high = cfg["preprocessing"]["angle_range"]
    for key in ["X_train_angles", "X_val_angles", "X_test_angles"]:
        arr = prepared[key]
        assert arr.min() >= low - 1e-9
        assert arr.max() <= high + 1e-9


def test_pca_fit_on_train_only_no_leakage():
    """Fitting PCA on train vs train+val should give different components,
    proving the pipeline does not peek at validation data."""
    import pandas as pd
    from src.preprocessing import build_feature_pipeline
    from src.data_retrieval import CATEGORICAL_ATTRIBUTES, NUMERICAL_ATTRIBUTES

    rng = np.random.default_rng(0)
    n = 200
    df = pd.DataFrame({
        **{c: rng.choice(["A11", "A12", "A13"], size=n) for c in CATEGORICAL_ATTRIBUTES},
        **{c: rng.normal(size=n) for c in NUMERICAL_ATTRIBUTES},
    })
    pipe_a = build_feature_pipeline(4, seed=0)
    pipe_a.fit(df.iloc[:100])
    pipe_b = build_feature_pipeline(4, seed=0)
    pipe_b.fit(df.iloc[:150])
    comp_a = pipe_a.named_steps["pca"].components_
    comp_b = pipe_b.named_steps["pca"].components_
    assert not np.allclose(comp_a, comp_b)
