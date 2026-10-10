"""Preprocessing: stratified split, categorical/numerical encoding, PCA
dimensionality reduction to n_qubits features, angle scaling to [0, pi].

Leakage prevention: all fitted transforms (OneHotEncoder, StandardScaler,
PCA, angle min-max scaler) are fit on the TRAIN split only, then applied
(transform-only) to validation and test splits.
"""
import os

import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from src.data_retrieval import (
    CATEGORICAL_ATTRIBUTES, NUMERICAL_ATTRIBUTES, TARGET_COLUMN,
    FEATURE_COLUMNS,
)
from src.utils import project_path, save_json, ensure_dir


def encode_labels(df, cfg):
    """Map UCI classes {1=good, 2=bad} to the project label convention."""
    lc = cfg["label_convention"]
    mapping = {
        lc["good_credit_uci_class"]: lc["good_credit_encoded_label"],
        lc["bad_credit_uci_class"]: lc["bad_credit_encoded_label"],
    }
    y = df[TARGET_COLUMN].map(mapping)
    if y.isnull().any():
        raise ValueError("Unmapped class label encountered during label encoding.")
    return y.astype(int).values


def stratified_splits(df, y, cfg):
    sp = cfg["data"]["split"]
    seed = sp["split_seed"]
    X = df[FEATURE_COLUMNS].reset_index(drop=True)
    y = pd.Series(y).reset_index(drop=True)

    test_frac = sp["test_fraction"]
    val_frac = sp["val_fraction"]

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=(test_frac + val_frac), random_state=seed,
        stratify=y if sp["stratify"] else None,
    )
    rel_test_frac = test_frac / (test_frac + val_frac)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=rel_test_frac, random_state=seed,
        stratify=y_temp if sp["stratify"] else None,
    )

    idx_train = set(X_train.index)
    idx_val = set(X_val.index)
    idx_test = set(X_test.index)
    assert not (idx_train & idx_val), "Train/val index overlap detected"
    assert not (idx_train & idx_test), "Train/test index overlap detected"
    assert not (idx_val & idx_test), "Val/test index overlap detected"

    return (
        X_train.reset_index(drop=True), y_train.reset_index(drop=True),
        X_val.reset_index(drop=True), y_val.reset_index(drop=True),
        X_test.reset_index(drop=True), y_test.reset_index(drop=True),
    )


def build_feature_pipeline(n_qubits, seed):
    """ColumnTransformer (OHE + StandardScaler) -> PCA(n_qubits).
    Fit exclusively on training data by the caller.
    """
    categorical_tf = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    numerical_tf = StandardScaler()

    pre = ColumnTransformer(
        transformers=[
            ("cat", categorical_tf, CATEGORICAL_ATTRIBUTES),
            ("num", numerical_tf, NUMERICAL_ATTRIBUTES),
        ]
    )
    pipeline = Pipeline(steps=[
        ("pre", pre),
        ("pca", PCA(n_components=n_qubits, random_state=seed)),
    ])
    return pipeline


class AngleScaler:
    """Per-feature min-max scaler to [low, high], fit on training data only.
    Validation/test values outside the training min/max range are clipped
    (documented behaviour; such values are naturally rare for PCA scores but
    not impossible for an unseen distribution tail).
    """

    def __init__(self, low=0.0, high=np.pi):
        self.low = low
        self.high = high
        self.min_ = None
        self.max_ = None

    def fit(self, X):
        self.min_ = X.min(axis=0)
        self.max_ = X.max(axis=0)
        self.range_ = np.where(self.max_ - self.min_ == 0, 1.0, self.max_ - self.min_)
        return self

    def transform(self, X):
        scaled = (X - self.min_) / self.range_
        scaled = np.clip(scaled, 0.0, 1.0)
        return self.low + scaled * (self.high - self.low)

    def to_dict(self):
        return {
            "low": self.low, "high": self.high,
            "min_": self.min_.tolist(), "max_": self.max_.tolist(),
        }


def prepare_dataset_for_qubits(df, y, n_qubits, cfg, seed=None):
    """Full leakage-safe preprocessing pipeline for a given qubit count.

    Returns a dict with train/val/test angle-encoded features + labels,
    plus the fitted pipeline and angle scaler for persistence.
    """
    split_seed = cfg["data"]["split"]["split_seed"]
    pca_seed = seed if seed is not None else split_seed

    X_train, y_train, X_val, y_val, X_test, y_test = stratified_splits(df, y, cfg)

    pipeline = build_feature_pipeline(n_qubits, pca_seed)
    pipeline.fit(X_train)

    Z_train = pipeline.transform(X_train)
    Z_val = pipeline.transform(X_val)
    Z_test = pipeline.transform(X_test)

    angle_low, angle_high = cfg["preprocessing"]["angle_range"]
    scaler = AngleScaler(low=angle_low, high=angle_high)
    scaler.fit(Z_train)

    A_train = scaler.transform(Z_train)
    A_val = scaler.transform(Z_val)
    A_test = scaler.transform(Z_test)

    explained_var = pipeline.named_steps["pca"].explained_variance_ratio_.tolist()

    return {
        "X_train_angles": A_train, "y_train": y_train.values,
        "X_val_angles": A_val, "y_val": y_val.values,
        "X_test_angles": A_test, "y_test": y_test.values,
        "pipeline": pipeline, "angle_scaler": scaler,
        "n_qubits": n_qubits,
        "pca_explained_variance_ratio": explained_var,
        "n_train": len(y_train), "n_val": len(y_val), "n_test": len(y_test),
    }


def save_pipeline_artifacts(prepared, cfg):
    n_qubits = prepared["n_qubits"]
    pipeline_path = project_path(
        cfg["preprocessing"]["pipeline_path_template"].format(n_qubits=n_qubits)
    )
    ensure_dir(os.path.dirname(pipeline_path))
    joblib.dump(
        {"pipeline": prepared["pipeline"], "angle_scaler": prepared["angle_scaler"]},
        pipeline_path,
    )

    meta_path = project_path(cfg["preprocessing"]["processed_metadata_path"])
    meta = {}
    if os.path.exists(meta_path):
        import json
        with open(meta_path) as f:
            meta = json.load(f)
    meta[str(n_qubits)] = {
        "n_qubits": n_qubits,
        "n_train": prepared["n_train"],
        "n_val": prepared["n_val"],
        "n_test": prepared["n_test"],
        "pca_explained_variance_ratio": prepared["pca_explained_variance_ratio"],
        "angle_scaler": prepared["angle_scaler"].to_dict(),
        "pipeline_path": pipeline_path,
    }
    save_json(meta, meta_path)
    return pipeline_path
