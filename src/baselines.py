"""Classical baselines: Logistic Regression and a small MLP.

Two input representations are evaluated and clearly labeled:
  - "full_features": all one-hot-encoded categorical + scaled numerical
    features (the standard classical representation, NOT dimensionality
    reduced).
  - "pca_reduced": the same n_qubits-dimensional PCA representation fed
    to the VQC, so a fair comparison at matched input dimensionality is
    also available.
Both use the same stratified train/val/test split and fit exclusively on
training data (ColumnTransformer / scaler / PCA fit on train only).
"""
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.data_retrieval import CATEGORICAL_ATTRIBUTES, NUMERICAL_ATTRIBUTES
from src.preprocessing import stratified_splits
from src import metrics as M


def build_full_feature_pipeline():
    return ColumnTransformer(transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL_ATTRIBUTES),
        ("num", StandardScaler(), NUMERICAL_ATTRIBUTES),
    ])


def _to_01(y_pm1):
    """UCI-style +-1 labels (bad=+1,good=-1) -> {0,1} for sklearn, 1=bad."""
    return (np.asarray(y_pm1) == 1).astype(int)


def _scores_to_pm1_outputs(scores_pos_class):
    """sklearn predict_proba[:,1] in [0,1] -> pseudo circuit-output in
    [-1,1] so the same metrics.compute_classification_metrics function
    can be reused uniformly across classical and quantum models.
    """
    return 2.0 * np.asarray(scores_pos_class) - 1.0


def train_and_evaluate_baselines(df, y, cfg, n_qubits_for_pca=4, seed=0):
    X_train, y_train, X_val, y_val, X_test, y_test = stratified_splits(df, y, cfg)

    y_train01 = _to_01(y_train.values)
    y_val01 = _to_01(y_val.values)
    y_test01 = _to_01(y_test.values)

    results = {}

    # --- full-feature representation ---
    pre = build_full_feature_pipeline()
    Xtr = pre.fit_transform(X_train)
    Xva = pre.transform(X_val)
    Xte = pre.transform(X_test)

    results["logreg_full"] = _fit_eval_sklearn(
        LogisticRegression(max_iter=2000, random_state=seed),
        Xtr, y_train01, Xva, y_val01, Xte, y_test01, y_train.values, y_val.values, y_test.values, cfg,
        model_name="LogisticRegression", feature_rep="full_features",
    )
    results["mlp_full"] = _fit_eval_sklearn(
        MLPClassifier(hidden_layer_sizes=(16,), max_iter=2000, random_state=seed,
                      early_stopping=True, validation_fraction=0.15),
        Xtr, y_train01, Xva, y_val01, Xte, y_test01, y_train.values, y_val.values, y_test.values, cfg,
        model_name="MLP_1hidden16", feature_rep="full_features",
    )

    # --- PCA-reduced representation (matched to VQC input dimensionality) ---
    from src.preprocessing import build_feature_pipeline, AngleScaler
    pca_pipe = build_feature_pipeline(n_qubits_for_pca, seed)
    pca_pipe.fit(X_train)
    Ztr = pca_pipe.transform(X_train)
    Zva = pca_pipe.transform(X_val)
    Zte = pca_pipe.transform(X_test)

    results["logreg_pca"] = _fit_eval_sklearn(
        LogisticRegression(max_iter=2000, random_state=seed),
        Ztr, y_train01, Zva, y_val01, Zte, y_test01, y_train.values, y_val.values, y_test.values, cfg,
        model_name="LogisticRegression", feature_rep=f"pca_{n_qubits_for_pca}d",
    )
    results["mlp_pca"] = _fit_eval_sklearn(
        MLPClassifier(hidden_layer_sizes=(16,), max_iter=2000, random_state=seed,
                      early_stopping=True, validation_fraction=0.15),
        Ztr, y_train01, Zva, y_val01, Zte, y_test01, y_train.values, y_val.values, y_test.values, cfg,
        model_name="MLP_1hidden16", feature_rep=f"pca_{n_qubits_for_pca}d",
    )

    return results


def _fit_eval_sklearn(model, Xtr, ytr01, Xva, yva01, Xte, yte01, y_train_pm1, y_val_pm1, y_test_pm1, cfg,
                       model_name, feature_rep):
    model.fit(Xtr, ytr01)

    threshold = cfg["label_convention"]["decision_threshold"]
    cm_cfg = cfg["cost_matrix"]
    cost_matrix = {
        ("good", "bad"): cm_cfg["cost_actual_good_predicted_bad"],
        ("bad", "good"): cm_cfg["cost_actual_bad_predicted_good"],
        ("good", "good"): cm_cfg["cost_actual_good_predicted_good"],
        ("bad", "bad"): cm_cfg["cost_actual_bad_predicted_bad"],
    }

    def eval_split(X, y01, y_pm1):
        proba = model.predict_proba(X)[:, 1]
        outputs_pm1 = _scores_to_pm1_outputs(proba)
        return M.compute_classification_metrics(y_pm1, outputs_pm1, threshold=threshold, cost_matrix=cost_matrix)

    return {
        "model_name": model_name,
        "feature_representation": feature_rep,
        "val_metrics": eval_split(Xva, yva01, y_val_pm1),
        "test_metrics": eval_split(Xte, yte01, y_test_pm1),
    }
