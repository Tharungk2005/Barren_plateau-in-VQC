from src import data_retrieval as dr
from src import preprocessing as pp
from src import baselines as B


def test_baselines_train_and_report_both_representations(cfg):
    df = dr.fetch_raw_dataset(cfg)
    y = pp.encode_labels(df, cfg)
    results = B.train_and_evaluate_baselines(df, y, cfg, n_qubits_for_pca=4, seed=0)
    assert set(results.keys()) == {"logreg_full", "mlp_full", "logreg_pca", "mlp_pca"}
    for key, res in results.items():
        assert 0.0 <= res["val_metrics"]["accuracy"] <= 1.0
        assert 0.0 <= res["test_metrics"]["accuracy"] <= 1.0
        assert res["feature_representation"] in ("full_features", "pca_4d")
