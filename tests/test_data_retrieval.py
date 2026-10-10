from src import data_retrieval as dr


def test_fetch_raw_dataset_schema(cfg):
    df = dr.fetch_raw_dataset(cfg)
    assert df.shape[0] == cfg["data"]["expected_n_instances"]
    assert len(dr.FEATURE_COLUMNS) == cfg["data"]["expected_n_features"]
    assert set(df["class"].unique()).issubset({1, 2})
    assert df.isnull().sum().sum() == 0


def test_class_counts_match_documentation(cfg):
    df = dr.fetch_raw_dataset(cfg)
    counts = df["class"].value_counts().to_dict()
    assert counts[1] == 700
    assert counts[2] == 300


def test_validate_schema_rejects_bad_row_count(cfg):
    df = dr.fetch_raw_dataset(cfg)
    bad_df = df.iloc[:10].copy()
    try:
        dr.validate_schema(bad_df, cfg)
        assert False, "expected DatasetRetrievalError"
    except dr.DatasetRetrievalError:
        pass
