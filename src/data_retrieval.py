"""Retrieval, caching, and schema validation of the real UCI Statlog German
Credit dataset (id=144). No synthetic data is generated here.

Source: https://archive.ics.uci.edu/dataset/144/statlog+german+credit+data
DOI: https://doi.org/10.24432/C5NC77
"""
import os

import pandas as pd

from src.utils import project_path, save_json, load_json, now_iso, ensure_dir

FEATURE_COLUMNS = [f"Attribute{i}" for i in range(1, 21)]
TARGET_COLUMN = "class"

CATEGORICAL_ATTRIBUTES = [
    "Attribute1", "Attribute3", "Attribute4", "Attribute6", "Attribute7",
    "Attribute9", "Attribute10", "Attribute12", "Attribute14", "Attribute15",
    "Attribute17", "Attribute19", "Attribute20",
]
NUMERICAL_ATTRIBUTES = [
    "Attribute2", "Attribute5", "Attribute8", "Attribute11", "Attribute13",
    "Attribute16", "Attribute18",
]

ATTRIBUTE_DESCRIPTIONS = {
    "Attribute1": "Status of existing checking account (qualitative)",
    "Attribute2": "Duration in month (numerical)",
    "Attribute3": "Credit history (qualitative)",
    "Attribute4": "Purpose (qualitative)",
    "Attribute5": "Credit amount (numerical)",
    "Attribute6": "Savings account/bonds (qualitative)",
    "Attribute7": "Present employment since (qualitative)",
    "Attribute8": "Installment rate in percentage of disposable income (numerical)",
    "Attribute9": "Personal status and sex (qualitative)",
    "Attribute10": "Other debtors / guarantors (qualitative)",
    "Attribute11": "Present residence since (numerical)",
    "Attribute12": "Property (qualitative)",
    "Attribute13": "Age in years (numerical)",
    "Attribute14": "Other installment plans (qualitative)",
    "Attribute15": "Housing (qualitative)",
    "Attribute16": "Number of existing credits at this bank (numerical)",
    "Attribute17": "Job (qualitative)",
    "Attribute18": "Number of people being liable to provide maintenance for (numerical)",
    "Attribute19": "Telephone (qualitative)",
    "Attribute20": "foreign worker (qualitative)",
    "class": "Target: 1 = good credit risk, 2 = bad credit risk",
}

# Cost matrix transcribed verbatim from the UCI metadata additional_info.summary
# field (rows = actual class, columns = predicted class; 1=Good, 2=Bad).
UCI_COST_MATRIX = {
    ("good", "bad"): 1,   # actual good, predicted bad -> cost 1
    ("bad", "good"): 5,   # actual bad, predicted good -> cost 5 (worse mistake)
    ("good", "good"): 0,
    ("bad", "bad"): 0,
}


class DatasetRetrievalError(RuntimeError):
    pass


def fetch_raw_dataset(cfg, force_download=False):
    """Fetch the raw German Credit dataset, using a local cache when present.

    Returns a DataFrame with columns Attribute1..Attribute20 + class.
    Raises DatasetRetrievalError with recovery instructions if the official
    source is unreachable and no local cache exists.
    """
    raw_path = project_path(cfg["data"]["raw_cache_path"])
    meta_path = project_path(cfg["data"]["raw_metadata_path"])

    if os.path.exists(raw_path) and not force_download:
        df = pd.read_csv(raw_path)
        validate_schema(df, cfg)
        return df

    try:
        from ucimlrepo import fetch_ucirepo
        repo = fetch_ucirepo(id=cfg["data"]["uci_dataset_id"])
        X = repo.data.features.copy()
        y = repo.data.targets.copy()
        df = pd.concat([X, y], axis=1)
        df.columns = list(X.columns) + [TARGET_COLUMN]
    except Exception as exc:
        raise DatasetRetrievalError(
            "Could not retrieve the official UCI Statlog German Credit "
            "dataset (id=144) via the 'ucimlrepo' package.\n"
            f"Underlying error: {exc}\n\n"
            "Recovery instructions:\n"
            "1. Check your internet connection.\n"
            "2. Retry: python -c \"from ucimlrepo import fetch_ucirepo; "
            "fetch_ucirepo(id=144)\"\n"
            "3. If ucimlrepo remains unreachable, manually download "
            "'german.data' from "
            "https://archive.ics.uci.edu/dataset/144/statlog+german+credit+data "
            "and place the parsed CSV at "
            f"{raw_path} with columns Attribute1..Attribute20,class.\n"
            "4. Do NOT substitute a different dataset. Re-run once the file "
            "is in place."
        ) from exc

    validate_schema(df, cfg)

    ensure_dir(os.path.dirname(raw_path))
    df.to_csv(raw_path, index=False)

    metadata = {
        "source_url": cfg["data"]["uci_url"],
        "doi": cfg["data"]["uci_doi"],
        "uci_dataset_id": cfg["data"]["uci_dataset_id"],
        "retrieval_method": "ucimlrepo.fetch_ucirepo",
        "retrieved_at": now_iso(),
        "n_instances": int(df.shape[0]),
        "n_features": int(len(FEATURE_COLUMNS)),
        "target_column": TARGET_COLUMN,
        "class_counts": {str(k): int(v) for k, v in df[TARGET_COLUMN].value_counts().items()},
        "categorical_attributes": CATEGORICAL_ATTRIBUTES,
        "numerical_attributes": NUMERICAL_ATTRIBUTES,
        "attribute_descriptions": ATTRIBUTE_DESCRIPTIONS,
        "cost_matrix_source": "UCI metadata additional_info.summary (verbatim)",
        "cost_matrix": {f"{a}->{b}": v for (a, b), v in UCI_COST_MATRIX.items()},
    }
    save_json(metadata, meta_path)
    return df


def validate_schema(df, cfg):
    expected_n = cfg["data"]["expected_n_instances"]
    expected_feat = cfg["data"]["expected_n_features"]

    missing_cols = [c for c in FEATURE_COLUMNS + [TARGET_COLUMN] if c not in df.columns]
    if missing_cols:
        raise DatasetRetrievalError(f"Dataset schema invalid, missing columns: {missing_cols}")

    if df.shape[0] != expected_n:
        raise DatasetRetrievalError(
            f"Dataset row count mismatch: expected {expected_n}, got {df.shape[0]}."
        )
    if len(FEATURE_COLUMNS) != expected_feat:
        raise DatasetRetrievalError(
            f"Feature column count mismatch: expected {expected_feat}, got {len(FEATURE_COLUMNS)}."
        )

    valid_classes = {1, 2}
    found_classes = set(df[TARGET_COLUMN].unique().tolist())
    if not found_classes.issubset(valid_classes):
        raise DatasetRetrievalError(f"Unexpected target classes found: {found_classes}")

    if df[FEATURE_COLUMNS + [TARGET_COLUMN]].isnull().any().any():
        raise DatasetRetrievalError("Unexpected missing values found in raw dataset.")

    return True


def load_cached_metadata(cfg):
    meta_path = project_path(cfg["data"]["raw_metadata_path"])
    return load_json(meta_path)
