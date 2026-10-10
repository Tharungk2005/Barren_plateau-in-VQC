import os
import pandas as pd

from src import data_retrieval as dr
from src import preprocessing as pp
from src import experiment_runner as ER


def test_run_one_handles_failure_gracefully(cfg, monkeypatch):
    df = dr.fetch_raw_dataset(cfg)
    y = pp.encode_labels(df, cfg)
    data = pp.prepare_dataset_for_qubits(df, y, 4, cfg, seed=0)

    def broken_direct(*args, **kwargs):
        raise RuntimeError("synthetic failure for test")

    monkeypatch.setattr(ER.T, "direct_training_run", broken_direct)
    row, ok = ER._run_one(4, 2, "direct", 0, data, cfg, convergence_target=0.5, n_updates=5)
    assert ok is False
    assert row["status"] == "failed"
    assert "synthetic failure" in row["error_message"]


def test_resumability_skips_completed_runs(cfg, tmp_path, monkeypatch):
    monkeypatch.setattr(ER, "project_path", lambda *p: str(tmp_path.joinpath(*p)))
    df = dr.fetch_raw_dataset(cfg)
    y = pp.encode_labels(df, cfg)

    cfg_local = dict(cfg)
    grid = {"qubit_counts": [4], "depths": [2], "protocols": ["direct"], "seeds": [0]}

    path1 = ER.run_main_grid(cfg_local, df, y, convergence_target=0.5, grid_override=grid)
    d1 = pd.read_csv(path1)
    assert len(d1) == 1

    path2 = ER.run_main_grid(cfg_local, df, y, convergence_target=0.5, grid_override=grid)
    d2 = pd.read_csv(path2)
    assert len(d2) == 1  # unchanged: the run was skipped, not duplicated
