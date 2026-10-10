"""Shared helpers: config loading, paths, seeding, timing, JSON/CSV IO."""
import json
import os
import random
import time
import platform
import sys
from datetime import datetime, timezone

import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def project_path(*parts):
    return os.path.join(PROJECT_ROOT, *parts)


def load_config(config_path=None):
    path = config_path or project_path("config.json")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def results_dir(cfg):
    sub = "results_quick" if cfg["quick_mode"] else "results_full"
    return project_path(cfg["paths"][sub])


def set_global_seed(seed):
    random.seed(seed)
    np.random.seed(seed)


def now_iso():
    return datetime.now(timezone.utc).isoformat()


class Timer:
    def __enter__(self):
        self.start = time.perf_counter()
        return self

    def __exit__(self, *exc):
        self.elapsed = time.perf_counter() - self.start


def environment_snapshot():
    import numpy, scipy, pandas, sklearn, matplotlib
    try:
        import pennylane
        pl_version = pennylane.__version__
    except Exception:
        pl_version = "not available"
    return {
        "python_version": sys.version,
        "platform": platform.platform(),
        "numpy": numpy.__version__,
        "scipy": scipy.__version__,
        "pandas": pandas.__version__,
        "scikit_learn": sklearn.__version__,
        "matplotlib": matplotlib.__version__,
        "pennylane": pl_version,
        "captured_at": now_iso(),
    }


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)
    return path


def save_json(obj, path):
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=_json_default)


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f"Object of type {type(o)} is not JSON serializable")
