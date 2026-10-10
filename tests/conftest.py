import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from src.utils import load_config


@pytest.fixture(scope="session")
def cfg():
    c = load_config()
    c["quick_mode"] = True
    return c
