import os
import sys
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest

import config
from src import data_loader as dl


@pytest.fixture(scope="session")
def raw_df():
    if not config.DATA_FILE.exists():
        pytest.skip("dataset not present")
    return dl.load_raw()


@pytest.fixture(scope="session")
def splits(raw_df):
    """Small stratified split -- fast enough to use across the whole test session."""
    return dl.make_splits(raw_df, sample_rows=4000)
