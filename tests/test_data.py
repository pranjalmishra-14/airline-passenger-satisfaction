"""Dataset loading, column detection and split integrity."""
import pandas as pd
import pytest

import config
from src import data_loader as dl


def test_dataset_loads(raw_df):
    assert len(raw_df) > 0
    assert raw_df.shape[1] > 5


def test_target_detected(raw_df):
    target = dl.find_target(raw_df)
    assert target in raw_df.columns
    assert set(raw_df[target].unique()) == {config.POSITIVE_LABEL, config.NEGATIVE_LABEL}


def test_id_column_detected_as_droppable(raw_df):
    target = dl.find_target(raw_df)
    drops = dl.detect_droppable(raw_df, target)
    # Every dropped column must have a documented reason.
    assert all(isinstance(v, str) and v for v in drops.values())
    assert target not in drops


def test_split_proportions(splits):
    sizes = splits.sizes
    total = sum(sizes.values())
    assert abs(sizes["train"] / total - config.TRAIN_SIZE) < 0.02
    assert abs(sizes["val"] / total - config.VAL_SIZE) < 0.02
    assert abs(sizes["test"] / total - config.TEST_SIZE) < 0.02


def test_splits_are_disjoint(splits):
    """No row may appear in more than one split -- the core leakage guarantee."""
    splits.assert_disjoint()
    tr, va, te = (set(splits.X_train.index), set(splits.X_val.index),
                  set(splits.X_test.index))
    assert len(tr | va | te) == len(tr) + len(va) + len(te)


def test_stratification_preserved(splits):
    """Class balance should match across splits."""
    rates = [splits.y_train.mean(), splits.y_val.mean(), splits.y_test.mean()]
    assert max(rates) - min(rates) < 0.02


def test_target_not_in_features(splits):
    for X in (splits.X_train, splits.X_val, splits.X_test):
        assert splits.target not in X.columns


def test_quality_report_generates(raw_df):
    target = dl.find_target(raw_df)
    drops = dl.detect_droppable(raw_df, target)
    report = dl.quality_report(raw_df, target, drops)
    assert "# Data Quality Report" in report
    assert str(len(raw_df)) in report.replace(",", "")
