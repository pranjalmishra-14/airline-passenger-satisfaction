"""Preprocessing pipeline behaviour and leakage protection."""
import numpy as np
import pytest

from src.preprocessing import build_preprocessor, get_feature_names
from src.feature_engineering import FeatureEngineer, detect_rating_columns


@pytest.mark.parametrize("scale", [True, False])
def test_pipeline_transforms(splits, scale):
    pipe = build_preprocessor(splits.X_train, scale=scale)
    Xt = pipe.fit_transform(splits.X_train)
    assert Xt.shape[0] == len(splits.X_train)
    assert Xt.shape[1] >= splits.X_train.shape[1]
    assert not np.isnan(Xt).any(), "preprocessing must leave no NaNs"


def test_val_and_test_transform_consistently(splits):
    pipe = build_preprocessor(splits.X_train, scale=True)
    pipe.fit(splits.X_train)
    a = pipe.transform(splits.X_val)
    b = pipe.transform(splits.X_test)
    assert a.shape[1] == b.shape[1]


def test_scaling_fitted_on_train_only(splits):
    """
    The scaler's statistics must come from training data alone.

    Fitting on train produces mean~0 on train; the validation set will NOT be
    exactly centred, which is the expected, leakage-free behaviour.
    """
    pipe = build_preprocessor(splits.X_train, scale=True)
    Xtr = pipe.fit_transform(splits.X_train)
    Xva = pipe.transform(splits.X_val)
    assert abs(float(Xtr[:, :4].mean())) < 1e-6
    assert abs(float(Xva[:, :4].mean())) > 0  # not re-centred on itself


def test_feature_engineering_adds_documented_features(splits):
    fe = FeatureEngineer().fit(splits.X_train)
    out = fe.transform(splits.X_train)
    assert "Total Delay" in out.columns
    assert "Service Quality Score" in out.columns
    assert "Flight Distance Category" in out.columns


def test_engineered_features_are_row_wise(splits):
    """
    Derived features must depend only on the row itself.

    Transforming a subset must give identical values to transforming the whole
    frame -- proving no cross-row (and therefore no cross-split) information use.
    """
    fe = FeatureEngineer().fit(splits.X_train)
    full = fe.transform(splits.X_train)
    subset = fe.transform(splits.X_train.head(50))
    np.testing.assert_allclose(
        full["Service Quality Score"].head(50).to_numpy(),
        subset["Service Quality Score"].to_numpy(),
    )


def test_rating_columns_detected(splits):
    ratings = detect_rating_columns(splits.X_train)
    assert len(ratings) > 0
    for c in ratings:
        assert splits.X_train[c].dropna().between(0, 5).all()


def test_feature_names_match_width(splits):
    pipe = build_preprocessor(splits.X_train, scale=False)
    Xt = pipe.fit_transform(splits.X_train)
    assert len(get_feature_names(pipe)) == Xt.shape[1]
