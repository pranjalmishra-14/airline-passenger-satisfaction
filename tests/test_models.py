"""Training, saved-model round-trip, and metric sanity."""
import numpy as np
import pytest

import config
from src import evaluate as ev
from src import train_ml
from src.registry import REGISTRY, get_specs


def test_registry_is_well_formed():
    names = [s.name for s in REGISTRY]
    assert len(names) == len(set(names)), "model names must be unique"
    for s in REGISTRY:
        assert s.family in {"baseline", "classical", "ensemble"}
        assert callable(s.factory)
        assert s.note, f"{s.name} must document its purpose"


def test_fast_mode_is_subset_of_full():
    assert {s.name for s in get_specs("fast")} <= {s.name for s in get_specs("full")}


def test_train_and_predict_roundtrip(splits, tmp_path):
    """raw input -> preprocessing -> model -> prediction must work end to end."""
    spec = next(s for s in REGISTRY if s.name == "Decision Tree")
    pipe = train_ml.build_pipeline(spec, splits.X_train)
    pipe.fit(splits.X_train, splits.y_train)

    preds = pipe.predict(splits.X_val)
    assert len(preds) == len(splits.X_val)
    assert set(np.unique(preds)) <= {0, 1}

    import joblib
    p = tmp_path / "m.pkl"
    joblib.dump(pipe, p)
    reloaded = joblib.load(p)
    np.testing.assert_array_equal(preds, reloaded.predict(splits.X_val))


def test_single_row_prediction(splits):
    """The app path: one raw row in, one probability out."""
    spec = next(s for s in REGISTRY if s.name == "Decision Tree")
    pipe = train_ml.build_pipeline(spec, splits.X_train)
    pipe.fit(splits.X_train, splits.y_train)

    row = splits.X_val.head(1)
    assert pipe.predict(row).shape == (1,)
    proba = pipe.predict_proba(row)
    assert proba.shape == (1, 2)
    assert 0.0 <= float(proba[0, 1]) <= 1.0
    assert abs(float(proba[0].sum()) - 1.0) < 1e-6


def test_model_beats_dummy_baseline(splits):
    """A real model must outperform the majority-class floor."""
    dummy_spec = next(s for s in REGISTRY if s.name == "Dummy Baseline")
    tree_spec = next(s for s in REGISTRY if s.name == "Decision Tree")

    dummy = train_ml.build_pipeline(dummy_spec, splits.X_train)
    dummy.fit(splits.X_train, splits.y_train)
    tree = train_ml.build_pipeline(tree_spec, splits.X_train)
    tree.fit(splits.X_train, splits.y_train)

    d = ev.compute_metrics(splits.y_val, dummy.predict(splits.X_val))
    t = ev.compute_metrics(splits.y_val, tree.predict(splits.X_val))
    assert t["Accuracy"] > d["Accuracy"]
    assert t["F1"] > d["F1"]


def test_metrics_in_valid_range(splits):
    spec = next(s for s in REGISTRY if s.name == "Decision Tree")
    pipe = train_ml.build_pipeline(spec, splits.X_train)
    pipe.fit(splits.X_train, splits.y_train)
    m, _, score = ev.evaluate_model(pipe, splits.X_val, splits.y_val)
    for k, v in m.items():
        assert 0.0 <= v <= 1.0, f"{k} out of range: {v}"


def test_bootstrap_ci_brackets_estimate():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 400)
    p = y.copy()
    p[:40] = 1 - p[:40]
    from sklearn.metrics import f1_score
    point = f1_score(y, p)
    ci = ev.bootstrap_ci(y, p, "f1", n=200)
    assert ci["f1_ci_low"] <= point <= ci["f1_ci_high"]
