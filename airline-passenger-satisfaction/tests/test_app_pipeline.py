"""
Saved-artifact tests: exactly the path the Streamlit app takes.

These run against the artifacts produced by train.py. If training has not been
run, they skip rather than fail.
"""
import json

import numpy as np
import pandas as pd
import pytest

import config

pytestmark = pytest.mark.skipif(
    not config.SCHEMA_PATH.exists(),
    reason="run `python train.py --mode full` first",
)


@pytest.fixture(scope="module")
def schema():
    return json.loads(config.SCHEMA_PATH.read_text())


def test_schema_is_complete(schema):
    assert schema["raw_features"], "schema must list the model's raw features"
    assert schema["final_model"]
    assert schema["positive_label"] == config.POSITIVE_LABEL
    for f in schema["raw_features"]:
        assert f["kind"] in {"numeric", "categorical"}
        if f["kind"] == "categorical":
            assert f["categories"], f"{f['name']} must list its categories for the UI"
        else:
            assert f["median"] is not None


def _row_from_schema(schema) -> pd.DataFrame:
    """Build one valid raw input row the way the Streamlit form does."""
    values = {}
    for f in schema["raw_features"]:
        if f["kind"] == "categorical":
            values[f["name"]] = f["categories"][0]
        else:
            values[f["name"]] = f["median"]
    return pd.DataFrame([values])[[f["name"] for f in schema["raw_features"]]]


def test_saved_sklearn_model_predicts(schema):
    """raw dict -> DataFrame -> saved pipeline -> probability."""
    import joblib
    path = config.MODELS_DIR / "random_forest.pkl"
    if not path.exists():
        pytest.skip("random_forest.pkl not present")
    pipe = joblib.load(path)
    row = _row_from_schema(schema)

    pred = pipe.predict(row)
    assert pred.shape == (1,)
    assert int(pred[0]) in (0, 1)

    proba = pipe.predict_proba(row)
    assert proba.shape == (1, 2)
    assert abs(float(proba.sum()) - 1.0) < 1e-6


def test_saved_keras_model_predicts(schema):
    """The neural-network path: preprocessor + .keras model reload and predict."""
    import joblib
    pre_path = config.MODELS_DIR / "dl_preprocessor.pkl"
    keras_files = sorted(config.MODELS_DIR.glob("neural_net_*.keras"))
    if not pre_path.exists() or not keras_files:
        pytest.skip("deep learning artifacts not present")

    import keras
    pre = joblib.load(pre_path)
    model = keras.models.load_model(keras_files[0])

    X = pre.transform(_row_from_schema(schema)).astype("float32")
    prob = float(model.predict(X, verbose=0).ravel()[0])
    assert 0.0 <= prob <= 1.0


def test_final_model_artifact_exists(schema):
    """best_model.pkl must exist AND point at artifacts that are really on disk."""
    import joblib
    assert config.FINAL_MODEL_PATH.exists(), "best_model.pkl must be saved for the app"

    obj = joblib.load(config.FINAL_MODEL_PATH)
    if isinstance(obj, dict) and obj.get("kind") == "keras":
        # Neural network winner: the pointer's targets must exist.
        assert (config.MODELS_DIR / obj["model_file"]).exists(), obj["model_file"]
        assert (config.MODELS_DIR / obj["preprocessor_file"]).exists()
        assert obj["model_name"] == schema["final_model"]
    else:
        # sklearn winner: must be a usable fitted pipeline.
        assert hasattr(obj, "predict")


def test_final_model_matches_best_result(schema):
    """The saved final model must be the top-ranked model in the results table."""
    df = pd.read_csv(config.RESULTS_CSV)
    ok = df[df["Status"] == "ok"].sort_values("F1", ascending=False)
    assert schema["final_model"] == ok.iloc[0]["Model"]


def test_results_csv_is_consistent():
    assert config.RESULTS_CSV.exists()
    df = pd.read_csv(config.RESULTS_CSV)
    assert {"Model", "Family", "Accuracy", "F1", "Status"} <= set(df.columns)
    ok = df[df["Status"] == "ok"]
    assert len(ok) >= 5
    for col in ("Accuracy", "Precision", "Recall", "F1", "ROC-AUC"):
        vals = ok[col].dropna()
        assert vals.between(0, 1).all(), f"{col} outside [0,1]"


def test_dummy_baseline_is_present_and_weakest():
    """The baseline must be reported, and real models must beat it."""
    df = pd.read_csv(config.RESULTS_CSV)
    ok = df[df["Status"] == "ok"]
    dummy = ok[ok["Model"] == "Dummy Baseline"]
    if dummy.empty:
        pytest.skip("dummy not in results")
    assert float(dummy["F1"].iloc[0]) < float(ok["F1"].max())


def test_manifest_records_protocol():
    assert config.MANIFEST_JSON.exists()
    m = json.loads(config.MANIFEST_JSON.read_text())
    assert m["seed"] == config.RANDOM_STATE
    assert {"train", "val", "test"} <= set(m["split"])
    assert m["library_versions"]
    assert m["final_model"]


def test_no_fabricated_metrics_in_results():
    """Every reported metric must be a real number in range, not a placeholder."""
    df = pd.read_csv(config.RESULTS_CSV)
    ok = df[df["Status"] == "ok"]
    assert not ok[["Accuracy", "F1"]].isna().any().any()
    # A perfect score would indicate leakage rather than a good model.
    assert float(ok["Accuracy"].max()) < 1.0, "accuracy of exactly 1.0 suggests leakage"
