"""
Classical and ensemble ML training.

Each registry spec is wrapped in a Pipeline(preprocessor -> model) so that all
preprocessing is fitted on training data only. A model that fails is caught,
recorded with its traceback, and reported as FAILED -- never silently skipped.
"""
from __future__ import annotations

import time
import traceback

import joblib
import numpy as np
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline

import config
from src import evaluate as ev
from src import utils
from src.preprocessing import build_preprocessor
from src.registry import ModelSpec, get_specs


def build_pipeline(spec: ModelSpec, X_train) -> Pipeline:
    """Preprocessor + estimator, with scaling applied only where it matters."""
    pre = build_preprocessor(X_train, scale=spec.needs_scaling)
    return Pipeline([("prep", pre), ("model", spec.factory())])


def _slug(name: str) -> str:
    return name.lower().replace(" ", "_").replace("-", "_")


def train_one(spec: ModelSpec, splits, budgets: dict, manifest) -> dict | None:
    """
    Train, optionally tune, and evaluate one model on the VALIDATION set.

    The test set is not touched here -- final test evaluation happens once, later,
    in train.py after the final model has been selected.
    """
    pipe = build_pipeline(spec, splits.X_train)
    record: dict = {"Model": spec.name, "Family": spec.family, "Note": spec.note}

    t0 = time.perf_counter()
    best_params = None

    if spec.tune and spec.search_space:
        utils.info(
            f"tuning ({budgets['search_iter']} candidates x {budgets['search_cv']} folds, train only)"
        )
        search = RandomizedSearchCV(
            pipe,
            param_distributions=spec.search_space,
            n_iter=budgets["search_iter"],
            cv=budgets["search_cv"],
            scoring="f1",
            n_jobs=-1,
            random_state=config.RANDOM_STATE,
            error_score="raise",
        )
        search.fit(splits.X_train, splits.y_train)
        pipe = search.best_estimator_
        best_params = {k: _jsonable(v) for k, v in search.best_params_.items()}
        utils.info(f"best CV F1 during search: {search.best_score_:.4f}")
    else:
        pipe.fit(splits.X_train, splits.y_train)

    fit_seconds = time.perf_counter() - t0

    # Model selection uses the VALIDATION set.
    t1 = time.perf_counter()
    metrics, y_pred, y_score = ev.evaluate_model(pipe, splits.X_val, splits.y_val)
    predict_seconds = time.perf_counter() - t1

    record.update(metrics)
    record["Fit (s)"] = round(fit_seconds, 2)
    record["Predict (s)"] = round(predict_seconds, 2)
    record["Status"] = "ok"

    if spec.cross_validate and budgets["cv_folds"]:
        utils.info(f"{budgets['cv_folds']}-fold CV on train")
        fresh = build_pipeline(spec, splits.X_train)
        cv = ev.cross_validate_model(fresh, splits.X_train, splits.y_train, budgets["cv_folds"])
        record.update({
            "CV Acc Mean": round(cv["cv_accuracy_mean"], 4),
            "CV Acc Std": round(cv["cv_accuracy_std"], 4),
            "CV F1 Mean": round(cv["cv_f1_mean"], 4),
            "CV F1 Std": round(cv["cv_f1_std"], 4),
        })

    path = config.MODELS_DIR / f"{_slug(spec.name)}.pkl"
    joblib.dump(pipe, path, compress=3)  # keeps artifacts deployable (see README)

    manifest.record_model(
        spec.name,
        family=spec.family,
        fit_seconds=round(fit_seconds, 2),
        predict_seconds=round(predict_seconds, 2),
        best_params=best_params,
        val_metrics={k: round(float(v), 4) for k, v in metrics.items() if not np.isnan(v)},
        saved_to=str(path.relative_to(config.ROOT)),
        status="ok",
    )
    utils.info(
        f"val acc {metrics['Accuracy']:.4f} | F1 {metrics['F1']:.4f} | "
        f"AUC {metrics['ROC-AUC']:.4f} | {fit_seconds:.1f}s"
    )
    return record


def _jsonable(v):
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return float(v)
    return v


def train_all(splits, mode: str, manifest) -> list[dict]:
    """Train every model enabled for this mode. Failures are recorded, not hidden."""
    budgets = config.MODE_BUDGETS[mode]
    specs = get_specs(mode)
    rows: list[dict] = []

    for spec in specs:
        utils.step(f"Training {spec.name}...")
        try:
            row = train_one(spec, splits, budgets, manifest)
            if row:
                rows.append(row)
        except Exception as exc:  # noqa: BLE001 -- we want every failure reported
            tb = traceback.format_exc()
            utils.warn(f"{spec.name} FAILED: {type(exc).__name__}: {exc}")
            manifest.record_failure(spec.name, f"{type(exc).__name__}: {exc}", tb)
            rows.append({
                "Model": spec.name,
                "Family": spec.family,
                "Status": f"FAILED: {type(exc).__name__}: {exc}",
                "Note": spec.note,
            })
    return rows
