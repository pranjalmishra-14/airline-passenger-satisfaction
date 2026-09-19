"""
Evaluation: metrics, confusion matrices, ROC/PR curves, CV and bootstrap CIs.

Positive class (1) = "Satisfied"; negative class (0) = "Neutral or Dissatisfied".
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    auc,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, cross_validate

import config

METRIC_COLUMNS = ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC"]


def get_scores(model, X) -> np.ndarray | None:
    """
    Continuous scores for ROC/PR.

    Uses predict_proba when available, else decision_function (e.g. SVC with
    probability=False). Returns None if the model exposes neither.
    """
    if hasattr(model, "predict_proba"):
        try:
            return model.predict_proba(X)[:, 1]
        except Exception:
            pass
    if hasattr(model, "decision_function"):
        try:
            return model.decision_function(X)
        except Exception:
            pass
    return None


def compute_metrics(y_true, y_pred, y_score=None) -> dict:
    """Standard binary-classification metrics. ROC-AUC omitted if no scores."""
    out = {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1": f1_score(y_true, y_pred, zero_division=0),
    }
    out["ROC-AUC"] = roc_auc_score(y_true, y_score) if y_score is not None else np.nan
    return out


def evaluate_model(model, X, y) -> tuple[dict, np.ndarray, np.ndarray | None]:
    """Return (metrics, predictions, scores) for a fitted model."""
    y_pred = model.predict(X)
    y_score = get_scores(model, X)
    return compute_metrics(y, y_pred, y_score), y_pred, y_score


def confusion_frame(y_true, y_pred) -> pd.DataFrame:
    """Labelled 2x2 confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    return pd.DataFrame(
        cm,
        index=[f"Actual: {config.NEGATIVE_LABEL}", f"Actual: {config.POSITIVE_LABEL}"],
        columns=[f"Pred: {config.NEGATIVE_LABEL}", f"Pred: {config.POSITIVE_LABEL}"],
    )


def text_report(y_true, y_pred) -> str:
    return classification_report(
        y_true, y_pred,
        target_names=[config.NEGATIVE_LABEL, config.POSITIVE_LABEL],
        digits=4,
        zero_division=0,
    )


# ---------------------------------------------------------------------------
# Cross-validation -- TRAIN ONLY. The test set is never passed to this function.
# ---------------------------------------------------------------------------
def cross_validate_model(pipeline, X_train, y_train, folds: int, n_jobs: int = -1) -> dict:
    """Stratified k-fold CV on the training set to gauge stability."""
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=config.RANDOM_STATE)
    res = cross_validate(
        pipeline, X_train, y_train,
        cv=cv, scoring=["accuracy", "f1"], n_jobs=n_jobs, error_score="raise",
    )
    return {
        "cv_folds": folds,
        "cv_accuracy_mean": float(res["test_accuracy"].mean()),
        "cv_accuracy_std": float(res["test_accuracy"].std()),
        "cv_f1_mean": float(res["test_f1"].mean()),
        "cv_f1_std": float(res["test_f1"].std()),
    }


# ---------------------------------------------------------------------------
# Bootstrap confidence intervals on test metrics
# ---------------------------------------------------------------------------
def bootstrap_ci(y_true, y_pred, metric: str = "f1", n: int = 1000, alpha: float = 0.05) -> dict:
    """
    Percentile bootstrap CI for a test metric.

    Used to judge whether the gap between top models is meaningful, without
    overclaiming statistical significance.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    fn = f1_score if metric == "f1" else accuracy_score
    rng = np.random.default_rng(config.RANDOM_STATE)
    idx = np.arange(len(y_true))
    stats = [
        fn(y_true[s], y_pred[s], **({"zero_division": 0} if metric == "f1" else {}))
        for s in (rng.choice(idx, size=len(idx), replace=True) for _ in range(n))
    ]
    lo, hi = np.percentile(stats, [alpha / 2 * 100, (1 - alpha / 2) * 100])
    return {
        f"{metric}_ci_low": float(lo),
        f"{metric}_ci_high": float(hi),
        f"{metric}_ci_width": float(hi - lo),
    }


def roc_points(y_true, y_score):
    fpr, tpr, _ = roc_curve(y_true, y_score)
    return fpr, tpr, auc(fpr, tpr)


def pr_points(y_true, y_score):
    prec, rec, _ = precision_recall_curve(y_true, y_score)
    return rec, prec, auc(rec, prec)


def results_table(rows: list[dict]) -> pd.DataFrame:
    """Consolidated comparison table, sorted by F1 (best first)."""
    df = pd.DataFrame(rows)
    if "F1" in df.columns:
        df = df.sort_values("F1", ascending=False, na_position="last")
    return df.reset_index(drop=True)
