"""
Experiment registry.

Every model is declared exactly once here. Training scripts iterate the registry
instead of hard-coding model lists, so adding or disabling a model is a one-line
change and every model is treated identically by the evaluation code.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from sklearn.dummy import DummyClassifier
from sklearn.ensemble import (
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
    StackingClassifier,
    VotingClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

import config

SEED = config.RANDOM_STATE


@dataclass
class ModelSpec:
    """Declarative description of one experiment."""
    name: str
    family: str                       # baseline | classical | ensemble
    factory: Callable[[], object]     # returns an unfitted estimator
    needs_scaling: bool = False
    search_space: dict = field(default_factory=dict)
    tune: bool = False                # include in hyperparameter search
    cross_validate: bool = False      # include in stratified CV
    enabled_fast: bool = True
    note: str = ""


def _xgb_factory():
    """Imported lazily so a missing libomp fails one model, not the whole run."""
    from xgboost import XGBClassifier
    return XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.9,
        colsample_bytree=0.9,
        tree_method="hist",
        eval_metric="logloss",
        random_state=SEED,
        n_jobs=-1,
    )


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------
REGISTRY: list[ModelSpec] = [
    ModelSpec(
        name="Dummy Baseline",
        family="baseline",
        factory=lambda: DummyClassifier(strategy="most_frequent", random_state=SEED),
        note="Majority-class floor: establishes whether real models learn anything.",
    ),
    ModelSpec(
        name="Logistic Regression",
        family="classical",
        factory=lambda: LogisticRegression(max_iter=2000, random_state=SEED),
        needs_scaling=True,
        cross_validate=True,
        note="Linear, highly interpretable reference model.",
    ),
    ModelSpec(
        name="Decision Tree",
        family="classical",
        factory=lambda: DecisionTreeClassifier(max_depth=12, random_state=SEED),
        cross_validate=True,
        note="Single interpretable tree; depth capped to limit overfitting.",
    ),
    ModelSpec(
        name="Random Forest",
        family="ensemble",
        factory=lambda: RandomForestClassifier(
            n_estimators=300, random_state=SEED, n_jobs=-1
        ),
        search_space={
            "model__n_estimators": [200, 300, 500],
            "model__max_depth": [None, 12, 20, 30],
            "model__min_samples_split": [2, 5, 10],
            "model__min_samples_leaf": [1, 2, 4],
            "model__max_features": ["sqrt", "log2", 0.5],
        },
        tune=True,
        cross_validate=True,
        note="Bagged trees; the reference paper's best model.",
    ),
    ModelSpec(
        name="Support Vector Machine",
        family="classical",
        factory=lambda: SVC(kernel="rbf", random_state=SEED),
        needs_scaling=True,
        note="RBF kernel. Probability calibration is omitted (deprecated in sklearn 1.9 and "
             "costly to fit); decision_function supplies the scores used for ROC-AUC.",
    ),
    ModelSpec(
        name="K-Nearest Neighbors",
        family="classical",
        factory=lambda: KNeighborsClassifier(n_neighbors=15, n_jobs=-1),
        needs_scaling=True,
        note="Distance-based; requires scaling to be meaningful.",
    ),
    ModelSpec(
        name="XGBoost",
        family="ensemble",
        factory=_xgb_factory,
        search_space={
            "model__n_estimators": [200, 400, 600],
            "model__max_depth": [4, 6, 8, 10],
            "model__learning_rate": [0.03, 0.1, 0.2],
            "model__subsample": [0.7, 0.85, 1.0],
            "model__colsample_bytree": [0.7, 0.85, 1.0],
        },
        tune=True,
        cross_validate=True,
        note="Gradient boosted trees; requires OpenMP (brew install libomp on macOS).",
    ),
    ModelSpec(
        name="Gradient Boosting",
        family="ensemble",
        factory=lambda: GradientBoostingClassifier(n_estimators=150, random_state=SEED),
        note="Classical sklearn boosting; slower than XGBoost but a useful comparison.",
    ),
    ModelSpec(
        name="Extra Trees",
        family="ensemble",
        factory=lambda: ExtraTreesClassifier(
            n_estimators=300, random_state=SEED, n_jobs=-1
        ),
        cross_validate=True,
        note="Extremely randomised trees; higher variance reduction than RF.",
    ),
]


def _voting_factory():
    from xgboost import XGBClassifier
    return VotingClassifier(
        estimators=[
            ("rf", RandomForestClassifier(n_estimators=200, random_state=SEED, n_jobs=-1)),
            ("xgb", XGBClassifier(
                n_estimators=300, max_depth=6, learning_rate=0.1, tree_method="hist",
                eval_metric="logloss", random_state=SEED, n_jobs=-1)),
            ("et", ExtraTreesClassifier(n_estimators=200, random_state=SEED, n_jobs=-1)),
        ],
        voting="soft",
        n_jobs=1,  # inner estimators already parallel; avoid oversubscription
    )


def _stacking_factory():
    from xgboost import XGBClassifier
    return StackingClassifier(
        estimators=[
            ("rf", RandomForestClassifier(n_estimators=200, random_state=SEED, n_jobs=-1)),
            ("xgb", XGBClassifier(
                n_estimators=300, max_depth=6, learning_rate=0.1, tree_method="hist",
                eval_metric="logloss", random_state=SEED, n_jobs=-1)),
        ],
        final_estimator=LogisticRegression(max_iter=1000, random_state=SEED),
        cv=3,
        n_jobs=1,
    )


REGISTRY += [
    ModelSpec(
        name="Voting Ensemble",
        family="ensemble",
        factory=_voting_factory,
        enabled_fast=False,
        note="Soft voting over RF + XGBoost + Extra Trees.",
    ),
    ModelSpec(
        name="Stacking Ensemble",
        family="ensemble",
        factory=_stacking_factory,
        enabled_fast=False,
        note="RF + XGBoost base learners with a logistic-regression meta-learner.",
    ),
]


def get_specs(mode: str) -> list[ModelSpec]:
    """Return the model specs enabled for the given mode."""
    if mode == "fast":
        return [s for s in REGISTRY if s.enabled_fast]
    return list(REGISTRY)
