"""
Preprocessing pipelines.

Critical property: every transformer lives INSIDE an sklearn Pipeline, so it is
fitted on the training fold only. Scalers, imputers and encoders therefore never
see validation or test data -- this is what prevents leakage in practice, rather
than relying on convention.
"""
from __future__ import annotations

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.feature_engineering import FeatureEngineer


def split_column_types(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    """Return (numeric_columns, categorical_columns) for the given frame."""
    num = df.select_dtypes(include="number").columns.tolist()
    cat = df.select_dtypes(exclude="number").columns.tolist()
    return num, cat


def _ohe() -> OneHotEncoder:
    """OneHotEncoder that tolerates unseen categories at inference time."""
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:  # older sklearn
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def build_preprocessor(X: pd.DataFrame, scale: bool) -> Pipeline:
    """
    Build the preprocessing pipeline.

    scale=True  -> for Logistic Regression, SVM, KNN and neural networks
    scale=False -> for tree-based models, which are scale-invariant
    """
    fe = FeatureEngineer()
    engineered = fe.fit(X).transform(X.head(50))
    num_cols, cat_cols = split_column_types(engineered)

    num_steps = [("impute", SimpleImputer(strategy="median"))]
    if scale:
        num_steps.append(("scale", StandardScaler()))

    ct = ColumnTransformer(
        transformers=[
            ("num", Pipeline(num_steps), num_cols),
            (
                "cat",
                Pipeline([
                    ("impute", SimpleImputer(strategy="most_frequent")),
                    ("onehot", _ohe()),
                ]),
                cat_cols,
            ),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    return Pipeline([("engineer", FeatureEngineer()), ("columns", ct)])


def get_feature_names(fitted_pipeline: Pipeline) -> list[str]:
    """Recover output feature names from a fitted preprocessing pipeline."""
    ct = fitted_pipeline.named_steps["columns"]
    try:
        return [str(n) for n in ct.get_feature_names_out()]
    except Exception:
        return [f"f{i}" for i in range(ct.transform(ct._df_sample).shape[1])]
