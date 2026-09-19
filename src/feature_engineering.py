"""
Derived features.

Every feature here is justified on domain grounds and documented in the report.
None is derived from the target, so no leakage is introduced. The transformer is
stateless (it computes row-wise quantities only), which means applying it to
train/val/test cannot leak information between splits.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

# Documented rationale, surfaced in the report and the data-quality appendix.
FEATURE_RATIONALE = {
    "Total Delay": (
        "Departure Delay + Arrival Delay. Passengers experience cumulative disruption, "
        "not two independent delays; the sum captures total time lost."
    ),
    "Service Quality Score": (
        "Mean of the service-rating columns. A single summary of perceived service "
        "quality, which the EDA shows is the dominant satisfaction signal."
    ),
    "Flight Distance Category": (
        "Short/Medium/Long haul bucket. Service expectations differ by haul length, "
        "and the effect of distance on satisfaction is non-linear."
    ),
}


def detect_rating_columns(df: pd.DataFrame) -> list[str]:
    """Service-rating columns: integer-like, values within 0-5, at most 6 levels."""
    out = []
    for c in df.columns:
        s = df[c]
        if not pd.api.types.is_numeric_dtype(s):
            continue
        vals = s.dropna()
        if vals.empty:
            continue
        if vals.between(0, 5).all() and vals.nunique() <= 6 and float(vals.max()) <= 5:
            out.append(c)
    return out


def _find(df: pd.DataFrame, *keywords: str) -> str | None:
    """Find a column whose lowercased name contains all keywords."""
    for c in df.columns:
        low = str(c).lower()
        if all(k in low for k in keywords):
            return c
    return None


class FeatureEngineer(BaseEstimator, TransformerMixin):
    """Adds the derived features above. Stateless: fit() only records schema."""

    def __init__(self, enabled: bool = True):
        self.enabled = enabled

    def fit(self, X: pd.DataFrame, y=None):
        self.rating_cols_ = detect_rating_columns(X)
        self.dep_delay_ = _find(X, "departure", "delay")
        self.arr_delay_ = _find(X, "arrival", "delay")
        self.distance_ = _find(X, "distance")
        self.feature_names_in_ = list(X.columns)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        if not self.enabled:
            return X

        if self.dep_delay_ and self.arr_delay_:
            X["Total Delay"] = (
                X[self.dep_delay_].fillna(0) + X[self.arr_delay_].fillna(0)
            )

        if self.rating_cols_:
            X["Service Quality Score"] = X[self.rating_cols_].mean(axis=1)

        if self.distance_:
            # Fixed, domain-based cut points -- not learned from data, so no leakage.
            X["Flight Distance Category"] = pd.cut(
                X[self.distance_],
                bins=[-np.inf, 800, 2200, np.inf],
                labels=["Short", "Medium", "Long"],
            ).astype(str)

        return X

    def get_feature_names_out(self, input_features=None):
        return np.asarray(self.feature_names_in_, dtype=object)
