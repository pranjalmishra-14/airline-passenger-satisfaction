"""
Explainable AI: tree feature importance, SHAP, and permutation importance.

IMPORTANT FRAMING: SHAP and permutation importance describe *model behaviour* --
how the fitted model uses each feature. They are associational, not causal. The
report and the app phrase findings accordingly ("the model placed substantial
predictive weight on X", never "X caused satisfaction").
"""
from __future__ import annotations

import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import config
from src import utils
from src.preprocessing import get_feature_names

SHAP_DISCLAIMER = (
    "SHAP values explain how the fitted model uses each feature. They describe "
    "model behaviour and association, NOT causation."
)


def _transform(pipeline, X):
    """Run X through the fitted preprocessing steps of a model pipeline."""
    return pipeline.named_steps["prep"].transform(X)


def tree_feature_importance(pipelines: dict, X_ref) -> pd.DataFrame:
    """Impurity-based importance for every tree model exposing feature_importances_."""
    rows = []
    for name, pipe in pipelines.items():
        model = pipe.named_steps.get("model")
        if model is None or not hasattr(model, "feature_importances_"):
            continue
        try:
            names = get_feature_names(pipe.named_steps["prep"])
            imps = np.asarray(model.feature_importances_, dtype=float)
            if len(names) != len(imps):
                names = [f"f{i}" for i in range(len(imps))]
            for f, v in zip(names, imps):
                rows.append({"Model": name, "Feature": f, "Importance": float(v)})
        except Exception as exc:  # noqa: BLE001
            utils.warn(f"feature importance unavailable for {name}: {exc}")
    return pd.DataFrame(rows)


def shap_analysis(pipeline, X_sample, model_name: str, max_display: int = 18) -> dict:
    """
    TreeExplainer SHAP analysis for a tree-based pipeline.

    Returns a summary dict; writes summary/bar/waterfall plots to outputs/shap_results/.
    """
    import shap

    def _shap_plot(fn, *a, **kw):
        """Call a shap plotting fn, suppressing its internal NumPy RNG FutureWarning."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            return fn(*a, **kw)

    out: dict = {"model": model_name, "disclaimer": SHAP_DISCLAIMER}
    model = pipeline.named_steps["model"]
    Xt = _transform(pipeline, X_sample)
    names = get_feature_names(pipeline.named_steps["prep"])
    if len(names) != Xt.shape[1]:
        names = [f"f{i}" for i in range(Xt.shape[1])]
    Xdf = pd.DataFrame(Xt, columns=names)

    explainer = shap.TreeExplainer(model)
    sv = explainer.shap_values(Xdf)
    # Binary classifiers may return a list (one array per class) or a 3-D array.
    if isinstance(sv, list):
        sv = sv[1] if len(sv) == 2 else sv[0]
    sv = np.asarray(sv)
    if sv.ndim == 3:
        sv = sv[:, :, 1] if sv.shape[2] == 2 else sv[:, :, 0]

    # Global importance = mean |SHAP|
    mean_abs = np.abs(sv).mean(axis=0)
    ranking = (pd.DataFrame({"Feature": names, "Mean |SHAP|": mean_abs})
               .sort_values("Mean |SHAP|", ascending=False).reset_index(drop=True))
    ranking.to_csv(config.SHAP_DIR / "shap_global_importance.csv", index=False)
    out["top_features"] = ranking.head(10).to_dict("records")

    # Beeswarm summary plot
    plt.figure()
    _shap_plot(shap.summary_plot, sv, Xdf, max_display=max_display, show=False)
    plt.title(f"SHAP summary -- {model_name}\n(model behaviour, not causation)", fontsize=10)
    plt.tight_layout()
    plt.savefig(config.SHAP_DIR / "shap_summary.png", dpi=140, bbox_inches="tight")
    plt.savefig(config.FIGURES_DIR / "15_shap_summary.png", dpi=140, bbox_inches="tight")
    plt.close()

    # Global bar plot
    plt.figure()
    _shap_plot(shap.summary_plot, sv, Xdf, plot_type="bar", max_display=max_display, show=False)
    plt.title(f"SHAP global importance -- {model_name}", fontsize=10)
    plt.tight_layout()
    plt.savefig(config.SHAP_DIR / "shap_bar.png", dpi=140, bbox_inches="tight")
    plt.savefig(config.FIGURES_DIR / "16_shap_bar.png", dpi=140, bbox_inches="tight")
    plt.close()

    # Individual explanation for one passenger
    try:
        base = explainer.expected_value
        if isinstance(base, (list, np.ndarray)):
            base = np.asarray(base).ravel()[-1]
        expl = shap.Explanation(
            values=sv[0], base_values=float(base),
            data=Xdf.iloc[0].values, feature_names=names,
        )
        plt.figure()
        _shap_plot(shap.plots.waterfall, expl, max_display=14, show=False)
        plt.title("SHAP explanation for a single passenger", fontsize=10)
        plt.tight_layout()
        plt.savefig(config.SHAP_DIR / "shap_individual.png", dpi=140, bbox_inches="tight")
        plt.savefig(config.FIGURES_DIR / "17_shap_individual.png", dpi=140, bbox_inches="tight")
        plt.close()
        out["individual_plot"] = "shap_individual.png"
    except Exception as exc:  # noqa: BLE001
        utils.warn(f"individual SHAP plot skipped: {exc}")
        out["individual_plot"] = None

    # Save a compact background sample for the Streamlit app.
    Xdf.head(100).to_csv(config.SHAP_DIR / "shap_background.csv", index=False)
    return out


def permutation_importance_nn(model, preprocessor, X_val, y_val, n_repeats: int = 3) -> pd.DataFrame:
    """
    Model-agnostic permutation importance for the neural network.

    Used instead of SHAP KernelExplainer, which is prohibitively slow on a dataset
    of this size -- a limitation documented in the report.
    """
    from sklearn.inspection import permutation_importance
    from sklearn.metrics import accuracy_score

    Xt = preprocessor.transform(X_val).astype("float32")
    names = get_feature_names(preprocessor)
    if len(names) != Xt.shape[1]:
        names = [f"f{i}" for i in range(Xt.shape[1])]

    from sklearn.base import BaseEstimator, ClassifierMixin

    class _Wrapper(ClassifierMixin, BaseEstimator):
        """Minimal sklearn-compatible wrapper around the Keras model."""
        classes_ = np.array([0, 1])

        def fit(self, X, y=None):
            self.is_fitted_ = True
            return self

        def __sklearn_is_fitted__(self):
            return True

        def predict(self, X):
            return (model.predict(X, verbose=0).ravel() >= 0.5).astype(int)

        def score(self, X, y):
            return accuracy_score(y, self.predict(X))

        def get_params(self, deep=True):
            return {}

        def set_params(self, **kwargs):
            return self

    res = permutation_importance(
        _Wrapper(), Xt, np.asarray(y_val),
        n_repeats=n_repeats, random_state=config.RANDOM_STATE, scoring="accuracy",
    )
    return (pd.DataFrame({
        "Feature": names,
        "Importance": res.importances_mean,
        "Std": res.importances_std,
    }).sort_values("Importance", ascending=False).reset_index(drop=True))
