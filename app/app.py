"""
Airline Passenger Satisfaction Simulator — Streamlit application.

A flight-deck themed analytics interface over the artifacts produced by
train.py. The app NEVER retrains: models and preprocessing pipelines are loaded
from disk and cached.

Design system lives in app/components/theme.py; parallax hero in
app/components/hero.py; interactive Plotly charts in app/components/charts.py.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import joblib
import numpy as np
import pandas as pd
import streamlit as st

import config
from app.components import charts, hero
from app.components.theme import TOKENS as T
from app.components.ui import (
    caveat, inject_css, metric_card, section, verdict_banner,
)

st.set_page_config(
    page_title="Airline Satisfaction Simulator",
    page_icon="✈",
    layout="wide",
    initial_sidebar_state="expanded",
)
inject_css()


# ---------------------------------------------------------------------------
# Cached artifact loading -- the app never retrains
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_schema() -> dict | None:
    if config.SCHEMA_PATH.exists():
        return json.loads(config.SCHEMA_PATH.read_text())
    return None


@st.cache_resource(show_spinner=False)
def load_sklearn_model(name: str):
    path = config.MODELS_DIR / f"{name}.pkl"
    return joblib.load(path) if path.exists() else None


@st.cache_resource(show_spinner=False)
def load_keras_bundle():
    """Return (keras_model, preprocessor) for the best neural network, if saved."""
    pre_path = config.MODELS_DIR / "dl_preprocessor.pkl"
    if not pre_path.exists():
        return None, None
    candidates = []
    # Prefer the pointer written by train.py when a neural network was selected.
    if config.FINAL_MODEL_PATH.exists():
        try:
            ptr = joblib.load(config.FINAL_MODEL_PATH)
            if isinstance(ptr, dict) and ptr.get("kind") == "keras":
                candidates.append(config.MODELS_DIR / ptr["model_file"])
        except Exception:
            pass
    schema = load_schema() or {}
    slug = (schema.get("final_model", "").lower().replace(" ", "_")
            .replace("(", "").replace(")", "").replace("-", "_"))
    if slug:
        candidates.append(config.MODELS_DIR / f"{slug}.keras")
    candidates += sorted(config.MODELS_DIR.glob("neural_net_*.keras"))
    for c in candidates:
        if c.exists():
            import keras
            return keras.models.load_model(c), joblib.load(pre_path)
    return None, None


@st.cache_data(show_spinner=False)
def load_results() -> pd.DataFrame | None:
    if config.RESULTS_CSV.exists():
        return pd.read_csv(config.RESULTS_CSV)
    return None


@st.cache_data(show_spinner=False)
def load_manifest() -> dict | None:
    if config.MANIFEST_JSON.exists():
        return json.loads(config.MANIFEST_JSON.read_text())
    return None


@st.cache_data(show_spinner=False)
def load_feature_importance() -> pd.DataFrame | None:
    if config.FEATURE_IMPORTANCE_CSV.exists():
        return pd.read_csv(config.FEATURE_IMPORTANCE_CSV)
    return None


@st.cache_data(show_spinner=False)
def load_test_predictions() -> dict:
    """
    Recompute test-set scores for the saved models so the app can draw
    interactive ROC curves and confusion matrices. Cached, so it runs once.
    """
    import joblib
    from src import data_loader as dl
    from src import evaluate as ev

    out: dict = {}
    if not config.DATA_FILE.exists():
        return out
    splits = dl.make_splits(dl.load_raw())
    y_test = splits.y_test.to_numpy()

    for path in sorted(config.MODELS_DIR.glob("*.pkl")):
        if path.name in {"best_model.pkl", "dl_preprocessor.pkl"}:
            continue
        try:
            pipe = joblib.load(path)
            pred = pipe.predict(splits.X_test)
            score = ev.get_scores(pipe, splits.X_test)
            name = path.stem.replace("_", " ").title()
            out[name] = {"y_true": y_test, "y_pred": pred, "y_score": score}
        except Exception:
            continue

    model, pre = load_keras_bundle()
    if model is not None:
        X = pre.transform(splits.X_test).astype("float32")
        score = model.predict(X, verbose=0).ravel()
        out["Neural Network"] = {
            "y_true": y_test, "y_pred": (score >= 0.5).astype(int), "y_score": score,
        }
    return out


@st.cache_data(show_spinner=False)
def load_dataset_sample(n: int = 5000) -> tuple[pd.DataFrame | None, bool]:
    """
    Load data for the explorer page.

    Returns (dataframe, is_full_dataset). Falls back to the committed 8,000-row
    stratified sample when the full CSV is absent -- which is the case on a
    deployed instance, since the 12 MB dataset is not committed.
    """
    if config.DATA_FILE.exists():
        df = pd.read_csv(config.DATA_FILE)
        return df.sample(min(n, len(df)), random_state=config.RANDOM_STATE), True
    sample = config.DATA_DIR / "sample_for_app.csv"
    if sample.exists():
        return pd.read_csv(sample), False
    return None, False


def artifacts_ready() -> bool:
    return load_schema() is not None and load_results() is not None


# ---------------------------------------------------------------------------
# Prediction helper
# ---------------------------------------------------------------------------
def predict(raw_row: pd.DataFrame, model_choice: str):
    """
    Return (probability_satisfied, model_label).

    Works for both sklearn pipelines and the Keras bundle.
    """
    if model_choice == "keras":
        model, pre = load_keras_bundle()
        if model is None:
            return None, None
        X = pre.transform(raw_row).astype("float32")
        return float(model.predict(X, verbose=0).ravel()[0]), "Neural network"

    pipe = load_sklearn_model(model_choice)
    if pipe is None:
        return None, None
    if hasattr(pipe, "predict_proba"):
        return float(pipe.predict_proba(raw_row)[0, 1]), "Tree/linear model"
    score = float(pipe.decision_function(raw_row)[0])
    return 1 / (1 + np.exp(-score)), "Decision-function score (approximate probability)"


# ---------------------------------------------------------------------------
# Pages
# ---------------------------------------------------------------------------

def _explain_prediction(row: pd.DataFrame, model_choice: str, values: dict) -> None:
    """Per-prediction SHAP explanation for tree models; global fallback otherwise."""
    if model_choice == "keras":
        st.info(
            "Per-prediction SHAP is computed for tree-based models. For the neural "
            "network the project reports global permutation importance instead, "
            "since SHAP's KernelExplainer is computationally expensive."
        )
        perm_path = config.OUTPUTS_DIR / "nn_permutation_importance.csv"
        if perm_path.exists():
            perm = pd.read_csv(perm_path).head(12).rename(
                columns={"Importance": "Importance"}
            )
            perm["Model"] = "Neural Network"
            st.plotly_chart(
                charts.feature_importance_bar(perm, "Neural Network", top_n=12),
                width='stretch',
            )
        return

    pipe = load_sklearn_model(model_choice)
    model = pipe.named_steps.get("model") if pipe is not None else None
    if model is None or not hasattr(model, "feature_importances_"):
        st.info("This model does not expose a per-prediction SHAP explanation.")
        return

    try:
        import shap
        from src.preprocessing import get_feature_names

        Xt = pipe.named_steps["prep"].transform(row)
        names = get_feature_names(pipe.named_steps["prep"])
        if len(names) != Xt.shape[1]:
            names = [f"f{i}" for i in range(Xt.shape[1])]

        sv = shap.TreeExplainer(model).shap_values(pd.DataFrame(Xt, columns=names))
        if isinstance(sv, list):
            sv = sv[1] if len(sv) == 2 else sv[0]
        sv = np.asarray(sv)
        if sv.ndim == 3:
            sv = sv[:, :, 1] if sv.shape[2] == 2 else sv[:, :, 0]

        contrib = (pd.DataFrame({"Feature": names, "SHAP value": sv[0]})
                   .assign(abs=lambda d: d["SHAP value"].abs())
                   .nlargest(10, "abs").drop(columns="abs")
                   .sort_values("SHAP value"))
        st.plotly_chart(charts.shap_contribution(contrib), width='stretch')
        top = contrib.reindex(contrib["SHAP value"].abs().sort_values(ascending=False).index)
        lines = [
            f"- The model placed substantial predictive weight on **{r.Feature}** "
            f"({'toward Satisfied' if r._2 > 0 else 'toward Neutral/Dissatisfied'})."
            for r in top.head(4).itertuples()
        ]
        st.markdown("\n".join(lines))
    except Exception as exc:  # noqa: BLE001
        st.warning(f"SHAP explanation unavailable: {exc}")




# ---------------------------------------------------------------------------
# Page: Flight Deck (home)
# ---------------------------------------------------------------------------
def page_home() -> None:
    results = load_results()
    manifest = load_manifest() or {}

    if results is None:
        st.title("Airline Satisfaction Simulator")
        st.warning("No results yet. Open **Run Pipeline** and launch a run, or "
                   "execute `python train.py --mode full`.")
        return

    ok = results[results["Status"] == "ok"]
    best = ok.iloc[0]
    split = manifest.get("split", {})
    total = sum(v for k, v in split.items() if k in ("train", "val", "test"))
    tied = ok[(ok["f1_ci_low"] <= best["f1_ci_high"]) &
              (ok["f1_ci_high"] >= best["f1_ci_low"])] if "f1_ci_low" in ok else ok.head(1)

    # --- parallax hero (own scroll container, real scroll-driven motion) ---
    hero.parallax_hero(
        total_records=total or 129_880,
        n_models=len(ok),
        best_f1=float(best["F1"]),
        best_model=str(best["Model"]),
        n_tied=len(tied),
    )

    # --- live instrument readout ---
    section("01", "Instrument readout")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        metric_card("Best accuracy", f"{best['Accuracy']:.4f}", str(best["Model"])[:28])
    with c2:
        metric_card("Best F1", f"{best['F1']:.4f}", "primary selection metric")
    with c3:
        metric_card("Models tied", str(len(tied)), "overlapping 95% CIs", accent="amber")
    with c4:
        spread = ok["F1"].head(5).max() - ok["F1"].head(5).min()
        width = float(ok["f1_ci_width"].head(5).mean()) if "f1_ci_width" in ok else float("nan")
        metric_card("Noise ratio", f"{width / spread:.1f}×",
                    "CI width vs top-5 spread", accent="amber")

    # --- interactive leaderboard (no static images anywhere on this page) ---
    section("02", "Model leaderboard")
    lc, rc = st.columns([3, 1])
    with rc:
        metric = st.radio("Metric", ["F1", "Accuracy", "Precision", "Recall", "ROC-AUC"],
                          key="home_metric")
        families = st.multiselect(
            "Families", sorted(ok["Family"].unique()),
            default=sorted(ok["Family"].unique()), key="home_fam",
        )
        st.caption("Hover any bar for exact values. Click legend entries to toggle.")
    with lc:
        filtered = results[results["Family"].isin(families)] if families else results
        st.plotly_chart(charts.model_comparison_bar(filtered, metric),
                        width='stretch', key="home_bar")

    # --- the headline finding ---
    section("03", "Is the difference real?")
    forest = charts.ci_forest(results)
    if forest is not None:
        st.plotly_chart(forest, width='stretch', key="home_forest")
        st.markdown(
            f"""<div class="hud-panel accent-amber">
                  <div class="hud-label">Key finding</div>
                  <div style="color:{T['text']};line-height:1.7;font-size:.94rem;">
                    The top five models differ by <b style="color:{T['cyan']}">
                    {spread:.5f} F1</b>, while the mean 95% bootstrap confidence
                    interval is <b style="color:{T['amber']}">{width:.5f}</b> wide —
                    about {width / spread:.1f}× larger. <b>{len(tied)} models</b> have
                    overlapping intervals, so no winner has been demonstrated.
                    Selection should rest on training cost and interpretability.
                  </div>
                </div>""",
            unsafe_allow_html=True,
        )

    # --- performance vs cost, interactive ---
    section("04", "Performance versus cost")
    times = {k: v.get("fit_seconds") for k, v in manifest.get("models", {}).items()}
    pts = [(times.get(r["Model"]), r["F1"], r["Model"], r["Family"])
           for _, r in ok.iterrows()
           if times.get(r["Model"]) is not None and r["F1"] > 0.85]
    if pts:
        import plotly.graph_objects as go
        from app.components.theme import FAMILY_COLORS, merge_layout
        fig = go.Figure()
        for fam in sorted({p[3] for p in pts}):
            sub = [p for p in pts if p[3] == fam]
            fig.add_trace(go.Scatter(
                x=[p[0] for p in sub], y=[p[1] for p in sub],
                mode="markers", name=fam,
                marker=dict(size=15, color=FAMILY_COLORS.get(fam, T["blue"]),
                            line=dict(color=T["bg"], width=1.5)),
                text=[p[2] for p in sub],
                hovertemplate="<b>%{text}</b><br>Train: %{x:.1f}s<br>"
                              "F1: %{y:.4f}<extra></extra>",
            ))
        fig.update_layout(**merge_layout(
            title=dict(text="Training cost vs test F1 — cheaper and higher is better"),
            xaxis=dict(title="Training time (seconds, log scale)", type="log"),
            yaxis=dict(title="Test F1"),
            height=460,
        ))
        st.plotly_chart(fig, width='stretch', key="home_cost")
        caveat(
            "Because predictive performance is statistically tied, training cost and "
            "explainability decide. XGBoost matches the deep networks while training "
            "roughly five times faster and supporting exact SHAP explanations."
        )

    # --- run configuration ---
    with st.expander("Run configuration and environment"):
        c1, c2 = st.columns(2)
        with c1:
            st.json({
                "run_id": manifest.get("run_id"),
                "mode": manifest.get("mode"),
                "seed": manifest.get("seed"),
                "final_model": manifest.get("final_model"),
                "split": {k: v for k, v in split.items() if isinstance(v, int)},
            })
        with c2:
            st.json(manifest.get("library_versions", {}))


# ---------------------------------------------------------------------------
# Page: Simulate a passenger
# ---------------------------------------------------------------------------
PRESETS = {
    "Business flyer, premium service": {
        "ratings": 5, "Type of Travel": "Business", "Class": "Business",
        "Customer Type": "Returning", "Age": 42, "delays": 0,
    },
    "Budget leisure, poor service": {
        "ratings": 2, "Type of Travel": "Personal", "Class": "Economy",
        "Customer Type": "First-time", "Age": 27, "delays": 45,
    },
    "Average passenger": {
        "ratings": 3, "Type of Travel": "Business", "Class": "Economy",
        "Customer Type": "Returning", "Age": 40, "delays": 0,
    },
    "Severely delayed flight": {
        "ratings": 3, "Type of Travel": "Personal", "Class": "Economy Plus",
        "Customer Type": "Returning", "Age": 35, "delays": 240,
    },
}


def _apply_preset(schema: dict, preset: dict) -> None:
    """Write preset values into session state so widgets pick them up."""
    for f in schema["raw_features"]:
        name, key = f["name"], f"in_{f['name']}"
        if f["kind"] == "categorical":
            if name in preset and preset[name] in (f["categories"] or []):
                st.session_state[key] = preset[name]
            elif f["categories"]:
                st.session_state[key] = f["categories"][0]
        elif f["min"] == 0 and f["max"] == 5:
            st.session_state[key] = preset["ratings"]
        elif "delay" in name.lower():
            st.session_state[key] = float(preset["delays"])
        elif name in preset:
            st.session_state[key] = float(preset[name])
        else:
            st.session_state[key] = float(f["median"])


def page_prediction() -> None:
    st.title("Simulate a passenger")
    schema = load_schema()
    if not schema:
        st.error("No trained model found. Run the pipeline first.")
        return

    st.caption(
        "Inputs are generated from the saved feature schema, so this form always "
        "matches the features the model was trained on. Predictions update live."
    )

    available = {}
    if (config.MODELS_DIR / "dl_preprocessor.pkl").exists() and list(
        config.MODELS_DIR.glob("neural_net_*.keras")
    ):
        available["Neural network (deep MLP)"] = "keras"
    for label, slug in [
        ("Random Forest", "random_forest"), ("XGBoost", "xgboost"),
        ("Extra Trees", "extra_trees"), ("Stacking Ensemble", "stacking_ensemble"),
        ("Logistic Regression", "logistic_regression"),
    ]:
        if (config.MODELS_DIR / f"{slug}.pkl").exists():
            available[label] = slug
    if not available:
        st.error("No saved models found in models/.")
        return

    feats = schema["raw_features"]
    for f in feats:  # seed defaults once
        key = f"in_{f['name']}"
        if key not in st.session_state:
            st.session_state[key] = (
                (f["categories"] or [""])[0] if f["kind"] == "categorical"
                else float(f["median"])
            )

    section("01", "Flight profile presets")
    pc = st.columns(len(PRESETS))
    for i, (name, preset) in enumerate(PRESETS.items()):
        with pc[i]:
            if st.button(name, key=f"preset_{i}", width='stretch'):
                _apply_preset(schema, preset)
                st.rerun()

    numeric = [f for f in feats if f["kind"] == "numeric"]
    categorical = [f for f in feats if f["kind"] == "categorical"]
    ratings = [f for f in numeric if f["min"] == 0 and f["max"] == 5]
    others = [f for f in numeric if f not in ratings]

    left, right = st.columns([1.15, 1])

    with left:
        section("02", "Passenger and itinerary")
        cols = st.columns(2)
        for i, f in enumerate(categorical):
            with cols[i % 2]:
                st.selectbox(f["name"], f["categories"] or [], key=f"in_{f['name']}")
        for i, f in enumerate(others):
            with cols[(i + len(categorical)) % 2]:
                lo, hi = float(f["min"]), float(f["max"])
                st.number_input(f["name"], min_value=lo, max_value=max(hi, lo + 1),
                                step=1.0, key=f"in_{f['name']}")

        if ratings:
            section("03", "Service ratings")
            st.caption(
                "Scale 1–5, higher is better. **0 means 'not applicable'** in this "
                "dataset — passengers who rate 0 never used the service."
            )
            b1, b2, b3 = st.columns(3)
            with b1:
                if st.button("Set all to 5", width='stretch', key="all5"):
                    for f in ratings:
                        st.session_state[f"in_{f['name']}"] = 5
                    st.rerun()
            with b2:
                if st.button("Set all to 3", width='stretch', key="all3"):
                    for f in ratings:
                        st.session_state[f"in_{f['name']}"] = 3
                    st.rerun()
            with b3:
                if st.button("Set all to 1", width='stretch', key="all1"):
                    for f in ratings:
                        st.session_state[f"in_{f['name']}"] = 1
                    st.rerun()

            rcols = st.columns(2)
            for i, f in enumerate(ratings):
                with rcols[i % 2]:
                    st.slider(f["name"], 0, 5, key=f"in_{f['name']}",
                              help="0 = not applicable")

    values = {f["name"]: st.session_state[f"in_{f['name']}"] for f in feats}
    row = pd.DataFrame([values])[[f["name"] for f in feats]]

    with right:
        section("04", "Live prediction")
        default_label = next(
            (k for k in available
             if schema.get("final_model", "").lower().startswith("neural")
             and available[k] == "keras"),
            list(available)[0],
        )
        choice_label = st.selectbox(
            "Model", list(available), index=list(available).index(default_label),
            key="pred_model",
            help="The model selected by the experiment is pre-selected.",
        )

        prob, kind = predict(row, available[choice_label])
        if prob is None:
            st.error("Prediction failed: model could not be loaded.")
            return

        satisfied = prob >= 0.5
        label = config.POSITIVE_LABEL if satisfied else config.NEGATIVE_LABEL
        verdict_banner(label.upper(), prob if satisfied else 1 - prob, satisfied)
        st.plotly_chart(charts.satisfaction_gauge(prob), width='stretch',
                        key="pred_gauge")

        m1, m2 = st.columns(2)
        with m1:
            metric_card("P(Satisfied)", f"{prob:.1%}", choice_label[:24],
                        accent="green" if satisfied else "")
        with m2:
            metric_card("P(Neutral/Dissat.)", f"{1 - prob:.1%}", kind or "",
                        accent="" if satisfied else "amber")

        if st.checkbox("Compare across all models", key="cmp_all"):
            rows = []
            for lbl, slug in available.items():
                p, _ = predict(row, slug)
                if p is not None:
                    rows.append({"Model": lbl, "P(Satisfied)": p,
                                 "Prediction": config.POSITIVE_LABEL if p >= .5
                                 else config.NEGATIVE_LABEL})
            if rows:
                cmp_df = pd.DataFrame(rows).sort_values("P(Satisfied)", ascending=False)
                st.dataframe(
                    cmp_df.style.format({"P(Satisfied)": "{:.1%}"}),
                    width='stretch', hide_index=True,
                )
                agree = cmp_df["Prediction"].nunique() == 1
                st.caption(
                    "All models agree on this passenger."
                    if agree else
                    "Models disagree — this passenger sits near the decision boundary."
                )

    section("05", "Why did the model predict this?")
    _explain_prediction(row, available[choice_label], values)
    caveat(
        "This is a statistical prediction from historical data. It describes patterns "
        "the model learned — not the cause of any individual passenger's satisfaction."
    )


# ---------------------------------------------------------------------------
# Page: Data explorer
# ---------------------------------------------------------------------------
def page_analytics() -> None:
    st.title("Data explorer")
    df, is_full = load_dataset_sample()
    if df is None:
        st.error(
            "No data available. Place `airline_passenger_satisfaction.csv` in `data/` "
            "(see data/README.md)."
        )
        return
    if not is_full:
        st.info(
            "Showing the committed **8,000-row stratified sample** — the full "
            "129,880-row dataset is not distributed with this app. Class balance "
            "and every category level are preserved, so the patterns below match "
            "the full data."
        )

    target = next((c for c in ("Satisfaction", "satisfaction") if c in df.columns), None)
    if not target:
        st.error("Target column not found.")
        return

    section("01", "Dataset at a glance")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        metric_card("Sample rows", f"{len(df):,}", "random sample for responsiveness")
    with c2:
        metric_card("Columns", str(df.shape[1]), "raw dataset")
    with c3:
        metric_card("Missing", f"{int(df.isna().sum().sum()):,}", "in this sample")
    with c4:
        rate = (df[target] == config.POSITIVE_LABEL).mean()
        metric_card("Satisfied", f"{rate:.1%}", "share in sample", accent="green")

    # interactive cross-filter
    section("02", "Filter the cohort")
    f1, f2, f3 = st.columns(3)
    view = df.copy()
    with f1:
        if "Class" in df.columns:
            pick = st.multiselect("Cabin class", sorted(df["Class"].unique()),
                                  default=sorted(df["Class"].unique()), key="fx_class")
            if pick:
                view = view[view["Class"].isin(pick)]
    with f2:
        if "Type of Travel" in df.columns:
            pick = st.multiselect("Travel type", sorted(df["Type of Travel"].unique()),
                                  default=sorted(df["Type of Travel"].unique()),
                                  key="fx_travel")
            if pick:
                view = view[view["Type of Travel"].isin(pick)]
    with f3:
        if "Age" in df.columns:
            lo, hi = int(df["Age"].min()), int(df["Age"].max())
            a, b = st.slider("Age range", lo, hi, (lo, hi), key="fx_age")
            view = view[(view["Age"] >= a) & (view["Age"] <= b)]

    if view.empty:
        st.warning("No passengers match these filters.")
        return
    st.caption(
        f"**{len(view):,}** of {len(df):,} passengers match — "
        f"satisfaction rate **{(view[target] == config.POSITIVE_LABEL).mean():.1%}** "
        f"(vs {(df[target] == config.POSITIVE_LABEL).mean():.1%} overall)."
    )

    tabs = st.tabs(["Distribution", "Service ratings", "Delays", "Correlations"])

    with tabs[0]:
        c1, c2 = st.columns(2)
        with c1:
            st.plotly_chart(
                charts.class_distribution(view, target, config.POSITIVE_LABEL),
                width='stretch', key="an_dist")
        with c2:
            cats = [c for c in ("Class", "Type of Travel", "Customer Type", "Gender")
                    if c in view.columns]
            if cats:
                pick = st.selectbox("Break down by", cats, key="an_cat")
                st.plotly_chart(
                    charts.satisfaction_by_category(view, pick, target,
                                                    config.POSITIVE_LABEL),
                    width='stretch', key="an_cat_chart")

    with tabs[1]:
        rating_cols = [c for c in view.select_dtypes("number").columns
                       if view[c].dropna().between(0, 5).all() and view[c].nunique() <= 6]
        if rating_cols:
            st.plotly_chart(
                charts.rating_comparison(view, rating_cols, target, config.POSITIVE_LABEL),
                width='stretch', key="an_rating")
            st.markdown("**Satisfaction rate by rating value**")
            pick = st.selectbox("Service dimension", rating_cols, key="an_rating_pick")
            import plotly.graph_objects as go
            from app.components.theme import merge_layout
            rate = view.groupby(pick, observed=True)[target].apply(
                lambda s: (s == config.POSITIVE_LABEL).mean() * 100)
            counts = view.groupby(pick, observed=True).size()
            colors = [T["amber"] if i == 0 else T["cyan"] for i in rate.index]
            fig = go.Figure(go.Bar(
                x=[str(i) for i in rate.index], y=rate.values, marker_color=colors,
                customdata=[counts[i] for i in rate.index],
                hovertemplate="Rating %{x}<br>Satisfied: %{y:.1f}%<br>"
                              "Passengers: %{customdata:,}<extra></extra>",
            ))
            fig.update_layout(**merge_layout(
                title=dict(text=f"{pick} — note the anomalous 0 bar"),
                xaxis=dict(title="Rating given (0 = not applicable)"),
                yaxis=dict(title="% satisfied"), height=380,
            ))
            st.plotly_chart(fig, width='stretch', key="an_rating_detail")
            caveat(
                "Rating 0 breaks the monotonic pattern: it denotes 'not applicable' "
                "rather than 'worst'. This project keeps ratings as plain 0–5 numeric "
                "values for comparability with published work, and documents the effect."
            )

    with tabs[2]:
        dcols = [c for c in view.columns if "delay" in c.lower()]
        if dcols:
            import plotly.graph_objects as go
            from app.components.theme import merge_layout
            dep, arr = dcols[0], (dcols[1] if len(dcols) > 1 else dcols[0])
            tmp = view.assign(total=view[dep].fillna(0) + view[arr].fillna(0))
            bins = [-0.1, 0, 15, 60, 180, np.inf]
            labels = ["0 min", "1–15", "16–60", "61–180", ">180"]
            tmp["band"] = pd.cut(tmp["total"], bins=bins, labels=labels)
            g = tmp.groupby("band", observed=True)[target].apply(
                lambda s: (s == config.POSITIVE_LABEL).mean() * 100)
            n = tmp.groupby("band", observed=True).size()
            fig = go.Figure(go.Bar(
                x=list(g.index.astype(str)), y=g.values, marker_color=T["cyan"],
                customdata=[n[i] for i in g.index],
                hovertemplate="%{x}<br>Satisfied: %{y:.1f}%<br>"
                              "Passengers: %{customdata:,}<extra></extra>"))
            fig.update_layout(**merge_layout(
                title=dict(text="Satisfaction rate by total delay"),
                xaxis=dict(title="Total delay"), yaxis=dict(title="% satisfied"),
                height=400))
            st.plotly_chart(fig, width='stretch', key="an_delay")
            st.caption(
                f"Delays are heavily right-skewed: median "
                f"{view[dep].median():.0f} min, maximum {view[dep].max():.0f} min."
            )

    with tabs[3]:
        import plotly.graph_objects as go
        from app.components.theme import merge_layout
        num = view.select_dtypes("number").copy()
        num["__satisfied__"] = (view[target] == config.POSITIVE_LABEL).astype(int)
        corr = num.corr(numeric_only=True)["__satisfied__"].drop("__satisfied__")
        corr = corr.sort_values()
        fig = go.Figure(go.Bar(
            x=corr.values, y=corr.index, orientation="h",
            marker_color=[T["red"] if v < 0 else T["cyan"] for v in corr.values],
            hovertemplate="<b>%{y}</b><br>r = %{x:.3f}<extra></extra>"))
        fig.update_layout(**merge_layout(
            title=dict(text="Linear correlation with satisfaction"),
            xaxis=dict(title="Pearson r"), height=max(400, 24 * len(corr))))
        st.plotly_chart(fig, width='stretch', key="an_corr")
        caveat(
            "No single feature shows a strong linear correlation — which is why "
            "non-linear models outperform Logistic Regression by roughly 0.09 F1."
        )


# ---------------------------------------------------------------------------
# Page: Model bay
# ---------------------------------------------------------------------------
def page_models() -> None:
    st.title("Model bay")
    results = load_results()
    if results is None:
        st.error("No results found. Run the pipeline first.")
        return

    ok = results[results["Status"] == "ok"].copy()
    failed = results[results["Status"] != "ok"]

    section("01", "Compare any two models")
    c1, c2, c3 = st.columns([1, 1, 2])
    names = list(ok["Model"])
    with c1:
        a = st.selectbox("Model A", names, index=0, key="cmp_a")
    with c2:
        b = st.selectbox("Model B", names, index=min(2, len(names) - 1), key="cmp_b")
    ra = ok[ok["Model"] == a].iloc[0]
    rb = ok[ok["Model"] == b].iloc[0]
    with c3:
        if "f1_ci_low" in ok:
            overlap = (ra["f1_ci_low"] <= rb["f1_ci_high"]) and \
                      (rb["f1_ci_low"] <= ra["f1_ci_high"])
            if overlap:
                st.markdown(
                    f"""<div class="hud-panel accent-amber">
                    <div class="hud-label">Verdict</div>
                    <div style="color:{T['text']};font-size:.92rem;line-height:1.6;">
                    Confidence intervals <b>overlap</b> — the {abs(ra['F1']-rb['F1']):.5f}
                    F1 gap is inside the noise. No difference demonstrated.</div></div>""",
                    unsafe_allow_html=True)
            else:
                better = a if ra["F1"] > rb["F1"] else b
                st.markdown(
                    f"""<div class="hud-panel accent-green">
                    <div class="hud-label">Verdict</div>
                    <div style="color:{T['text']};font-size:.92rem;line-height:1.6;">
                    Intervals do <b>not</b> overlap — <b>{better}</b> is meaningfully
                    ahead by {abs(ra['F1']-rb['F1']):.4f} F1.</div></div>""",
                    unsafe_allow_html=True)

    metrics = ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC"]
    import plotly.graph_objects as go
    from app.components.theme import merge_layout
    fig = go.Figure()
    for name, row, colour in [(a, ra, T["cyan"]), (b, rb, T["violet"])]:
        fig.add_trace(go.Scatterpolar(
            r=[float(row[m]) for m in metrics] + [float(row[metrics[0]])],
            theta=metrics + [metrics[0]], fill="toself", name=str(name),
            line=dict(color=colour),
            hovertemplate="<b>%{theta}</b><br>%{r:.4f}<extra>" + str(name) + "</extra>"))
    fig.update_layout(**merge_layout(
        title=dict(text="Head-to-head across all metrics"),
        polar=dict(radialaxis=dict(range=[0.8, 1.0], gridcolor=T["grid"]),
                   angularaxis=dict(gridcolor=T["grid"]),
                   bgcolor="rgba(0,0,0,0)"),
        height=460))
    st.plotly_chart(fig, width='stretch', key="mb_radar")

    section("02", "Full comparison")
    metric = st.radio("Metric", metrics, horizontal=True, key="mb_metric")
    st.plotly_chart(charts.model_comparison_bar(results, metric),
                    width='stretch', key="mb_bar")

    by_family = st.checkbox("Group table by family", value=True, key="mb_group")
    order = {"baseline": 0, "classical": 1, "ensemble": 2, "deep": 3}
    if by_family:
        for fam in sorted(ok["Family"].dropna().unique(), key=lambda f: order.get(f, 9)):
            sub = ok[ok["Family"] == fam].sort_values("F1", ascending=False)
            st.markdown(f"**{fam.capitalize()}**")
            st.dataframe(
                sub[["Model"] + metrics].style.format({m: "{:.4f}" for m in metrics}),
                width='stretch', hide_index=True)
    else:
        st.dataframe(
            ok[["Model", "Family"] + metrics].style.format({m: "{:.4f}" for m in metrics}),
            width='stretch', hide_index=True)

    section("03", "Confidence intervals")
    forest = charts.ci_forest(results)
    if forest is not None:
        st.plotly_chart(forest, width='stretch', key="mb_forest")

    cv_cols = [c for c in ok.columns if c.startswith("CV ")]
    if cv_cols:
        cv = ok.dropna(subset=cv_cols, how="all")
        if not cv.empty:
            with st.expander("Cross-validation stability (training data only)"):
                st.dataframe(
                    cv[["Model"] + cv_cols].style.format({c: "{:.4f}" for c in cv_cols}),
                    width='stretch', hide_index=True)

    section("04", "ROC curves and confusion matrices")
    with st.spinner("Scoring saved models on the test set (cached after first run)..."):
        preds = load_test_predictions()

    if preds:
        pick_models = st.multiselect(
            "Curves to show", sorted(preds), default=sorted(preds)[:6], key="mb_roc_pick")
        curve_data = {n: (preds[n]["y_true"], preds[n]["y_score"])
                      for n in pick_models if preds[n]["y_score"] is not None}
        if curve_data:
            st.plotly_chart(charts.roc_curves(curve_data), width='stretch', key="mb_roc")

        cm_pick = st.selectbox("Confusion matrix for", sorted(preds), key="mb_cm")
        from sklearn.metrics import confusion_matrix
        d = preds[cm_pick]
        cm = confusion_matrix(d["y_true"], d["y_pred"])
        c1, c2 = st.columns([1.3, 1])
        with c1:
            st.plotly_chart(charts.confusion_heatmap(cm, cm_pick),
                            width='stretch', key="mb_cm_chart")
        with c2:
            tn, fp, fn, tp = cm.ravel()
            metric_card("True positives", f"{tp:,}", "satisfied, predicted satisfied",
                        accent="green")
            st.write("")
            metric_card("False positives", f"{fp:,}",
                        "dissatisfied, predicted satisfied", accent="amber")
            st.write("")
            metric_card("False negatives", f"{fn:,}",
                        "satisfied, predicted dissatisfied")
            caveat(
                "A false positive is the costlier error: the airline takes no recovery "
                "action for a dissatisfied passenger who may not return."
            )

    section("05", "Feature importance")
    imp = load_feature_importance()
    if imp is not None and not imp.empty:
        c1, c2 = st.columns([1, 3])
        with c1:
            model = st.selectbox("Model", sorted(imp["Model"].unique()), key="mb_imp")
            top_n = st.slider("Features shown", 5, 30, 15, key="mb_topn")
        with c2:
            st.plotly_chart(charts.feature_importance_bar(imp, model, top_n),
                            width='stretch', key="mb_imp_chart")

    perm_path = config.OUTPUTS_DIR / "nn_permutation_importance.csv"
    if perm_path.exists():
        with st.expander("Neural network permutation importance"):
            perm = pd.read_csv(perm_path).head(15)
            perm["Model"] = "Neural Network"
            st.plotly_chart(
                charts.feature_importance_bar(perm, "Neural Network", 15),
                width='stretch', key="mb_perm")
            st.caption(
                "SHAP's KernelExplainer is too slow at this dataset size, so the "
                "neural network is explained with permutation importance instead — "
                "a documented limitation."
            )

    if not failed.empty:
        section("06", "Models that did not complete")
        st.caption("Reported rather than silently skipped.")
        st.dataframe(failed[["Model", "Status"]], width='stretch', hide_index=True)


# ---------------------------------------------------------------------------
# Page: Run pipeline
# ---------------------------------------------------------------------------
def page_pipeline() -> None:
    st.title("Run pipeline")
    st.caption(
        "Launch the training pipeline and watch it progress. Output is streamed "
        "from `train.py` — the same command you would run in a terminal."
    )

    manifest = load_manifest()
    if manifest:
        section("01", "Last run")
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            metric_card("Run ID", str(manifest.get("run_id", "n/a")),
                        f"mode: {manifest.get('mode', '')}")
        with c2:
            metric_card("Final model", str(manifest.get("final_model", "n/a"))[:22],
                        "by test F1")
        with c3:
            nf = len(manifest.get("failures", {}))
            metric_card("Failures", str(nf), "models that did not complete",
                        accent="amber" if nf else "green")
        with c4:
            secs = sum(v.get("fit_seconds", 0) for v in manifest.get("models", {}).values())
            metric_card("Total fit time", f"{secs:.0f}s", "sum across models")

    section("02", "Configure")
    c1, c2 = st.columns([1, 2])
    with c1:
        mode = st.radio("Mode", ["fast", "full"], index=1, key="pl_mode",
                        help="fast: subsampled, ~40s. full: all records, ~7.5 min.")
        skip_eda = st.checkbox("Skip EDA figures", key="pl_eda")
        skip_dl = st.checkbox("Skip deep learning", key="pl_dl")
    with c2:
        est = "about 40 seconds" if mode == "fast" else "about 7–8 minutes"
        st.markdown(
            f"""<div class="hud-panel">
                  <div class="hud-label">Mode: {mode}</div>
                  <div style="color:{T['text']};line-height:1.7;font-size:.9rem;">
                    Estimated runtime <b style="color:{T['cyan']}">{est}</b>.
                    Re-running overwrites <code>outputs/</code> and <code>models/</code>;
                    every page will update to match the new run.
                  </div>
                </div>""",
            unsafe_allow_html=True)
        if mode == "fast":
            st.warning(
                "Fast mode uses a 15,000-row subsample and tiny tuning budgets. "
                "Its numbers are for debugging — do not quote them in the report."
            )

    if not st.button("Launch pipeline", type="primary", key="pl_go"):
        st.stop()

    cmd = [sys.executable, str(ROOT / "train.py"), "--mode", mode]
    if skip_eda:
        cmd.append("--skip-eda")
    if skip_dl:
        cmd.append("--skip-dl")

    section("03", "Telemetry")
    progress = st.progress(0.0, text="Starting...")
    status = st.empty()
    log_box = st.empty()

    import re
    import subprocess

    env = dict(os.environ, PYTHONUNBUFFERED="1", TF_CPP_MIN_LOG_LEVEL="3")
    proc = subprocess.Popen(cmd, cwd=str(ROOT), stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True, bufsize=1, env=env)

    lines: list[str] = []
    noise = ("oneDNN", "cpu_feature", "tensorflow/core", "external/local",
             "absl::", "WARNING: All log")
    step_re = re.compile(r"^\[(\d+)/(\d+)\]\s*(.*)")

    for raw in proc.stdout:
        line = raw.rstrip()
        if not line or any(n in line for n in noise):
            continue
        lines.append(line)
        m = step_re.match(line)
        if m:
            done, total = int(m.group(1)), int(m.group(2))
            progress.progress(min(done / max(total, 1), 1.0),
                              text=f"Step {done}/{total} — {m.group(3)}")
            status.info(m.group(3))
        log_box.code("\n".join(lines[-26:]), language="text")

    proc.wait()

    if proc.returncode == 0:
        progress.progress(1.0, text="Complete")
        status.success("Pipeline finished successfully.")
        st.cache_data.clear()
        st.cache_resource.clear()
        results = load_results()
        if results is not None:
            ok = results[results["Status"] == "ok"]
            section("04", "Results from this run")
            st.plotly_chart(charts.model_comparison_bar(results, "F1"),
                            width='stretch', key="pl_bar")
            st.dataframe(
                ok[["Model", "Family", "Accuracy", "Precision", "Recall", "F1", "ROC-AUC"]]
                .style.format({c: "{:.4f}" for c in
                               ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC"]}),
                width='stretch', hide_index=True)
            st.success(
                f"Caches cleared — all pages now reflect this run. "
                f"Top model: **{ok.iloc[0]['Model']}** (F1 {ok.iloc[0]['F1']:.4f})."
            )
    else:
        progress.progress(1.0, text="Failed")
        status.error(f"Pipeline exited with code {proc.returncode}.")
        st.code("\n".join(lines[-60:]), language="text")


# ---------------------------------------------------------------------------
# Navigation
# ---------------------------------------------------------------------------
PAGES = {
    "Flight Deck": page_home,
    "Simulate Passenger": page_prediction,
    "Data Explorer": page_analytics,
    "Model Bay": page_models,
    "Run Pipeline": page_pipeline,
}


def main() -> None:
    st.sidebar.markdown(
        f"""<div style="padding:.5rem 0 1rem;">
              <div style="font-family:'Fira Code',monospace;font-size:.62rem;
                   letter-spacing:.2em;color:{T['cyan']};text-transform:uppercase;">
                Airline Analytics
              </div>
              <div style="font-family:'Chakra Petch',sans-serif;font-size:1.22rem;
                   font-weight:700;color:{T['text']};line-height:1.2;margin-top:.2rem;">
                Satisfaction<br>Simulator
              </div>
            </div>""",
        unsafe_allow_html=True)

    if load_schema() is None or load_results() is None:
        st.sidebar.warning("Artifacts missing — open **Run Pipeline**.")

    page = st.sidebar.radio("Navigate", list(PAGES), label_visibility="collapsed")
    st.sidebar.markdown("---")

    manifest = load_manifest()
    if manifest:
        st.sidebar.markdown(
            f"""<div style="font-family:'Fira Code',monospace;font-size:.68rem;
                 line-height:1.9;color:{T['muted']};">
              <div><span style="color:{T['cyan']}">MODEL</span>
                   {str(manifest.get('final_model','n/a'))[:24]}</div>
              <div><span style="color:{T['cyan']}">RUN</span>
                   {manifest.get('run_id','n/a')}</div>
              <div><span style="color:{T['cyan']}">MODE</span>
                   {manifest.get('mode','n/a')}</div>
              <div><span style="color:{T['cyan']}">SEED</span>
                   {manifest.get('seed','n/a')}</div>
            </div>""",
            unsafe_allow_html=True)
    st.sidebar.markdown("---")
    st.sidebar.caption(
        "B.Tech CSE research project. Every metric shown is produced by this "
        "project's own experiments."
    )

    # Streamlit diffs element slots positionally, so switching pages can leave a
    # stale block from the previous page visible. Rendering into a placeholder
    # that is emptied first guarantees the old subtree is torn down.
    slot = st.empty()
    if st.session_state.get("_active_page") != page:
        slot.empty()
        st.session_state["_active_page"] = page
    with slot.container():
        PAGES[page]()
        hero.scroll_reveal()


if __name__ == "__main__":
    main()
