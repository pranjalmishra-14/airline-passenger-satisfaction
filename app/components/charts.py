"""
Interactive Plotly charts for the Streamlit app.

These replace the static matplotlib figures in the app itself (the PNGs remain
for the report). Every chart supports hover-for-exact-values, zoom and pan; the
multi-series charts support click-to-toggle via the legend.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from app.components.theme import FAMILY_COLORS, TITLE_FONT, TOKENS as T, merge_layout, plotly_layout

LAYOUT = plotly_layout()
ACCENT = T["cyan"]
ACCENT_2 = T["blue"]
POS_COLOR = T["green"]
NEG_COLOR = T["red"]


def model_comparison_bar(results: pd.DataFrame, metric: str = "F1") -> go.Figure:
    """Horizontal bar chart of a chosen metric, coloured by model family."""
    ok = results[results["Status"] == "ok"].sort_values(metric)
    fig = go.Figure()
    for fam in ok["Family"].unique():
        sub = ok[ok["Family"] == fam]
        fig.add_trace(go.Bar(
            x=sub[metric], y=sub["Model"], orientation="h", name=fam,
            marker_color=FAMILY_COLORS.get(fam, ACCENT_2),
            text=[f"{v:.4f}" for v in sub[metric]],
            textposition="outside",
            hovertemplate=(
                "<b>%{y}</b><br>" + metric + ": %{x:.4f}<br>"
                "Family: " + fam + "<extra></extra>"
            ),
        ))
    fig.update_layout(**merge_layout(
        title=dict(text=f"Test {metric} by model — click a family to toggle", font=TITLE_FONT),
        xaxis_title=metric, yaxis_title="",
        xaxis=dict(range=[0, 1.08]),
        height=max(380, 34 * len(ok)),
        legend_title="Family",
    ))
    return fig


def ci_forest(results: pd.DataFrame, top_n: int = 10) -> go.Figure | None:
    """
    Interactive forest plot of bootstrap confidence intervals.

    Visualises the headline finding: overlapping intervals among the leaders.
    """
    if "f1_ci_low" not in results.columns:
        return None
    ok = results[results["Status"] == "ok"].dropna(subset=["f1_ci_low"])
    if ok.empty:
        return None

    ok = ok.nlargest(top_n, "F1").sort_values("F1")
    best = ok.iloc[-1]
    tied = (ok["f1_ci_low"] <= best["f1_ci_high"]) & (ok["f1_ci_high"] >= best["f1_ci_low"])

    fig = go.Figure()
    # Shaded band = the best model's interval.
    fig.add_vrect(x0=best["f1_ci_low"], x1=best["f1_ci_high"],
                  fillcolor=ACCENT_2, opacity=0.10, line_width=0)

    for i, (_, row) in enumerate(ok.iterrows()):
        colour = NEG_COLOR if tied.iloc[i] else T["dim"]
        status = "tied with best" if tied.iloc[i] else "separated from best"
        fig.add_trace(go.Scatter(
            x=[row["f1_ci_low"], row["f1_ci_high"]], y=[row["Model"]] * 2,
            mode="lines", line=dict(color=colour, width=4),
            showlegend=False, hoverinfo="skip",
        ))
        fig.add_trace(go.Scatter(
            x=[row["F1"]], y=[row["Model"]], mode="markers",
            marker=dict(color=colour, size=11, line=dict(color="white", width=1.5)),
            showlegend=False,
            hovertemplate=(
                f"<b>{row['Model']}</b><br>F1: {row['F1']:.4f}<br>"
                f"95% CI: [{row['f1_ci_low']:.4f}, {row['f1_ci_high']:.4f}]<br>"
                f"Width: {row['f1_ci_width']:.4f}<br>{status}<extra></extra>"
            ),
        ))

    spread = ok["F1"].tail(5).max() - ok["F1"].tail(5).min()
    width = float(ok["f1_ci_width"].tail(5).mean())
    fig.update_layout(**merge_layout(
        title=(f"Are the top models actually different?<br>"
               f"<sub>Top-5 spread {spread:.5f} vs mean CI width {width:.5f} "
               f"({width / spread:.1f}x larger) — {int(tied.sum())} models are "
               f"statistically indistinguishable</sub>"),
        xaxis_title="Test F1 with 95% bootstrap confidence interval",
        height=max(400, 42 * len(ok)),
    ))
    return fig


def roc_curves(curve_data: dict) -> go.Figure:
    """Overlaid ROC curves; click the legend to isolate a model."""
    from sklearn.metrics import auc, roc_curve

    fig = go.Figure()
    for name, (y_true, y_score) in curve_data.items():
        fpr, tpr, _ = roc_curve(y_true, y_score)
        a = auc(fpr, tpr)
        fig.add_trace(go.Scatter(
            x=fpr, y=tpr, mode="lines", name=f"{name} ({a:.4f})",
            hovertemplate="FPR %{x:.3f}<br>TPR %{y:.3f}<extra></extra>",
        ))
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1], mode="lines", name="Chance",
        line=dict(dash="dash", color=T["dim"]), hoverinfo="skip",
    ))
    fig.update_layout(**merge_layout(
        title="ROC curves (test set) — click legend entries to toggle",
        xaxis_title="False positive rate", yaxis_title="True positive rate",
        height=560,
    ))
    return fig


def confusion_heatmap(cm: np.ndarray, model_name: str) -> go.Figure:
    """Confusion matrix with counts and row-normalised percentages on hover."""
    arr = np.asarray(cm)
    pct = arr / arr.sum(axis=1, keepdims=True) * 100
    labels = ["Neutral / Dissatisfied", "Satisfied"]
    text = [[f"{v:,}<br>({p:.1f}%)" for v, p in zip(r, pr)] for r, pr in zip(arr, pct)]

    fig = go.Figure(go.Heatmap(
        z=arr, x=[f"Predicted<br>{l}" for l in labels], y=[f"Actual<br>{l}" for l in labels],
        text=text, texttemplate="%{text}", colorscale=[[0,"rgba(56,189,248,.05)"],[1,T["cyan"]]], showscale=False,
        hovertemplate="%{y} → %{x}<br>Count: %{z:,}<extra></extra>",
    ))
    fig.update_layout(**merge_layout(title=dict(text=f"Confusion matrix — {model_name}"), height=420))
    return fig


def feature_importance_bar(imp: pd.DataFrame, model: str, top_n: int = 15) -> go.Figure:
    """Ranked feature importance for one model."""
    sub = imp[imp["Model"] == model].nlargest(top_n, "Importance").sort_values("Importance")
    fig = go.Figure(go.Bar(
        x=sub["Importance"], y=sub["Feature"], orientation="h",
        marker_color=POS_COLOR,
        hovertemplate="<b>%{y}</b><br>Importance: %{x:.4f}<extra></extra>",
    ))
    fig.update_layout(**merge_layout(
        title=f"Top {top_n} features — {model}",
        xaxis_title="Importance", height=max(400, 26 * len(sub)),
    ))
    return fig


def shap_contribution(contrib: pd.DataFrame) -> go.Figure:
    """Per-prediction SHAP contributions, coloured by direction."""
    colours = [NEG_COLOR if v < 0 else POS_COLOR for v in contrib["SHAP value"]]
    fig = go.Figure(go.Bar(
        x=contrib["SHAP value"], y=contrib["Feature"], orientation="h",
        marker_color=colours,
        hovertemplate="<b>%{y}</b><br>SHAP: %{x:+.4f}<extra></extra>",
    ))
    fig.add_vline(x=0, line_width=1, line_color=T["muted"])
    fig.update_layout(**merge_layout(
        title="Feature contributions for this passenger<br>"
              "<sub>Green pushes toward Satisfied; red toward Neutral/Dissatisfied</sub>",
        xaxis_title="SHAP value (model weight, not causation)",
        height=max(380, 28 * len(contrib)),
    ))
    return fig


def satisfaction_gauge(prob: float) -> go.Figure:
    """Gauge showing the predicted probability of satisfaction."""
    satisfied = prob >= 0.5
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=prob * 100,
        number={"suffix": "%", "font": {"size": 40, "color": T["text"],
                                        "family": "Chakra Petch, sans-serif"}},
        title={"text": "Probability of Satisfaction",
               "font": {"size": 13, "color": T["muted"]}},
        gauge={
            "axis": {"range": [0, 100], "tickcolor": T["muted"],
                     "tickfont": {"color": T["muted"], "size": 10}},
            "bgcolor": "rgba(0,0,0,0)", "borderwidth": 0,
            "bar": {"color": POS_COLOR if satisfied else NEG_COLOR},
            "steps": [
                {"range": [0, 50], "color": "rgba(251,113,133,.12)"},
                {"range": [50, 100], "color": "rgba(52,211,153,.12)"},
            ],
            "threshold": {"line": {"color": T["muted"], "width": 3},
                          "thickness": 0.8, "value": 50},
        },
    ))
    fig.update_layout(**merge_layout(height=300,
                                     margin=dict(l=20, r=20, t=58, b=10)))
    return fig


def training_history(history: dict, model_name: str) -> go.Figure:
    """Accuracy and loss versus epochs for one neural network."""
    from plotly.subplots import make_subplots

    epochs = list(range(1, len(history["loss"]) + 1))
    fig = make_subplots(rows=1, cols=2,
                        subplot_titles=("Accuracy vs epochs", "Loss vs epochs"))
    for key, name, col in [("accuracy", "train", 1), ("val_accuracy", "validation", 1),
                           ("loss", "train", 2), ("val_loss", "validation", 2)]:
        if key in history:
            fig.add_trace(go.Scatter(
                x=epochs, y=history[key], mode="lines", name=f"{name} ({'acc' if col == 1 else 'loss'})",
                hovertemplate="Epoch %{x}<br>%{y:.4f}<extra></extra>",
            ), row=1, col=col)
    fig.update_layout(**merge_layout(title=dict(text=f"Training history — {model_name}"), height=420))
    return fig


def class_distribution(df: pd.DataFrame, target: str, positive: str) -> go.Figure:
    """Donut chart of the target distribution."""
    vc = df[target].value_counts()
    fig = go.Figure(go.Pie(
        labels=vc.index, values=vc.values, hole=0.55,
        marker_colors=[NEG_COLOR if l != positive else POS_COLOR for l in vc.index],
        hovertemplate="<b>%{label}</b><br>%{value:,} passengers<br>%{percent}<extra></extra>",
    ))
    fig.update_layout(**merge_layout(title=dict(text="Satisfaction distribution"), height=380))
    return fig


def satisfaction_by_category(df: pd.DataFrame, col: str, target: str, positive: str) -> go.Figure:
    """Satisfaction rate broken down by a categorical feature."""
    grp = df.groupby(col, observed=True)[target]
    rate = (grp.apply(lambda s: (s == positive).mean() * 100)).sort_values()
    counts = grp.size()
    fig = go.Figure(go.Bar(
        x=rate.values, y=rate.index, orientation="h", marker_color=ACCENT_2,
        text=[f"{v:.1f}%" for v in rate.values], textposition="outside",
        customdata=[counts[i] for i in rate.index],
        hovertemplate="<b>%{y}</b><br>Satisfied: %{x:.1f}%<br>"
                      "Passengers: %{customdata:,}<extra></extra>",
    ))
    fig.update_layout(**merge_layout(
        title=dict(text=f"Satisfaction rate by {col}"),
        xaxis=dict(title="% satisfied", range=[0, 108]),
        height=max(280, 60 * len(rate)),
    ))
    return fig


def rating_comparison(df: pd.DataFrame, rating_cols: list[str], target: str,
                      positive: str) -> go.Figure:
    """Mean service rating by satisfaction group — a grouped bar chart."""
    means = df.groupby(target, observed=True)[rating_cols].mean().T
    fig = go.Figure()
    for col in means.columns:
        fig.add_trace(go.Bar(
            x=means.index, y=means[col], name=col,
            marker_color=POS_COLOR if col == positive else NEG_COLOR,
            hovertemplate="<b>%{x}</b><br>" + str(col) + ": %{y:.2f}<extra></extra>",
        ))
    fig.update_layout(**merge_layout(
        title="Mean service rating by satisfaction group",
        yaxis_title="Mean rating (0–5)", barmode="group",
        xaxis_tickangle=-40, height=480,
    ))
    return fig
