"""Model-evaluation figures: training curves, confusion matrices, ROC/PR, comparisons."""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

import config
from src import evaluate as ev

sns.set_theme(style="whitegrid", palette="deep")
plt.rcParams.update({"figure.dpi": 110, "savefig.dpi": 140, "savefig.bbox": "tight"})


def _save(fig, name: str) -> str:
    fig.savefig(config.FIGURES_DIR / name)
    plt.close(fig)
    return name


def plot_training_history(histories: dict) -> list[str]:
    """Accuracy-vs-epoch and loss-vs-epoch for every neural network."""
    made = []
    if not histories:
        return made
    n = len(histories)
    fig, axes = plt.subplots(2, n, figsize=(5.2 * n, 8), squeeze=False)
    for j, (name, h) in enumerate(histories.items()):
        ep = range(1, len(h["loss"]) + 1)
        axes[0][j].plot(ep, h["accuracy"], label="train")
        axes[0][j].plot(ep, h["val_accuracy"], label="validation")
        axes[0][j].set_title(f"{name}\nAccuracy vs epochs", fontsize=10)
        axes[0][j].set_xlabel("Epoch"); axes[0][j].set_ylabel("Accuracy"); axes[0][j].legend()
        axes[1][j].plot(ep, h["loss"], label="train")
        axes[1][j].plot(ep, h["val_loss"], label="validation")
        axes[1][j].set_title("Loss vs epochs", fontsize=10)
        axes[1][j].set_xlabel("Epoch"); axes[1][j].set_ylabel("Binary cross-entropy")
        axes[1][j].legend()
    fig.suptitle("Deep learning training history (early stopping restores best weights)")
    fig.tight_layout()
    made.append(_save(fig, "10_dl_training_history.png"))
    return made


def plot_confusion_matrices(cms: dict) -> list[str]:
    """Grid of confusion matrices (counts + row-normalised percentages)."""
    if not cms:
        return []
    n = len(cms)
    ncol = min(4, n)
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.4 * ncol, 3.9 * nrow), squeeze=False)
    for ax, (name, cm) in zip(axes.ravel(), cms.items()):
        arr = cm.values if hasattr(cm, "values") else np.asarray(cm)
        pct = arr / arr.sum(axis=1, keepdims=True) * 100
        labels = np.array([[f"{v:,}\n({p:.1f}%)" for v, p in zip(r, pr)]
                           for r, pr in zip(arr, pct)])
        sns.heatmap(arr, annot=labels, fmt="", cmap="Blues", cbar=False, ax=ax,
                    xticklabels=["Pred Neutral/Dis.", "Pred Satisfied"],
                    yticklabels=["True Neutral/Dis.", "True Satisfied"])
        ax.set_title(name, fontsize=10)
        ax.tick_params(labelsize=8)
    for ax in axes.ravel()[n:]:
        ax.axis("off")
    fig.suptitle("Confusion matrices on the held-out test set")
    fig.tight_layout()
    return [_save(fig, "11_confusion_matrices.png")]


def plot_roc_pr(curve_data: dict) -> list[str]:
    """Overlaid ROC and Precision-Recall curves for all models with scores."""
    if not curve_data:
        return []
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.6))
    for name, (y_true, y_score) in curve_data.items():
        fpr, tpr, a = ev.roc_points(y_true, y_score)
        axes[0].plot(fpr, tpr, lw=1.6, label=f"{name} (AUC={a:.4f})")
        rec, prec, pa = ev.pr_points(y_true, y_score)
        axes[1].plot(rec, prec, lw=1.6, label=f"{name} (AP-AUC={pa:.4f})")
    axes[0].plot([0, 1], [0, 1], "k--", lw=1, label="Chance")
    axes[0].set_xlabel("False positive rate"); axes[0].set_ylabel("True positive rate")
    axes[0].set_title("ROC curves (test set)"); axes[0].legend(fontsize=7, loc="lower right")
    axes[1].set_xlabel("Recall"); axes[1].set_ylabel("Precision")
    axes[1].set_title("Precision-Recall curves (test set)")
    axes[1].legend(fontsize=7, loc="lower left")
    fig.tight_layout()
    return [_save(fig, "12_roc_pr_curves.png")]


def plot_model_comparison(results: pd.DataFrame) -> list[str]:
    """Bar charts comparing accuracy and F1 across every model."""
    df = results[results["Status"] == "ok"].copy() if "Status" in results else results.copy()
    if df.empty:
        return []
    df = df.sort_values("F1", ascending=True)
    made = []
    fig, axes = plt.subplots(1, 2, figsize=(15, max(4.5, 0.42 * len(df))))
    palette = {"baseline": "#999999", "classical": "#4c72b0",
               "ensemble": "#55a868", "deep": "#c44e52"}
    colors = [palette.get(f, "#4c72b0") for f in df.get("Family", ["classical"] * len(df))]
    axes[0].barh(df["Model"], df["Accuracy"], color=colors)
    axes[0].set_xlabel("Accuracy"); axes[0].set_title("Test accuracy by model")
    axes[0].set_xlim(0, 1.08)
    for i, v in enumerate(df["Accuracy"]):
        axes[0].text(v + 0.005, i, f"{v:.4f}", va="center", fontsize=8)
    axes[1].barh(df["Model"], df["F1"], color=colors)
    axes[1].set_xlabel("F1 score"); axes[1].set_title("Test F1 by model")
    axes[1].set_xlim(0, 1.08)
    for i, v in enumerate(df["F1"]):
        axes[1].text(v + 0.005, i, f"{v:.4f}", va="center", fontsize=8)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in palette.values()]
    # Legend outside the axes so it cannot cover the value labels.
    axes[1].legend(handles, palette.keys(), title="Family", fontsize=8,
                   loc="upper left", bbox_to_anchor=(1.01, 1.0), borderaxespad=0)
    fig.suptitle("Model comparison on the held-out test set")
    fig.tight_layout()
    made.append(_save(fig, "13_model_comparison.png"))
    return made


def plot_feature_importance(importance: pd.DataFrame, top_n: int = 20) -> list[str]:
    """Ranked feature-importance chart, one panel per model."""
    if importance.empty:
        return []
    models = importance["Model"].unique()
    fig, axes = plt.subplots(1, len(models), figsize=(7.5 * len(models), 6.5), squeeze=False)
    for ax, mname in zip(axes[0], models):
        sub = (importance[importance["Model"] == mname]
               .nlargest(top_n, "Importance").sort_values("Importance"))
        ax.barh(sub["Feature"], sub["Importance"], color="#55a868")
        ax.set_title(f"{mname}: top {top_n} features", fontsize=11)
        ax.set_xlabel("Importance (impurity-based)")
        ax.tick_params(axis="y", labelsize=8)
    fig.suptitle("Feature importance from tree-based models")
    fig.tight_layout()
    return [_save(fig, "14_feature_importance.png")]


def plot_ci_forest(results: pd.DataFrame, top_n: int = 10) -> list[str]:
    """
    Forest plot of bootstrap confidence intervals on test F1.

    This visualises the project's headline finding: the leading models' intervals
    overlap, so their ranking differences are inside the noise floor.
    """
    if "f1_ci_low" not in results.columns:
        return []
    ok = results[results["Status"] == "ok"].dropna(subset=["f1_ci_low"])
    if ok.empty:
        return []

    ok = ok.nlargest(top_n, "F1").sort_values("F1")
    best = ok.iloc[-1]
    # A model is "tied" with the leader when the two intervals overlap.
    tied = (ok["f1_ci_low"] <= best["f1_ci_high"]) & (ok["f1_ci_high"] >= best["f1_ci_low"])

    fig, ax = plt.subplots(figsize=(11, max(4.5, 0.5 * len(ok))))
    band_lo, band_hi = best["f1_ci_low"], best["f1_ci_high"]
    ax.axvspan(band_lo, band_hi, color="#4c72b0", alpha=0.10,
               label=f"95% CI of best model ({best['Model']})")

    for i, (_, row) in enumerate(ok.iterrows()):
        colour = "#c44e52" if tied.iloc[i] else "#888888"
        ax.plot([row["f1_ci_low"], row["f1_ci_high"]], [i, i],
                color=colour, lw=2.4, solid_capstyle="round", zorder=3)
        ax.plot(row["F1"], i, "o", color=colour, ms=8,
                markeredgecolor="white", markeredgewidth=1.2, zorder=4)
        ax.text(row["f1_ci_high"] + 0.0004, i, f"{row['F1']:.4f}",
                va="center", fontsize=8, color="#333")

    ax.set_yticks(range(len(ok)))
    ax.set_yticklabels(ok["Model"], fontsize=9)
    ax.set_xlabel("Test F1 score with 95% bootstrap confidence interval")
    spread = ok["F1"].tail(5).max() - ok["F1"].tail(5).min()
    width = float(ok["f1_ci_width"].tail(5).mean())
    ax.set_title(
        "Are the top models actually different?\n"
        f"Top-5 spread {spread:.5f} vs mean CI width {width:.5f} "
        f"({width / spread:.1f}x larger) — {int(tied.sum())} models are statistically indistinguishable",
        fontsize=11,
    )
    handles = [
        plt.Line2D([], [], color="#c44e52", lw=2.4, marker="o",
                   label="Overlaps the best model (tied)"),
        plt.Line2D([], [], color="#888888", lw=2.4, marker="o",
                   label="Separated from the best model"),
    ]
    ax.legend(handles=handles, fontsize=8, loc="lower right")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    return [_save(fig, "18_ci_forest_plot.png")]
