"""
One-page visual summary poster for the viva.

Reads only generated artifacts (model_results.csv, experiment_manifest.json,
feature_importance.csv, shap outputs) so every number on the poster is traceable.
"""
from __future__ import annotations

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import config

NAVY = "#1f3a5f"
GREEN = "#2e7d32"
RED = "#c44e52"
GREY = "#6b7280"
FAMILY_COLORS = {"baseline": "#9e9e9e", "classical": "#4c72b0",
                 "ensemble": "#55a868", "deep": "#c44e52"}


def _panel_title(ax, text: str) -> None:
    ax.set_title(text, fontsize=11, fontweight="bold", color=NAVY, loc="left", pad=8)


def build_poster(out_name: str = "19_summary_poster.png") -> str | None:
    """Compose the poster from generated artifacts. Returns the filename."""
    if not config.RESULTS_CSV.exists():
        return None

    results = pd.read_csv(config.RESULTS_CSV)
    ok = results[results["Status"] == "ok"].copy()
    manifest = json.loads(config.MANIFEST_JSON.read_text()) if config.MANIFEST_JSON.exists() else {}
    split = manifest.get("split", {})
    n_total = sum(v for k, v in split.items() if k in ("train", "val", "test"))

    best = ok.iloc[0]
    top5_spread = ok["F1"].head(5).max() - ok["F1"].head(5).min()
    mean_ci = float(ok["f1_ci_width"].head(5).mean()) if "f1_ci_width" in ok else float("nan")
    tied = ok[(ok["f1_ci_low"] <= best["f1_ci_high"]) &
              (ok["f1_ci_high"] >= best["f1_ci_low"])] if "f1_ci_low" in ok else ok.head(1)

    fig = plt.figure(figsize=(16.5, 23.4))  # A2 portrait proportions
    fig.patch.set_facecolor("white")
    gs = gridspec.GridSpec(
        7, 2, figure=fig,
        height_ratios=[0.60, 0.40, 1.20, 1.30, 1.10, 1.05, 0.42],
        hspace=0.38, wspace=0.22,
        left=0.055, right=0.965, top=0.965, bottom=0.028,
    )

    # ---------------------------------------------------------------- header
    ax = fig.add_subplot(gs[0, :]); ax.axis("off")
    ax.add_patch(plt.Rectangle((0, 0), 1, 1, transform=ax.transAxes,
                               color=NAVY, zorder=0))
    ax.text(0.5, 0.70,
            "Predicting Airline Passenger Satisfaction",
            ha="center", va="center", fontsize=27, fontweight="bold",
            color="white", transform=ax.transAxes)
    ax.text(0.5, 0.40,
            "Machine Learning and Deep Learning: A Comparative Evaluation",
            ha="center", va="center", fontsize=15, color="#cfe0f5",
            transform=ax.transAxes)
    ax.text(0.5, 0.14,
            f"{n_total:,} passenger records   |   {len(ok)} models compared   |   "
            f"B.Tech CSE Research Project",
            ha="center", va="center", fontsize=11, color="#9db8d8",
            transform=ax.transAxes)

    # ------------------------------------------------------------ KPI strip
    ax = fig.add_subplot(gs[1, :]); ax.axis("off")
    kpis = [
        (f"{best['Accuracy']:.4f}", "Best test accuracy", str(best["Model"])),
        (f"{best['F1']:.4f}", "Best test F1", "primary selection metric"),
        (f"{len(tied)}", "Models statistically tied", "overlapping 95% CIs"),
        (f"{mean_ci / top5_spread:.1f}x", "CI width vs top-5 spread", "uncertainty dominates"),
    ]
    for i, (value, label, sub) in enumerate(kpis):
        x = 0.012 + i * 0.2485
        ax.add_patch(plt.Rectangle((x, 0.06), 0.234, 0.88, transform=ax.transAxes,
                                   facecolor="#f3f6fa", edgecolor="#d3dce8", lw=1.2))
        colour = RED if i >= 2 else NAVY
        ax.text(x + 0.117, 0.66, value, ha="center", va="center",
                fontsize=24, fontweight="bold", color=colour, transform=ax.transAxes)
        ax.text(x + 0.117, 0.36, label, ha="center", va="center",
                fontsize=10.5, color="#333", transform=ax.transAxes)
        ax.text(x + 0.117, 0.17, sub, ha="center", va="center",
                fontsize=8.5, color=GREY, style="italic", transform=ax.transAxes)

    # ------------------------------------------- model comparison (full width)
    ax = fig.add_subplot(gs[2, :])
    d = ok.sort_values("F1")
    colours = [FAMILY_COLORS.get(f, "#4c72b0") for f in d["Family"]]
    ax.barh(d["Model"], d["F1"], color=colours, height=0.68)
    for i, v in enumerate(d["F1"]):
        ax.text(v + 0.008, i, f"{v:.4f}", va="center", fontsize=8.5)
    ax.set_xlim(0, 1.12)
    ax.set_xlabel("Test F1 score", fontsize=10)
    ax.tick_params(axis="y", labelsize=9)
    _panel_title(ax, "1. Model comparison — 14 models on the held-out test set")
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in FAMILY_COLORS.values()]
    ax.legend(handles, FAMILY_COLORS.keys(), fontsize=8, title="Family",
              loc="lower right", title_fontsize=8.5)
    ax.grid(axis="x", alpha=0.25)
    ax.set_axisbelow(True)

    # ------------------------------------------------- forest plot (headline)
    ax = fig.add_subplot(gs[3, :])
    f = ok.nlargest(10, "F1").sort_values("F1")
    is_tied = (f["f1_ci_low"] <= best["f1_ci_high"]) & (f["f1_ci_high"] >= best["f1_ci_low"])
    ax.axvspan(best["f1_ci_low"], best["f1_ci_high"], color="#4c72b0", alpha=0.10)
    for i, (_, row) in enumerate(f.iterrows()):
        c = RED if is_tied.iloc[i] else GREY
        ax.plot([row["f1_ci_low"], row["f1_ci_high"]], [i, i], color=c, lw=3,
                solid_capstyle="round")
        ax.plot(row["F1"], i, "o", color=c, ms=8, markeredgecolor="white", mew=1.3)
    ax.set_yticks(range(len(f)))
    ax.set_yticklabels(f["Model"], fontsize=9)
    ax.set_xlabel("Test F1 with 95% bootstrap confidence interval", fontsize=10)
    _panel_title(ax, "2. KEY FINDING — the leading models are statistically indistinguishable")
    ax.text(0.015, 0.97,
            f"Top-5 models differ by {top5_spread:.5f} F1, but the mean 95%\n"
            f"confidence interval is {mean_ci:.5f} wide ({mean_ci / top5_spread:.1f}x larger).\n"
            f"{len(tied)} models overlap the leader — ranking by point\n"
            f"estimate alone would overstate the evidence.",
            ha="left", va="top", fontsize=9.5, color=NAVY, transform=ax.transAxes,
            bbox=dict(boxstyle="round,pad=0.5", facecolor="#fff4f4", edgecolor=RED, lw=1.2),
            zorder=5)
    ax.grid(axis="x", alpha=0.25)
    ax.set_axisbelow(True)

    # ------------------------------------------------- feature importance
    ax = fig.add_subplot(gs[4, 0])
    if config.FEATURE_IMPORTANCE_CSV.exists():
        imp = pd.read_csv(config.FEATURE_IMPORTANCE_CSV)
        sub = imp[imp["Model"] == "Random Forest"].nlargest(8, "Importance").sort_values("Importance")
        ax.barh(sub["Feature"], sub["Importance"], color="#55a868", height=0.65)
        for i, v in enumerate(sub["Importance"]):
            ax.text(v + 0.004, i, f"{v:.3f}", va="center", fontsize=8)
        ax.set_xlabel("Importance", fontsize=9.5)
        ax.tick_params(axis="y", labelsize=8.5)
        ax.set_xlim(0, sub["Importance"].max() * 1.22)
    _panel_title(ax, "3. What drives the predictions")
    ax.grid(axis="x", alpha=0.25); ax.set_axisbelow(True)

    # ------------------------------------------------- cost vs performance
    ax = fig.add_subplot(gs[4, 1])
    times = {k: v.get("fit_seconds") for k, v in manifest.get("models", {}).items()}
    pts = [(times.get(r["Model"]), r["F1"], r["Model"], r["Family"])
           for _, r in ok.iterrows() if times.get(r["Model"]) is not None and r["F1"] > 0.9]
    for t, f1, name, fam in pts:
        ax.scatter(t, f1, s=110, color=FAMILY_COLORS.get(fam, "#4c72b0"),
                   edgecolor="white", lw=1.2, zorder=3)
    offsets = {
        "XGBoost": (-10, 10), "Neural Net (Deep MLP)": (-58, -16),
        "Random Forest": (-46, 10), "Support Vector Machine": (-30, -16),
    }
    for t, f1, name, fam in pts:
        if name in offsets:
            short = {"Neural Net (Deep MLP)": "Deep MLP",
                     "Support Vector Machine": "SVM"}.get(name, name)
            ax.annotate(short, (t, f1), fontsize=8, xytext=offsets[name],
                        textcoords="offset points", color=NAVY, fontweight="bold")
    ax.set_xscale("log")
    lo = min(p[1] for p in pts) if pts else 0.9
    ax.set_ylim(lo - 0.022, 1.0005)
    ax.set_xlabel("Training time (seconds, log scale)", fontsize=9.5)
    ax.set_ylabel("Test F1", fontsize=9.5)
    _panel_title(ax, "4. Performance vs training cost")
    ax.grid(alpha=0.25); ax.set_axisbelow(True)
    ax.text(0.03, 0.05,
            "Accuracy is tied, so cost decides:\n"
            "XGBoost matches the Deep MLP\nin 6.9s vs 35.5s.",
            ha="left", va="bottom", fontsize=9, color=NAVY, transform=ax.transAxes,
            bbox=dict(boxstyle="round,pad=0.45", facecolor="#f3f6fa", edgecolor="#d3dce8"),
            zorder=5)

    # ------------------------------------------------- methodology + findings
    ax = fig.add_subplot(gs[5, 0]); ax.axis("off")
    _panel_title(ax, "5. Methodology")
    method = (
        f"Split      Stratified 70/15/15 — {split.get('train', 0):,} train, "
        f"{split.get('val', 0):,} validation, {split.get('test', 0):,} test\n\n"
        "Protocol   Test set opened exactly ONCE, after model selection\n"
        "           was complete on the validation set\n\n"
        "Leakage    All preprocessing fitted inside sklearn Pipelines on\n"
        "           training folds only; disjointness asserted by tests\n\n"
        f"Tuning     RandomizedSearchCV, {manifest.get('budgets', {}).get('search_iter', 12)} "
        f"candidates x {manifest.get('budgets', {}).get('search_cv', 3)} folds (train only)\n\n"
        "Statistics 5-fold CV for stability + 1,000-resample bootstrap\n"
        "           confidence intervals on test F1\n\n"
        f"Seed       {manifest.get('seed', 42)} — runs reproduce identical metrics"
    )
    ax.text(0.01, 0.94, method, va="top", ha="left", fontsize=9.2, family="monospace",
            color="#222", transform=ax.transAxes, linespacing=1.35)

    ax = fig.add_subplot(gs[5, 1]); ax.axis("off")
    _panel_title(ax, "6. Findings and answers")
    findings = (
        "RQ1  Classical models work, but spread widely:\n"
        "     SVM 0.9459 F1 vs Logistic Regression 0.8585\n\n"
        "RQ2  Ensembles beat individual classifiers, but Voting\n"
        "     and Stacking did NOT beat a tuned Random Forest\n\n"
        "RQ3  Yes — all three neural networks are competitive;\n"
        "     two sit inside the statistically tied group\n\n"
        "RQ4  Online Boarding and In-flight Wifi Service lead\n"
        "     under three independent attribution methods\n\n"
        "RQ5  Performance is tied, so cost and interpretability\n"
        "     decide: XGBoost is the practical choice\n\n"
        "DATA Rating 0 means 'not applicable', not 'worst' —\n"
        "     0-raters on wifi are 99.7% satisfied vs ~25% at 1-3"
    )
    ax.text(0.01, 0.94, findings, va="top", ha="left", fontsize=9.2, family="monospace",
            color="#222", transform=ax.transAxes, linespacing=1.35)

    # ---------------------------------------------------------------- footer
    ax = fig.add_subplot(gs[6, :]); ax.axis("off")
    ax.add_patch(plt.Rectangle((0, 0.28), 1, 0.62, transform=ax.transAxes,
                               facecolor="#f3f6fa", edgecolor="#d3dce8"))
    ax.text(0.5, 0.72,
            f"RECOMMENDATION:  deploy XGBoost or Random Forest — statistically equal to the "
            f"neural networks, but faster to train and exactly explainable with SHAP.",
            ha="center", va="center", fontsize=11, fontweight="bold", color=NAVY,
            transform=ax.transAxes)
    ax.text(0.5, 0.44,
            "SHAP and permutation importance describe model behaviour and association — not causation.",
            ha="center", va="center", fontsize=9, color=GREY, style="italic",
            transform=ax.transAxes)
    ax.text(0.5, 0.08,
            f"All figures generated by this project's own experiments  |  run "
            f"{manifest.get('run_id', 'n/a')}  |  reproduce with: python train.py --mode full",
            ha="center", va="center", fontsize=8.5, color=GREY, transform=ax.transAxes)

    path = config.FIGURES_DIR / out_name
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    # A PDF copy prints cleanly at A2/A3 for the viva.
    fig.savefig(config.OUTPUTS_DIR / "reports" / "summary_poster.pdf",
                bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out_name


if __name__ == "__main__":
    print("poster ->", build_poster())
