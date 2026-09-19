"""
Exploratory data analysis.

Each figure answers a specific question rather than existing for volume.
All figures are written to outputs/figures/.
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")  # headless backend: works without a display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

import config
from src import utils
from src.feature_engineering import detect_rating_columns

sns.set_theme(style="whitegrid", palette="deep")
plt.rcParams.update({"figure.dpi": 110, "savefig.dpi": 140, "savefig.bbox": "tight"})

POS = config.POSITIVE_LABEL


def _save(fig, name: str) -> str:
    path = config.FIGURES_DIR / name
    fig.savefig(path)
    plt.close(fig)
    return str(path.name)


def _find(df, *keys):
    for c in df.columns:
        low = str(c).lower()
        if all(k in low for k in keys):
            return c
    return None


def run_eda(df: pd.DataFrame, target: str) -> list[str]:
    """Generate the EDA figure set. Returns the list of filenames written."""
    made: list[str] = []
    df = df.copy()
    sat = (df[target] == POS)

    # Q1: How balanced is the target?
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    vc = df[target].value_counts()
    sns.barplot(x=vc.index, y=vc.values, ax=axes[0], hue=vc.index, legend=False)
    axes[0].set_title("Satisfaction distribution (counts)")
    axes[0].set_ylabel("Passengers")
    for i, v in enumerate(vc.values):
        axes[0].text(i, v, f"{v:,}", ha="center", va="bottom")
    axes[1].pie(vc.values, labels=vc.index, autopct="%1.1f%%", startangle=90,
                colors=sns.color_palette("deep", len(vc)))
    axes[1].set_title("Satisfaction share")
    fig.suptitle("Q: Is the target balanced?  -> mild imbalance, no resampling needed")
    made.append(_save(fig, "01_target_distribution.png"))

    # Q2: Do demographics separate the classes?
    age = _find(df, "age")
    gender = _find(df, "gender")
    cust = _find(df, "customer", "type")
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.2))
    if age:
        sns.kdeplot(data=df, x=age, hue=target, fill=True, common_norm=False, ax=axes[0])
        axes[0].set_title("Age distribution by satisfaction")
    if gender:
        g = df.groupby(gender, observed=True)[target].apply(lambda s: (s == POS).mean() * 100)
        sns.barplot(x=g.index, y=g.values, ax=axes[1], hue=g.index, legend=False)
        axes[1].set_title("Satisfaction rate by gender (%)")
        axes[1].set_ylabel("% satisfied")
    if cust:
        g = df.groupby(cust, observed=True)[target].apply(lambda s: (s == POS).mean() * 100)
        sns.barplot(x=g.index, y=g.values, ax=axes[2], hue=g.index, legend=False)
        axes[2].set_title("Satisfaction rate by customer type (%)")
        axes[2].set_ylabel("% satisfied")
    fig.suptitle("Q: Do demographics separate satisfied from dissatisfied passengers?")
    made.append(_save(fig, "02_demographics.png"))

    # Q3: Do travel type, class and distance matter?
    ttype = _find(df, "type", "travel")
    cls = _find(df, "class")
    dist = _find(df, "distance")
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.2))
    for ax, col in zip(axes, [ttype, cls]):
        if col:
            g = df.groupby(col, observed=True)[target].apply(lambda s: (s == POS).mean() * 100)
            sns.barplot(x=g.index, y=g.values, ax=ax, hue=g.index, legend=False)
            ax.set_title(f"Satisfaction rate by {col} (%)")
            ax.set_ylabel("% satisfied")
            ax.tick_params(axis="x", rotation=15)
    if dist:
        sns.boxplot(data=df, x=target, y=dist, ax=axes[2], hue=target, legend=False)
        axes[2].set_title("Flight distance by satisfaction")
    fig.suptitle("Q: Does trip context (purpose, cabin class, distance) drive satisfaction?")
    made.append(_save(fig, "03_travel_context.png"))

    # Q4: Which service ratings track satisfaction most strongly?
    ratings = [c for c in detect_rating_columns(df.drop(columns=[target], errors="ignore"))]
    if ratings:
        rates = {}
        for c in ratings:
            grp = df.groupby(c, observed=True)[target].apply(lambda s: (s == POS).mean() * 100)
            rates[c] = grp
        rate_df = pd.DataFrame(rates).T.sort_index(axis=1)
        fig, ax = plt.subplots(figsize=(10, max(4, 0.42 * len(ratings))))
        sns.heatmap(rate_df, annot=True, fmt=".0f", cmap="RdYlGn", cbar_kws={"label": "% satisfied"}, ax=ax)
        ax.set_xlabel("Rating given (0 = not applicable)")
        ax.set_title("Q: How does each service rating relate to satisfaction?\n"
                     "Note the non-monotonic 0 column -- 0 means 'not applicable', not 'worst'.")
        made.append(_save(fig, "04_service_rating_heatmap.png"))

        # Spread of ratings by outcome for the strongest few
        spread = (rate_df.max(axis=1) - rate_df.min(axis=1)).sort_values(ascending=False)
        top = spread.head(6).index.tolist()
        fig, axes = plt.subplots(2, 3, figsize=(15, 7))
        for ax, c in zip(axes.ravel(), top):
            sns.violinplot(data=df, x=target, y=c, ax=ax, hue=target, legend=False, cut=0)
            ax.set_title(c, fontsize=10)
            ax.set_xlabel("")
            ax.tick_params(axis="x", labelsize=8)
        fig.suptitle("Q: Which service ratings differ most between the two groups?")
        fig.tight_layout()
        made.append(_save(fig, "05_top_service_ratings.png"))

    # Q5: Do delays matter, and how skewed are they?
    dep = _find(df, "departure", "delay")
    arr = _find(df, "arrival", "delay")
    if dep and arr:
        fig, axes = plt.subplots(1, 3, figsize=(16, 4.2))
        sns.histplot(df[dep].clip(upper=180), bins=40, ax=axes[0])
        axes[0].set_title(f"{dep} (clipped at 180 min)")
        axes[0].set_yscale("log")
        sns.histplot(df[arr].clip(upper=180), bins=40, ax=axes[1], color="darkorange")
        axes[1].set_title(f"{arr} (clipped at 180 min)")
        axes[1].set_yscale("log")
        tmp = df.assign(**{"Total Delay": df[dep].fillna(0) + df[arr].fillna(0)})
        bins = [-0.1, 0, 15, 60, 180, np.inf]
        labels = ["0 min", "1-15", "16-60", "61-180", ">180"]
        tmp["Delay band"] = pd.cut(tmp["Total Delay"], bins=bins, labels=labels)
        g = tmp.groupby("Delay band", observed=True)[target].apply(lambda s: (s == POS).mean() * 100)
        sns.barplot(x=g.index, y=g.values, ax=axes[2], hue=g.index, legend=False)
        axes[2].set_title("Satisfaction rate by total delay (%)")
        axes[2].set_ylabel("% satisfied")
        fig.suptitle("Q: How do delays relate to satisfaction? (delays are heavily right-skewed)")
        made.append(_save(fig, "06_delays.png"))

    # Q6: How do numeric features correlate with each other and the target?
    num = df.select_dtypes(include="number").copy()
    num["__target__"] = sat.astype(int)
    corr = num.corr(numeric_only=True)
    fig, ax = plt.subplots(figsize=(12, 9))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, cmap="coolwarm", center=0, annot=False,
                square=False, linewidths=0.4, cbar_kws={"label": "Pearson r"}, ax=ax)
    ax.set_title("Q: Which numeric features are correlated? (__target__ = Satisfied)")
    made.append(_save(fig, "07_correlation_heatmap.png"))

    # Q7: Ranked linear association with the target
    tcorr = corr["__target__"].drop("__target__").sort_values()
    fig, ax = plt.subplots(figsize=(8, max(4, 0.32 * len(tcorr))))
    colors = ["#c44e52" if v < 0 else "#4c72b0" for v in tcorr.values]
    ax.barh(tcorr.index, tcorr.values, color=colors)
    ax.set_xlabel("Pearson correlation with 'Satisfied'")
    ax.set_title("Q: Which features are most linearly associated with satisfaction?")
    ax.axvline(0, color="black", lw=0.8)
    made.append(_save(fig, "08_target_correlation.png"))

    utils.info(f"{len(made)} EDA figures written to {config.FIGURES_DIR.relative_to(config.ROOT)}")
    return made
