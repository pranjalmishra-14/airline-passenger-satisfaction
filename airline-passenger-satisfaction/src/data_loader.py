"""
Dataset loading, quality analysis, and the single authoritative train/val/test split.

Design rules enforced here:
  * Schema-agnostic: column names are inferred, never hard-coded.
  * There is exactly ONE split function in the project, so no module can re-split
    the data and accidentally leak the test set.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from sklearn.model_selection import train_test_split

import config


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
def load_raw(path=None) -> pd.DataFrame:
    """Load the raw CSV with no cleaning applied."""
    path = path or config.DATA_FILE
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {path}. See data/README.md for how to obtain it."
        )
    return pd.read_csv(path)


def find_target(df: pd.DataFrame) -> str:
    """Locate the target column without assuming an exact spelling."""
    for cand in config.TARGET_CANDIDATES:
        if cand in df.columns:
            return cand
    lowered = {c.lower(): c for c in df.columns}
    if "satisfaction" in lowered:
        return lowered["satisfaction"]
    raise KeyError(
        f"No target column found. Looked for {config.TARGET_CANDIDATES}; "
        f"available columns: {list(df.columns)}"
    )


def detect_droppable(df: pd.DataFrame, target: str) -> dict[str, str]:
    """
    Identify columns that are genuinely irrelevant to modelling.

    Returns {column: reason}. Only unnamed index columns, unique identifiers and
    constant columns qualify -- nothing is dropped merely because it looks odd.
    """
    drops: dict[str, str] = {}
    n = len(df)
    for col in df.columns:
        if col == target:
            continue
        s = df[col]
        if str(col).strip() == "" or str(col).lower().startswith("unnamed"):
            drops[col] = "unnamed index column carried over from the CSV"
        elif s.nunique(dropna=False) <= 1:
            drops[col] = "constant column (no predictive information)"
        elif str(col).lower() in {"id", "index"} and s.nunique() == n:
            drops[col] = f"unique identifier ({n} distinct values, one per row)"
        elif pd.api.types.is_integer_dtype(s) and s.nunique() == n and s.is_monotonic_increasing:
            drops[col] = "monotonically increasing unique integer (row identifier)"
    return drops


# ---------------------------------------------------------------------------
# Quality report
# ---------------------------------------------------------------------------
def quality_report(df: pd.DataFrame, target: str, drops: dict[str, str]) -> str:
    """Build a markdown data-quality report from the ACTUAL dataframe."""
    n_rows, n_cols = df.shape
    missing = df.isna().sum()
    missing = missing[missing > 0].sort_values(ascending=False)

    feature_df = df.drop(columns=list(drops))
    dup_excl_id = int(feature_df.duplicated().sum())

    num_cols = df.select_dtypes(include="number").columns.tolist()
    cat_cols = df.select_dtypes(exclude="number").columns.tolist()

    lines = [
        "# Data Quality Report",
        "",
        "_Generated automatically from the dataset; every figure below is computed, not assumed._",
        "",
        "## 1. Dimensions",
        "",
        f"- Rows: **{n_rows:,}**",
        f"- Columns: **{n_cols}**",
        f"- Target column: **`{target}`**",
        "",
        "## 2. Missing values",
        "",
    ]
    if missing.empty:
        lines.append("No missing values.")
    else:
        lines += ["| Column | Missing | % |", "|---|---|---|"]
        lines += [
            f"| {c} | {v:,} | {v / n_rows * 100:.2f}% |" for c, v in missing.items()
        ]
    lines += [
        "",
        "## 3. Duplicates",
        "",
        f"- Fully duplicated rows (excluding dropped identifier columns): **{dup_excl_id:,}**",
        "",
        "## 4. Columns removed before modelling",
        "",
    ]
    if drops:
        lines += ["| Column | Reason for removal |", "|---|---|"]
        lines += [f"| `{c}` | {r} |" for c, r in drops.items()]
    else:
        lines.append("No columns removed.")

    lines += ["", "## 5. Target distribution", ""]
    vc = df[target].value_counts()
    lines += ["| Class | Count | Share |", "|---|---|---|"]
    lines += [f"| {k} | {v:,} | {v / n_rows * 100:.2f}% |" for k, v in vc.items()]
    imbalance = vc.max() / vc.min()
    lines += [
        "",
        f"Imbalance ratio (majority:minority) = **{imbalance:.2f}:1**. "
        + (
            "Mild imbalance -- resampling (e.g. SMOTE) is not required; "
            "stratified splitting is sufficient."
            if imbalance < 1.5
            else "Noticeable imbalance -- class weighting considered."
        ),
        "",
        "## 6. Feature types",
        "",
        f"- Numeric columns ({len(num_cols)}): {', '.join(f'`{c}`' for c in num_cols)}",
        "",
        f"- Categorical columns ({len(cat_cols)}): {', '.join(f'`{c}`' for c in cat_cols)}",
        "",
        "## 7. Numeric summary",
        "",
        df[num_cols].describe().T.round(2).to_markdown(),
        "",
        "## 8. Categorical levels",
        "",
    ]
    for c in cat_cols:
        levels = df[c].value_counts().to_dict()
        pretty = ", ".join(f"{k} ({v:,})" for k, v in levels.items())
        lines.append(f"- `{c}`: {pretty}")

    # Rating-scale observation: 0 behaves as "not applicable", not "worst".
    rating_cols = [
        c for c in num_cols
        if df[c].dropna().between(0, 5).all() and df[c].nunique() <= 6 and c != target
    ]
    if rating_cols:
        lines += [
            "",
            "## 9. Service-rating scale note",
            "",
            "Service ratings are documented as 1-5 but contain **0** values. Measured "
            "satisfaction rate by rating shows 0 does *not* behave like 'worst' -- it behaves "
            "like *not applicable*:",
            "",
        ]
        pos = config.POSITIVE_LABEL
        header_done = False
        for c in rating_cols[:5]:
            rates = df.groupby(c, observed=True)[target].apply(lambda g: (g == pos).mean())
            if 0 not in rates.index:
                continue
            if not header_done:
                cols = " | ".join(str(i) for i in sorted(rates.index))
                lines += [f"| Feature | {cols} |", "|---" * (len(rates) + 1) + "|"]
                header_done = True
            vals = " | ".join(f"{rates[i]*100:.1f}%" for i in sorted(rates.index))
            lines.append(f"| {c} | {vals} |")
        lines += [
            "",
            "**Decision:** ratings are kept as plain 0-5 numeric values (consistent with the "
            "reference literature). This observation is documented and discussed in the report "
            "rather than encoded as a separate indicator.",
        ]

    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# The single authoritative split
# ---------------------------------------------------------------------------
@dataclass
class DataSplits:
    """Train / validation / test splits plus metadata. Created in exactly one place."""
    X_train: pd.DataFrame
    X_val: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_val: pd.Series
    y_test: pd.Series
    target: str
    dropped: dict[str, str]

    @property
    def sizes(self) -> dict[str, int]:
        return {
            "train": len(self.X_train),
            "val": len(self.X_val),
            "test": len(self.X_test),
        }

    def assert_disjoint(self) -> None:
        """Guarantee no row appears in more than one split."""
        tr, va, te = set(self.X_train.index), set(self.X_val.index), set(self.X_test.index)
        assert not (tr & va), "train/val overlap detected"
        assert not (tr & te), "train/test overlap detected"
        assert not (va & te), "val/test overlap detected"


def make_splits(df: pd.DataFrame, sample_rows: int | None = None) -> DataSplits:
    """
    Produce the stratified 70/15/15 train/val/test split.

    This is the ONLY place the dataset is split. Preprocessing is fitted on train
    only (inside pipelines), and the test set is not read until final evaluation.
    """
    target = find_target(df)
    drops = detect_droppable(df, target)
    df = df.drop(columns=list(drops))

    if sample_rows and sample_rows < len(df):
        df = df.sample(n=sample_rows, random_state=config.RANDOM_STATE).reset_index(drop=True)

    y = (df[target] == config.POSITIVE_LABEL).astype(int)
    X = df.drop(columns=[target])

    # First carve off the test set, then split the remainder into train/val.
    X_tr_val, X_test, y_tr_val, y_test = train_test_split(
        X, y,
        test_size=config.TEST_SIZE,
        stratify=y,
        random_state=config.RANDOM_STATE,
    )
    val_relative = config.VAL_SIZE / (1.0 - config.TEST_SIZE)
    X_train, X_val, y_train, y_val = train_test_split(
        X_tr_val, y_tr_val,
        test_size=val_relative,
        stratify=y_tr_val,
        random_state=config.RANDOM_STATE,
    )

    splits = DataSplits(
        X_train=X_train, X_val=X_val, X_test=X_test,
        y_train=y_train, y_val=y_val, y_test=y_test,
        target=target, dropped=drops,
    )
    splits.assert_disjoint()
    return splits
