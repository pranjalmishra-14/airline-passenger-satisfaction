"""
Central configuration for the Airline Passenger Satisfaction project.

Every tunable lives here so experiments are reproducible and auditable.
Nothing in src/ or app/ should hard-code a seed, path, or budget.
"""
from pathlib import Path

# ----------------------------------------------------------------------------
# Reproducibility
# ----------------------------------------------------------------------------
RANDOM_STATE = 42

# ----------------------------------------------------------------------------
# Paths
# ----------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
DATA_FILE = DATA_DIR / "airline_passenger_satisfaction.csv"

MODELS_DIR = ROOT / "models"
OUTPUTS_DIR = ROOT / "outputs"
FIGURES_DIR = OUTPUTS_DIR / "figures"
REPORTS_DIR = OUTPUTS_DIR / "reports"
SHAP_DIR = OUTPUTS_DIR / "shap_results"

RESULTS_CSV = OUTPUTS_DIR / "model_results.csv"
DL_RESULTS_CSV = OUTPUTS_DIR / "deep_learning_results.csv"
FEATURE_IMPORTANCE_CSV = OUTPUTS_DIR / "feature_importance.csv"
CV_RESULTS_CSV = OUTPUTS_DIR / "cv_results.csv"
MANIFEST_JSON = OUTPUTS_DIR / "experiment_manifest.json"

FINAL_MODEL_PATH = MODELS_DIR / "best_model.pkl"
SCHEMA_PATH = MODELS_DIR / "feature_schema.json"

for _d in (DATA_DIR, MODELS_DIR, OUTPUTS_DIR, FIGURES_DIR, REPORTS_DIR, SHAP_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ----------------------------------------------------------------------------
# Data split  (strict 70 / 15 / 15, stratified)
#   train -> fits models; the ONLY data CV and hyperparameter search ever see
#   val   -> early stopping, model selection, threshold choice
#   test  -> opened exactly once, at final evaluation
# ----------------------------------------------------------------------------
TEST_SIZE = 0.15
VAL_SIZE = 0.15          # fraction of the FULL dataset
TRAIN_SIZE = 1.0 - TEST_SIZE - VAL_SIZE

TARGET_CANDIDATES = ("Satisfaction", "satisfaction")
POSITIVE_LABEL = "Satisfied"          # class 1
NEGATIVE_LABEL = "Neutral or Dissatisfied"  # class 0

# ----------------------------------------------------------------------------
# Mode budgets  (lean scope: full run should finish well under 10 minutes)
# ----------------------------------------------------------------------------
MODE_BUDGETS = {
    "fast": {
        "sample_rows": 15_000,   # subsample for quick debugging
        "search_iter": 3,        # RandomizedSearchCV candidates
        "search_cv": 2,
        "cv_folds": 3,
        "epochs": 5,
        "bootstrap_n": 200,
        "enable_svm": True,
    },
    "full": {
        "sample_rows": None,     # use all 129,880 rows
        "search_iter": 12,       # lean tuning budget
        "search_cv": 3,
        "cv_folds": 5,
        "epochs": 60,            # EarlyStopping ends this well before 60
        "bootstrap_n": 1000,
        "enable_svm": True,
    },
}

# ----------------------------------------------------------------------------
# Deep learning
# ----------------------------------------------------------------------------
BATCH_SIZE = 256
LEARNING_RATE = 1e-3
EARLY_STOPPING_PATIENCE = 8

# ----------------------------------------------------------------------------
# Reference benchmarks (context only -- NEVER a target to optimise toward)
# ----------------------------------------------------------------------------
REFERENCE_PAPER = {
    "source": "ICSADL 2026, Random Forest",
    "accuracy": 0.9583,
    "f1": 0.9515,
    "note": "External reference point for discussion only; not a goal.",
}
