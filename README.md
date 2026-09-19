# Predicting Airline Passenger Satisfaction Using Machine Learning and Deep Learning: A Comparative Evaluation

A research-grade B.Tech Computer Science project that compares **classical machine learning**,
**ensemble methods** and **deep learning** for predicting airline passenger satisfaction, with
explainable AI and an interactive Streamlit application.

Every number in this repository — in the report, the figures and the app — was produced by
running the code here. Nothing is copied from any paper.

---

## Overview

The system predicts whether a passenger is **Satisfied** or **Neutral or Dissatisfied** from
demographic, travel, service-quality and delay information, and compares 14 models on an equal
footing using a strict train/validation/test protocol.

**Headline result:** the top seven models are **statistically indistinguishable** — their 95%
bootstrap confidence intervals all overlap. The project reports this honestly rather than
declaring a winner on a difference smaller than the measurement noise.

## Research motivation

This project is inspired by "Predicting Airline Passenger Satisfaction Using Machine Learning: A
Comparative Evaluation" (ICSADL 2026), which reports ~95.83% accuracy and ~95.15% F1 for Random
Forest. This is an **independent implementation**: those figures are treated as external
reference points for discussion only, never as targets to reproduce or optimise toward. The
project additionally asks a question the reference work does not: whether deep learning is
competitive with tree ensembles on this tabular dataset, and whether the differences between
leading models are statistically meaningful at all.

## Research questions

| | Question |
|---|---|
| RQ1 | How effectively can passenger satisfaction be predicted using traditional ML models? |
| RQ2 | Do ensemble methods outperform individual classifiers? |
| RQ3 | Can deep neural networks provide competitive performance on tabular data? |
| RQ4 | Which passenger/service features contribute most strongly to model predictions? |
| RQ5 | What trade-offs exist between performance, interpretability and computational cost? |

## Features

- 14 models spanning a dummy baseline, classical ML, ensembles and three neural networks
- Strict **70/15/15** stratified split; the test set is opened exactly **once**
- All preprocessing inside sklearn `Pipeline`s, fitted on training data only (no leakage)
- Hyperparameter tuning with `RandomizedSearchCV` on the training set only
- Stratified cross-validation plus **bootstrap confidence intervals** to test whether model
  differences are real
- SHAP explainability and permutation importance for the neural network
- An **experiment registry** and a JSON **manifest** recording seed, split sizes, library
  versions, timings and chosen hyperparameters for every run
- A **five-page Streamlit application** with a flight-deck themed dark UI, a scroll-driven
  parallax hero, 18 interactive Plotly charts (hover, zoom, toggle, cross-filter), a live
  pipeline runner, and a one-page summary poster — loads saved artifacts and never retrains
- 31 automated tests, including explicit anti-leakage assertions

---

## Dataset

The Airline Passenger Satisfaction dataset: **129,880 records × 24 columns**.

| Property | Value |
|---|---|
| Records | 129,880 |
| Features used | 22 (after dropping `ID`) |
| Missing values | 393, all in `Arrival Delay` (0.30%) |
| Duplicate rows | 0 |
| Class balance | 56.55% Neutral or Dissatisfied / 43.45% Satisfied |

Place the CSV at `data/airline_passenger_satisfaction.csv`. See [data/README.md](data/README.md).

### A note on the rating scale

Service ratings are documented as 1–5 but contain **0** values. Measured satisfaction rates show
0 behaves as *"not applicable"*, not as *"worst"* — passengers who rated in-flight wifi 0 are
99.7% satisfied, versus ~25% for ratings of 1–3. This project keeps ratings as plain 0–5 numeric
values (consistent with the reference literature) and documents the observation in the report
rather than encoding it. See `outputs/reports/data_quality.md`.

---

## Installation

**Prerequisites:** Python 3.11 (TensorFlow does not support 3.14) and, on macOS, OpenMP for
XGBoost.

```bash
# macOS only -- XGBoost fails to load without this
brew install libomp

# create the environment
python3.11 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Usage

```bash
# quick end-to-end run for debugging (subsampled, tiny budgets)
python train.py --mode fast

# the full experiment -- about 7 minutes on a modern laptop
python train.py --mode full

# useful flags
python train.py --mode full --skip-eda    # skip figure generation
python train.py --mode full --skip-dl     # classical/ensemble models only

# tests
pytest tests/ -v

# the application
streamlit run app/app.py
```

`train.py` writes every artifact the app and the report consume: trained models, `model_results.csv`,
`feature_importance.csv`, `experiment_manifest.json`, SHAP outputs and all figures.

---

## Results

Held-out test set (19,482 passengers), from `outputs/model_results.csv`:

| Model | Family | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|---|
| Neural Net (Deep MLP) | deep | 0.9660 | 0.9781 | 0.9429 | **0.9602** | 0.9959 |
| Neural Net (Basic MLP) | deep | 0.9659 | 0.9770 | 0.9438 | 0.9601 | 0.9957 |
| Random Forest | ensemble | 0.9656 | 0.9778 | 0.9422 | 0.9597 | 0.9951 |
| Voting Ensemble | ensemble | 0.9654 | 0.9758 | 0.9438 | 0.9595 | 0.9950 |
| Stacking Ensemble | ensemble | 0.9651 | 0.9711 | 0.9480 | 0.9594 | 0.9953 |
| Neural Net (Regularized MLP) | deep | 0.9644 | 0.9742 | 0.9431 | 0.9584 | 0.9954 |
| XGBoost | ensemble | 0.9640 | 0.9754 | 0.9408 | 0.9578 | 0.9950 |
| Extra Trees | ensemble | 0.9602 | 0.9695 | 0.9380 | 0.9535 | 0.9936 |
| Support Vector Machine | classical | 0.9537 | 0.9600 | 0.9323 | 0.9459 | 0.9887 |
| Gradient Boosting | ensemble | 0.9505 | 0.9571 | 0.9276 | 0.9421 | 0.9906 |
| Decision Tree | classical | 0.9505 | 0.9639 | 0.9205 | 0.9417 | 0.9827 |
| K-Nearest Neighbors | classical | 0.9291 | 0.9521 | 0.8811 | 0.9153 | 0.9789 |
| Logistic Regression | classical | 0.8794 | 0.8758 | 0.8419 | 0.8585 | 0.9292 |
| Dummy Baseline | baseline | 0.5655 | 0.0000 | 0.0000 | 0.0000 | 0.5000 |

### The models are tied

The top five models span just **0.00077** F1, while the average 95% bootstrap confidence interval
is **0.0059** wide — roughly eight times larger than the spread. Seven models have overlapping
confidence intervals with the leader:

| Model | F1 | 95% CI |
|---|---|---|
| Neural Net (Deep MLP) | 0.9602 | [0.9573, 0.9631] |
| Neural Net (Basic MLP) | 0.9601 | [0.9573, 0.9630] |
| Random Forest | 0.9597 | [0.9568, 0.9626] |
| Voting Ensemble | 0.9595 | [0.9566, 0.9627] |
| Stacking Ensemble | 0.9594 | [0.9564, 0.9624] |
| Neural Net (Regularized MLP) | 0.9584 | [0.9557, 0.9611] |
| XGBoost | 0.9578 | [0.9548, 0.9610] |

**Practical recommendation:** because predictive performance is tied, selection should rest on
cost and interpretability. Random Forest and XGBoost are preferable in practice — they match the
neural networks statistically, support exact SHAP explanations via `TreeExplainer`, and XGBoost
trains in 6.9s versus 35.5s for the Deep MLP.

### Comparison with the reference paper

| | Accuracy | F1 |
|---|---|---|
| Reference paper (ICSADL 2026), Random Forest | 0.9583 | 0.9515 |
| This project, Random Forest | 0.9656 | 0.9597 |

Our results are close to, and slightly above, the published figures. Plausible reasons include a
different split protocol (70/15/15 with a dedicated validation set), the engineered features, and
independent hyperparameter tuning. The comparison is contextual: different splits and
preprocessing make exact reproduction neither expected nor the goal.

### Explainability

SHAP (`TreeExplainer` on Random Forest) and neural-network permutation importance independently
rank the same features highest:

1. Online Boarding
2. In-flight Wifi Service
3. Type of Travel (Business / Personal)
4. Customer Type (First-time / Returning)
5. Class (Business)

Service-quality variables dominate, supporting hypothesis H3. **SHAP describes model behaviour,
not causation** — these are the features the model weighted, not proven causes of satisfaction.

---

## Project structure

```
airline-passenger-satisfaction/
├── config.py                  # seeds, paths, split sizes, mode budgets
├── train.py                   # end-to-end pipeline entry point
├── data/
│   └── airline_passenger_satisfaction.csv
├── src/
│   ├── data_loader.py         # loading, quality report, the single 70/15/15 split
│   ├── preprocessing.py       # leakage-safe pipelines
│   ├── feature_engineering.py # documented derived features
│   ├── registry.py            # experiment registry: every model declared once
│   ├── train_ml.py            # classical + ensemble training
│   ├── train_dl.py            # three Keras MLPs
│   ├── evaluate.py            # metrics, CV, bootstrap CIs
│   ├── explainability.py      # SHAP + permutation importance
│   ├── eda.py                 # exploratory figures
│   ├── plots.py               # evaluation figures
│   └── utils.py               # seeding, logging, experiment manifest
├── app/
│   ├── app.py                 # Streamlit application
│   └── components/ui.py
├── models/                    # saved pipelines, .keras models, feature schema
├── outputs/
│   ├── model_results.csv      # the consolidated comparison table
│   ├── feature_importance.csv
│   ├── experiment_manifest.json
│   ├── figures/               # all generated figures
│   ├── reports/               # data quality report
│   └── shap_results/
├── report/                    # academic report, methodology, results, viva Q&A
├── tests/                     # 31 tests including anti-leakage assertions
└── requirements.txt
```

## Methodology

**Split protocol.** The data is split once, in `src/data_loader.make_splits`:

| Split | Rows | Role |
|---|---|---|
| Train | 90,916 (70%) | Fits all models. The only data CV and hyperparameter search ever see. |
| Validation | 19,482 (15%) | Early stopping, model selection, threshold choice. |
| Test | 19,482 (15%) | Opened exactly once, after the final model is chosen. |

Leakage is prevented structurally, not by convention: preprocessing lives inside pipelines fitted
on training folds, and `tests/test_preprocessing.py` asserts that the scaler's statistics come
from training data alone.

**Reproducibility.** `random_state=42` throughout, with Python, NumPy and TensorFlow seeded.
Consecutive full runs reproduce identical metrics. Every run writes
`outputs/experiment_manifest.json` with the seed, split sizes, library versions, per-model
timings and selected hyperparameters.

## Deployment

The app runs on **Streamlit Community Cloud** (free tier).

> Streamlit **cannot** be hosted on static hosts such as Netlify or GitHub Pages — it needs a
> persistent Python server with WebSocket support. Netlify's Python support covers only
> short-lived serverless functions.

### Steps

```bash
git init                      # if not already a repo
git add -A
git commit -m "Airline passenger satisfaction: ML vs DL comparative evaluation"
git remote add origin https://github.com/<your-username>/<your-repo>.git
git push -u origin main
```

Then at [share.streamlit.io](https://share.streamlit.io): **New app** → pick the repo →
set main file to `app/app.py` → **Deploy**. First boot takes a few minutes while TensorFlow
installs.

### What gets committed, and why

The deployed app must never retrain, so the artifacts it loads are committed — but the large
ones are not, because they exceed platform limits:

| Artifact | Size | Committed | Reason |
|---|---|---|---|
| `best_model.pkl` + `feature_schema.json` | < 5 KB | yes | final model pointer and UI schema |
| `neural_net_deep_mlp.keras` + preprocessor | 1.4 MB | yes | the selected final model |
| `random_forest.pkl`, `xgboost.pkl`, `logistic_regression.pkl` | 14 MB | yes | offered in the model selector |
| `extra_trees.pkl` | 105 MB | **no** | exceeds GitHub's 100 MB per-file limit |
| `voting_ensemble.pkl` | 99 MB | **no** | near the limit; statistically tied with Random Forest |
| `stacking_ensemble.pkl` | 28 MB | **no** | statistically tied with Random Forest |
| `airline_passenger_satisfaction.csv` | 12 MB | **no** | an 8,000-row stratified sample ships instead |

**Total deployed repository: ~20 MB**, comfortably inside Streamlit Cloud's ~1 GB memory budget.

The app degrades gracefully: the model selector lists only models present on disk, and the Data
Explorer falls back to `data/sample_for_app.csv` (8,000 rows, identical class balance of 0.4345,
all category levels preserved) with a banner stating which data is in use. Regenerate every
excluded artifact locally with `python train.py --mode full`.

This was verified by running the app against a simulated fresh clone containing only the
committed files — all five pages render with zero errors.

## Screenshots

_Placeholders — add after running the app locally._

| Page | Screenshot |
|---|---|
| Home (with summary poster) | `docs/screenshot_home.png` |
| Passenger Prediction | `docs/screenshot_prediction.png` |
| Analytics Dashboard | `docs/screenshot_analytics.png` |
| Model Performance | `docs/screenshot_models.png` |
| Run the Pipeline | `docs/screenshot_pipeline.png` |

### Application pages

1. **Flight Deck** — scroll-driven parallax hero (starfield, altitude rings, self-drawing flight
   path) over a fully interactive readout: filterable leaderboard, confidence-interval forest
   plot, and a performance-vs-cost scatter. No static images.
2. **Simulate Passenger** — four one-click flight profiles, bulk rating controls, and a *live*
   prediction that updates as you adjust inputs; probability gauge, per-prediction SHAP
   breakdown, and an optional all-model agreement comparison.
3. **Data Explorer** — cross-filter the cohort by class, travel type and age, then explore
   distribution, service-rating, delay and correlation tabs that all respond to the filter.
4. **Model Bay** — pick any two models for a head-to-head radar chart with an automatic verdict
   on whether their confidence intervals overlap; plus ROC curves, confusion matrices with
   error-cost commentary, and adjustable feature-importance charts.
5. **Run Pipeline** — launch `train.py` from the browser and watch progress stream live, with a
   progress bar and rolling log; caches clear automatically when it finishes.

### Design system

A flight-deck / avionics theme (`app/components/theme.py`): OLED dark base, cyan-amber accent
pairing, Chakra Petch display + Fira Code instrument labels, glass panels with corner brackets.

Verified against the design checklist:

| Check | Result |
|---|---|
| Body text contrast | 17.07:1 (WCAG AAA) |
| Muted text contrast | 7.85:1 (AAA) |
| Accent (cyan / amber) | 13.60:1 / 8.08:1 (AAA) |
| Touch targets | all app buttons ≥ 44px |
| Keyboard focus | visible 3px focus rings |
| `prefers-reduced-motion` | respected — parallax and reveals disabled |
| Horizontal scroll | none at 375 / 768 / 1440px |
| Icons | inline SVG, no emoji |

## Limitations

- The dataset is historical and its provenance/labelling methodology is not fully documented.
- Findings may not generalise to other airlines, regions or time periods.
- The tuning budget is deliberately lean (12 candidates × 3 folds); larger searches might shift
  the ranking slightly, though the tie among leading models is unlikely to change.
- SHAP and permutation importance are associational, not causal.
- Ratings of 0 are modelled as numeric values despite evidence they mean "not applicable".

## Future work

Real-time feedback prediction · NLP sentiment analysis of passenger reviews · transformer-based
review models · multimodal models · cloud deployment with model monitoring · online learning ·
larger and more diverse datasets · treating rating 0 as an explicit "not applicable" indicator.

## License and attribution

Academic project for educational purposes. The dataset is a public Kaggle dataset.
