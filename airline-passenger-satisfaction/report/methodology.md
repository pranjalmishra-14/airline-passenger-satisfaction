# Methodology

A standalone description of the experimental design. All claims here are implemented in code and
verified by the test suite.

## 1. Design principles

1. **One split, one place.** The dataset is partitioned in exactly one function
   (`src/data_loader.make_splits`). No other module may re-split, which removes the most common
   source of accidental leakage in student projects.
2. **Preprocessing inside pipelines.** Every transformer is a stage of an sklearn `Pipeline`, so
   it is fitted on the training fold only. This is a structural guarantee, not a convention.
3. **The test set is opened once.** It is untouched during training, cross-validation,
   hyperparameter search, early stopping and model selection.
4. **Everything is declared once.** Models live in an experiment registry
   (`src/registry.py`), so all of them are handled identically by the training and evaluation code.
5. **Failures are reported, never hidden.** A model that raises is caught, its traceback stored in
   the manifest, and it appears in the results table marked `FAILED`.

## 2. Split protocol

Stratified 70/15/15, `random_state=42`:

| Split | Rows | Share | Role |
|---|---|---|---|
| Train | 90,916 | 70% | Fits all models. The only data CV and hyperparameter search see. |
| Validation | 19,482 | 15% | Early stopping, model selection, threshold choice. |
| Test | 19,482 | 15% | Opened once, after selection is complete. |

The split is produced in two stages: the test set is carved off first, then the remainder is
divided into train and validation. Stratification holds the positive-class rate at 0.4345 in all
three partitions.

### Verification

`tests/test_data.py` asserts that the three index sets are pairwise disjoint and that their union
equals the sum of their sizes. `tests/test_preprocessing.py` asserts that a scaler fitted on the
training data centres the training data but *not* the validation data — the signature of
leakage-free fitting.

## 3. Preprocessing

| Step | Treatment | Fitted on |
|---|---|---|
| Identifier removal | `ID` dropped (auto-detected as unique per row) | — |
| Numeric imputation | Median (393 missing `Arrival Delay` values) | Train only |
| Categorical imputation | Most frequent | Train only |
| Categorical encoding | One-hot, `handle_unknown="ignore"` | Train only |
| Scaling | Standardisation — **only** for LR, SVM, k-NN, neural networks | Train only |

Tree-based models are invariant to monotone scaling and are deliberately left unscaled.

## 4. Feature engineering

Three row-wise derived features (22 raw → 32 encoded model inputs). Being row-wise, they cannot
transfer information between rows and therefore cannot leak across splits — a property asserted
by `test_engineered_features_are_row_wise`.

| Feature | Definition | Justification |
|---|---|---|
| `Total Delay` | Departure + Arrival delay | Cumulative disruption is what a passenger experiences. |
| `Service Quality Score` | Mean of the 14 service ratings | Single summary of perceived service. |
| `Flight Distance Category` | Short / Medium / Long (800, 2200 min cut points) | Expectations vary by haul length; fixed domain thresholds, not learned. |

No feature is derived from the target.

## 5. Models

| Family | Models | Scaled |
|---|---|---|
| Baseline | DummyClassifier (most frequent) | no |
| Classical | Logistic Regression, Decision Tree, SVM (RBF), k-NN | LR/SVM/k-NN yes |
| Ensemble | Random Forest, Extra Trees, XGBoost, Gradient Boosting, Voting, Stacking | no |
| Deep | Basic MLP, Regularized MLP, Deep MLP | yes |

Deep learning is restricted to feed-forward architectures. CNNs, LSTMs and transformers were
excluded by design: the data has no spatial or temporal ordering for them to exploit.

## 6. Hyperparameter optimisation

`RandomizedSearchCV`, F1-scored, **12 candidates × 3 folds**, training partition only, applied to
Random Forest and XGBoost. The budget is deliberately lean and is reported explicitly so that the
strength of the tuning evidence is not overstated.

## 7. Deep learning training

| Setting | Value |
|---|---|
| Loss | Binary cross-entropy |
| Optimiser | Adam, learning rate 1e-3 |
| Batch size | 256 |
| Max epochs | 60 |
| EarlyStopping | patience 8 on `val_loss`, restores best weights |
| ModelCheckpoint | best `val_loss` saved to `.keras` |

Validation data for early stopping is the dedicated validation partition — never the test set.
Observed stopping points were 49, 49 and 44 epochs, all below the budget.

## 8. Evaluation

Accuracy, precision, recall, F1 and ROC-AUC, with confusion matrices, classification reports,
ROC and precision-recall curves. Positive class = *Satisfied*.

For models without `predict_proba` (the SVM, where probability calibration is deprecated in
scikit-learn 1.9 and costly), `decision_function` supplies the continuous scores used for
ROC-AUC.

## 9. Statistical comparison

Two complementary methods:

1. **Stratified 5-fold cross-validation** on the training partition — measures stability
   (mean ± std) without touching the test set.
2. **Percentile bootstrap confidence intervals** (1,000 resamples) on test-set F1 — measures
   whether the gap between models exceeds sampling noise.

Overlapping intervals are interpreted conservatively: they indicate that a difference has **not
been demonstrated**, which is not the same as proving equivalence. No formal significance test is
claimed.

## 10. Reproducibility

- `random_state=42` throughout; Python, NumPy and TensorFlow all seeded via `utils.set_seeds()`.
- Consecutive full runs reproduce identical metrics to four decimal places.
- `outputs/experiment_manifest.json` records run id, timestamp, mode, seed, split sizes, library
  versions, per-model timings, chosen hyperparameters and any failures.
- Two modes: `--mode fast` (subsampled, tiny budgets) for debugging and `--mode full` for the
  reported experiment.
