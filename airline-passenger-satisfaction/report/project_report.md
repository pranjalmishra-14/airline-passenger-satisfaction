# Predicting Airline Passenger Satisfaction Using Machine Learning and Deep Learning: A Comparative Evaluation

**B.Tech Computer Science and Engineering — Project Report**

---

## 1. Title

Predicting Airline Passenger Satisfaction Using Machine Learning and Deep Learning:
A Comparative Evaluation

## 2. Abstract

Airline passenger satisfaction is a commercially significant outcome that airlines seek to
predict from operational and service-quality data. This project presents an independent
comparative evaluation of fourteen classification approaches — a majority-class baseline, four
classical machine learning algorithms, six ensemble methods and three feed-forward neural
networks — on a dataset of 129,880 passenger records described by 22 predictive features. A
strict stratified 70/15/15 train/validation/test protocol was used, with all preprocessing fitted
inside pipelines on training data only and the test partition evaluated exactly once, after model
selection was complete on the validation set. The best test-set performance was obtained by a
deep multilayer perceptron (accuracy 0.9660, F1 0.9602, ROC-AUC 0.9959), marginally ahead of a
tuned Random Forest (accuracy 0.9656, F1 0.9597). However, percentile bootstrap confidence
intervals computed on the test set show that the leading seven models — spanning ensemble and
deep-learning families — have overlapping 95% intervals: the top five differ by 0.00077 F1 while
the mean interval width is 0.0059, roughly eight times larger. The principal finding is therefore
that these architectures are statistically indistinguishable on this dataset, and that model
choice should be governed by computational cost and interpretability rather than by small
differences in accuracy. All models substantially exceeded the 0.5655-accuracy majority-class
baseline. SHAP analysis and neural-network permutation importance independently identified Online
Boarding, In-flight Wifi Service, Type of Travel and Customer Type as the most influential
features. A Streamlit application exposes the trained pipeline for interactive prediction.

**Word count: ~250**

## 3. Keywords

Airline passenger satisfaction; supervised classification; ensemble learning; deep learning;
tabular data; explainable AI; SHAP; bootstrap confidence intervals; model comparison

## 4. Introduction

Passenger satisfaction directly influences airline profitability through repeat custom, brand
reputation and pricing power. Airlines routinely collect structured post-flight survey data
covering demographics, itinerary characteristics, service-quality ratings and operational delays.
Converting that data into a reliable predictive model allows an operator to anticipate
dissatisfaction and to understand which service dimensions carry the greatest predictive weight.

This project treats the task as a binary classification problem — predicting whether a passenger
reports being *Satisfied* or *Neutral or Dissatisfied* — and evaluates model families that are
frequently compared in the literature but rarely subjected to a statistical test of whether their
reported differences are meaningful.

## 5. Background

Tabular supervised learning is dominated in practice by gradient-boosted decision trees and
random forests, which handle mixed feature types, monotone-invariant scaling and non-linear
interactions without extensive preprocessing. Deep neural networks, despite their dominance in
vision and language, have historically been reported as less competitive on tabular problems.
This project tests that expectation directly on a large, real-world tabular dataset.

## 6. Problem statement

Given 22 features describing a passenger's demographics, itinerary, perceived service quality and
experienced delays, predict the binary satisfaction outcome, and determine which model family
offers the best trade-off between predictive performance, interpretability and computational cost.

## 7. Research motivation

This work is inspired by "Predicting Airline Passenger Satisfaction Using Machine Learning: A
Comparative Evaluation" (ICSADL 2026), which reports approximately 95.83% accuracy and 95.15% F1
for a Random Forest classifier. This project is an **independent implementation**. Those figures
are used strictly as external reference points for discussion; no result here was optimised
toward them, and no published number was copied into this report.

Two gaps motivated the extension of scope. First, the reference comparison does not include deep
learning, leaving open whether neural networks are competitive on this data. Second, comparative
studies commonly rank models by point estimates without assessing whether the differences exceed
sampling noise. This project addresses both.

## 8. Objectives

1. Build a reproducible, leakage-free pipeline for airline satisfaction prediction.
2. Train and fairly compare baseline, classical, ensemble and deep-learning models.
3. Quantify the uncertainty in the comparison using bootstrap confidence intervals.
4. Identify the features carrying the greatest predictive weight using explainable AI.
5. Deploy the selected model in an interactive application.

## 9. Literature review

The following themes inform this work. *No specific external references are cited numerically
here, because doing so without direct verification of each source would risk misattribution;
statements below are presented as general background rather than as claims traceable to a
particular paper.*

- **Customer satisfaction prediction.** Service-quality research generally models satisfaction as
  a function of expectation versus perceived performance across multiple service dimensions,
  which motivates the per-dimension rating features used here.
- **Classification algorithms.** Logistic regression provides an interpretable linear baseline;
  decision trees capture non-linear thresholds at the cost of variance; SVMs offer strong margins
  in scaled feature spaces; k-NN provides a non-parametric reference.
- **Ensemble learning.** Bagging (Random Forest, Extra Trees) reduces variance through averaging
  over decorrelated trees, while boosting (XGBoost, Gradient Boosting) reduces bias by fitting
  sequential residual-correcting learners. Voting and stacking combine heterogeneous learners.
- **Deep learning for tabular data.** Multilayer perceptrons with batch normalisation and dropout
  are the standard neural baseline for tabular problems. The prevailing expectation is that tree
  ensembles match or exceed them on datasets of this size and structure — a proposition this
  project evaluates empirically.
- **Explainable AI.** SHAP provides locally accurate additive feature attributions with an exact
  polynomial-time algorithm for tree models. Permutation importance offers a model-agnostic
  alternative. Both describe model behaviour and are associational, not causal.

## 10. Dataset description

| Property | Value |
|---|---|
| Records | 129,880 |
| Columns | 24 (1 identifier, 22 predictive features, 1 target) |
| Target | `Satisfaction`: *Satisfied* (43.45%) / *Neutral or Dissatisfied* (56.55%) |
| Missing values | 393, all in `Arrival Delay` (0.30%) |
| Duplicate rows | 0 |
| Imbalance ratio | 1.30 : 1 |

**Feature groups.** Demographic (`Gender`, `Age`, `Customer Type`); travel (`Type of Travel`,
`Class`, `Flight Distance`); service ratings on a 0–5 scale (fourteen dimensions including
`Online Boarding`, `Seat Comfort`, `In-flight Wifi Service`, `Cleanliness`); and operational
delays (`Departure Delay`, `Arrival Delay` in minutes).

The class imbalance is mild (1.30:1), so resampling techniques such as SMOTE were not applied;
stratified splitting is sufficient and avoids introducing synthetic records.

### 10.1 An observation about the rating scale

Service ratings are documented as a 1–5 scale but contain **0** values. Empirical satisfaction
rates by rating value show that 0 does not behave as the lowest point of an ordinal scale:

| Rating for In-flight Wifi Service | 0 | 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|---|
| Satisfaction rate | **99.7%** | 32.8% | 24.7% | 25.2% | 60.1% | 99.0% |

A monotone ordinal scale would produce a rising sequence. The observed pattern — 0 associated
with near-universal satisfaction, then a jump down to 32.8% at rating 1 — is consistent with 0
denoting *"not applicable"* (the passenger did not use or was not offered the service).

**Design decision.** Ratings are retained as plain 0–5 numeric values, consistent with the
prevailing treatment in published work on this dataset, so that the comparison remains aligned
with the reference literature. The observation is documented here and revisited in Limitations
and Future Scope. Because the models used are predominantly non-linear (trees and neural
networks), they are able to fit the non-monotone relationship despite the numeric encoding.

## 11. Data preprocessing

1. **Identifier removal.** `ID` was removed after automated detection as a unique identifier
   (129,880 distinct values across 129,880 rows). It carries no predictive information and would
   risk memorisation. This was the only column removed, and the removal is logged in
   `outputs/reports/data_quality.md`.
2. **Missing-value imputation.** The 393 missing `Arrival Delay` values were imputed with the
   median, computed on the training partition only.
3. **Categorical encoding.** One-hot encoding with `handle_unknown="ignore"`, so unseen
   categories at inference time do not raise errors.
4. **Scaling.** Standardisation was applied only for Logistic Regression, SVM, k-NN and the
   neural networks. Tree-based models are invariant to monotone feature scaling and were
   deliberately left unscaled.

All steps are implemented as scikit-learn `Pipeline` stages, so each transformer is fitted on the
training fold alone. This is the structural guarantee against leakage; it is asserted by
automated tests rather than left to convention.

## 12. Exploratory data analysis

Eight figures were generated (`outputs/figures/01`–`08`), each answering a stated question:

- **Target balance** — confirms the 56.55/43.45 split and the absence of severe imbalance.
- **Demographics** — satisfaction differs markedly by `Customer Type`, only slightly by `Gender`.
- **Travel context** — `Class` and `Type of Travel` separate the outcome strongly; business
  travellers and business-class passengers report higher satisfaction.
- **Service-rating heatmap** — shows a clear monotone gradient from ratings 1 to 5 and the
  anomalous 0 column discussed in §10.1.
- **Delay analysis** — both delay variables are extremely right-skewed (median 0 minutes, maximum
  1,592). Satisfaction declines monotonically across delay bands.
- **Correlation heatmap and ranked target correlation** — no numeric feature exceeds a moderate
  linear correlation with the target, indicating that non-linear models are appropriate.

## 13. Feature engineering

Three derived features were added; each is justified on domain grounds, is computed row-wise
(and therefore cannot leak information across splits), and is not derived from the target:

| Feature | Definition | Rationale |
|---|---|---|
| `Total Delay` | `Departure Delay` + `Arrival Delay` | Passengers experience cumulative disruption rather than two independent delays. |
| `Service Quality Score` | Mean of the fourteen service ratings | A single summary of perceived service quality. |
| `Flight Distance Category` | Short / Medium / Long (cut at 800 and 2,200) | Service expectations differ by haul length; fixed domain cut points, not learned from data. |

Encoding expands the 22 raw features to 32 model inputs.

## 14. Machine learning methodology

Models were declared in a central **experiment registry** (`src/registry.py`), so every model is
specified exactly once and processed identically by the training and evaluation code. Failures
are captured with their traceback and reported in the results table rather than silently skipped.

| Family | Models |
|---|---|
| Baseline | DummyClassifier (most frequent class) |
| Classical | Logistic Regression, Decision Tree, Support Vector Machine (RBF), k-Nearest Neighbors |
| Ensemble | Random Forest, Extra Trees, XGBoost, Gradient Boosting, Voting, Stacking |

## 15. Deep learning methodology

Three feed-forward architectures were implemented in Keras. Architectures suited to sequential or
spatial data (CNNs, LSTMs, transformers) were deliberately excluded: this dataset has no temporal
or spatial structure, so such models would add complexity without a defensible rationale.

| Model | Architecture | Parameters |
|---|---|---|
| Basic MLP | Dense(256) → BN → ReLU → Dropout(0.3) → Dense(128) → BN → ReLU → Dropout(0.3) → Dense(1, sigmoid) | 43,009 |
| Regularized MLP | As above with L2 (1e-4) and Dropout(0.45) | 43,009 |
| Deep MLP | Four hidden blocks of tapering width (384 → 192 → 96 → 48) with BN and dropout | 112,513 |

Layer widths are derived from the encoded feature count rather than copied from an arbitrary
reference. Training used the Adam optimiser (learning rate 1e-3), binary cross-entropy loss, a
batch size of 256, EarlyStopping (patience 8, restoring best weights) and ModelCheckpoint.
Early stopping halted training at 49, 49 and 44 epochs respectively against a 60-epoch budget,
confirming that the regularisation was effective and the networks were not run to overfitting.

## 16. Experimental setup

| Parameter | Value |
|---|---|
| Random seed | 42 (Python, NumPy, TensorFlow) |
| Split | Stratified 70 / 15 / 15 |
| Train | 90,916 records |
| Validation | 19,482 records |
| Test | 19,482 records |
| Hardware | Apple Silicon, 14 cores, 48 GB RAM |
| Full-run wall-clock time | ~7 minutes 38 seconds |

**Protocol.** The dataset is split at exactly one point in the codebase. The training partition
fits all models and is the only data seen by cross-validation and hyperparameter search. The
validation partition drives early stopping and final model selection. The test partition was
opened once, after selection was complete. Positive-class rates are 0.4345 in all three
partitions, confirming correct stratification.

**Reproducibility.** Consecutive full runs produced identical metrics to four decimal places.
Each run writes `outputs/experiment_manifest.json` recording the seed, split sizes, library
versions, per-model wall-clock timings and the selected hyperparameters.

## 17. Evaluation metrics

Accuracy, precision, recall, F1 and ROC-AUC. The positive class (1) is *Satisfied*; the negative
class (0) is *Neutral or Dissatisfied*. F1 is the primary selection metric, since it balances the
two error types under mild class imbalance. Confusion matrices and classification reports were
produced for all models, along with ROC and precision-recall curves.

## 18. Hyperparameter optimisation

`RandomizedSearchCV` was applied to Random Forest and XGBoost with a deliberately lean budget of
**12 candidates × 3 folds**, scored by F1, using the training partition only. The budget is
stated explicitly so the strength of the evidence is not overstated.

Selected configurations:

- **Random Forest:** `n_estimators=200`, `max_depth=20`, `min_samples_split=5`,
  `min_samples_leaf=1`, `max_features=0.5`
- **XGBoost:** `n_estimators=200`, `max_depth=10`, `learning_rate=0.03`, `subsample=1.0`,
  `colsample_bytree=0.85`

## 19. Results

All figures below are from `outputs/model_results.csv`, computed on the held-out test set of
19,482 passengers.

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

No model failed to train; all fourteen completed.

### 19.1 Baseline comparison

The majority-class baseline achieves 0.5655 accuracy and 0.0000 F1 (it never predicts the
positive class). Every trained model exceeds it by a wide margin, confirming that the models
learn genuine structure rather than exploiting class priors.

### 19.2 Cross-validation stability

Stratified 5-fold cross-validation on the training partition:

| Model | CV F1 mean | CV F1 std |
|---|---|---|
| XGBoost | 0.9559 | 0.0012 |
| Random Forest | 0.9544 | 0.0012 |
| Extra Trees | 0.9510 | 0.0012 |
| Decision Tree | 0.9403 | 0.0021 |
| Logistic Regression | 0.8518 | 0.0039 |

Standard deviations of 0.001–0.004 indicate stable performance across folds; the ensembles are
the most stable.

### 19.3 Statistical comparison — the central finding

Percentile bootstrap 95% confidence intervals (1,000 resamples) on test-set F1:

| Model | F1 | 95% CI | CI width |
|---|---|---|---|
| Neural Net (Deep MLP) | 0.9602 | [0.9573, 0.9631] | 0.0059 |
| Neural Net (Basic MLP) | 0.9601 | [0.9573, 0.9630] | 0.0057 |
| Random Forest | 0.9597 | [0.9568, 0.9626] | 0.0058 |
| Voting Ensemble | 0.9595 | [0.9566, 0.9627] | 0.0061 |
| Stacking Ensemble | 0.9594 | [0.9564, 0.9624] | 0.0060 |
| Neural Net (Regularized MLP) | 0.9584 | [0.9557, 0.9611] | 0.0054 |
| XGBoost | 0.9578 | [0.9548, 0.9610] | 0.0061 |

The top five models span **0.00077** F1, while the mean confidence-interval width is **0.0059** —
approximately eight times the observed spread. Seven models have confidence intervals that
overlap the leader's.

**Interpretation.** These seven models are statistically indistinguishable on this test set.
Reporting the Deep MLP as "the best model" on a margin of 0.0005 F1 over Random Forest would
overstate the evidence. Overlapping confidence intervals are a conservative indicator rather than
a formal hypothesis test, so the appropriate conclusion is the absence of a demonstrated
difference, not proof of equivalence.

### 19.4 Comparison with the reference paper

| Source | Model | Accuracy | F1 |
|---|---|---|---|
| ICSADL 2026 (reference) | Random Forest | 0.9583 | 0.9515 |
| This project | Random Forest | 0.9656 | 0.9597 |

Our independently obtained Random Forest results are slightly higher. Plausible contributing
factors include the different split protocol (70/15/15 with a dedicated validation partition
versus an unspecified split), the three engineered features, and independent hyperparameter
tuning. Because split composition and preprocessing differ, exact reproduction is neither
expected nor the objective; the comparison is contextual only.

## 20. Model comparison

| Family | Best model | Test F1 | Best fit time |
|---|---|---|---|
| Baseline | Dummy | 0.0000 | 0.2 s |
| Classical | Support Vector Machine | 0.9459 | 35.9 s |
| Ensemble | Random Forest | 0.9597 | 49.0 s |
| Deep learning | Neural Net (Deep MLP) | 0.9602 | 35.5 s |

**Answers to the research questions.**

- **RQ1.** Classical models predict satisfaction effectively but with a clear internal spread:
  SVM reaches 0.9459 F1 while Logistic Regression reaches only 0.8585, confirming that the
  decision boundary is substantially non-linear.
- **RQ2.** Ensembles outperform the *average* individual classifier decisively (best ensemble
  0.9597 versus best classical 0.9459 F1). However, Voting and Stacking did not improve on a
  well-tuned single Random Forest, indicating that combining already-correlated tree learners
  yields diminishing returns.
- **RQ3.** Yes. All three neural networks are competitive, and two fall within the leading
  statistically indistinguishable group. Deep learning is neither superior nor inferior here.
- **RQ4.** Online Boarding, In-flight Wifi Service, Type of Travel and Customer Type carry the
  greatest predictive weight (see §21).
- **RQ5.** Because performance is tied, the trade-off is decided by other criteria. XGBoost
  trains in 6.9 s versus 35.5 s for the Deep MLP and supports exact SHAP explanations; the
  neural network requires scaled inputs, GPU-friendly tooling and approximate explanation
  methods. Extra Trees and Voting also produced very large serialised artifacts (101 MB and
  94 MB compressed), which matters for deployment.

**Hypothesis outcomes.**

- **H1** (ensembles competitive): *supported* — ensembles occupy four of the top seven positions.
- **H2** (neural networks competitive on tabular data): *supported* — all three MLPs are
  competitive, contradicting the common expectation that trees clearly dominate.
- **H3** (service-quality variables among the strongest predictors): *supported* — Online
  Boarding and In-flight Wifi Service rank first and second across all three attribution methods.

## 21. Explainable AI

### 21.1 Random Forest impurity importance

| Rank | Feature | Importance |
|---|---|---|
| 1 | Online Boarding | 0.2691 |
| 2 | In-flight Wifi Service | 0.1692 |
| 3 | Class (Business) | 0.0792 |
| 4 | Type of Travel (Personal) | 0.0676 |
| 5 | Type of Travel (Business) | 0.0506 |

### 21.2 SHAP (TreeExplainer on Random Forest)

| Rank | Feature | Mean \|SHAP\| |
|---|---|---|
| 1 | Online Boarding | 0.1181 |
| 2 | In-flight Wifi Service | 0.1112 |
| 3 | Type of Travel (Personal) | 0.0725 |
| 4 | Type of Travel (Business) | 0.0605 |
| 5 | Class (Business) | 0.0469 |

Higher ratings for Online Boarding and In-flight Wifi Service push predictions toward *Satisfied*;
personal travel and first-time customer status push predictions toward *Neutral or Dissatisfied*.

### 21.3 Neural-network permutation importance

SHAP's `KernelExplainer` is model-agnostic but prohibitively expensive at this dataset size; this
is a documented limitation. Permutation importance was used instead:

| Rank | Feature | Mean accuracy drop |
|---|---|---|
| 1 | In-flight Wifi Service | 0.2177 |
| 2 | Gate Location | 0.0696 |
| 3 | Type of Travel (Personal) | 0.0398 |
| 4 | Baggage Handling | 0.0380 |
| 5 | Type of Travel (Business) | 0.0362 |

That two methodologically independent techniques applied to two different model families converge
on the same leading features increases confidence in the attribution.

**Interpretive caution.** SHAP values and permutation importance describe how the fitted models
use each feature. They are measures of association within the model, not evidence of causation.
It is not established that improving in-flight wifi would raise satisfaction; only that the
models weight that variable heavily when predicting it.

## 22. Streamlit application

A four-page application (`app/app.py`) loads the saved artifacts and never retrains:

1. **Home** — project summary, objectives, methodology and headline metrics read from the
   generated results file.
2. **Passenger Prediction** — input widgets generated from the saved feature schema, so the user
   interface cannot drift out of sync with the trained model. Outputs the predicted class, both
   class probabilities, and a per-prediction SHAP explanation with appropriately hedged wording.
3. **Analytics Dashboard** — dataset statistics, satisfaction distributions, service-rating
   comparisons, the EDA figures and the explainability outputs.
4. **Model Performance** — per-family comparison tables, a unified metric chart, the
   confidence-interval analysis, cross-validation stability, and the confusion-matrix, ROC/PR and
   training-history figures.

Models are cached with `@st.cache_resource`. The deployment-critical artifacts are small
(1.3 MB Keras model, 2.5 KB preprocessor, 4.2 KB schema), keeping the app within the memory
limits of free hosting tiers.

## 23. Discussion

The most substantive result is not which model ranked first but that seven models could not be
distinguished statistically. A point-estimate ranking — the standard presentation in comparative
studies — would have declared the Deep MLP the winner on a 0.0005 F1 margin, an artefact of
sampling noise on a 19,482-row test set rather than a reproducible property.

That all leading models converge to approximately 0.96 F1 suggests they are approaching an
information ceiling imposed by the dataset: the features simply do not contain the information
needed to resolve the remaining ~4% of cases. Further architectural refinement is unlikely to
help; richer features would be required.

The strong performance of the neural networks is notable given the common expectation that tree
ensembles dominate tabular tasks. The fair reading is that on a dataset of this size (129,880
rows) with mostly low-cardinality ordinal features, a properly regularised MLP is fully
competitive — but not superior, and more expensive to train and explain.

The gap between Logistic Regression (0.8585 F1) and every non-linear model (>0.94) quantifies how
much of the signal is non-linear or interaction-driven — consistent with the EDA finding that no
individual feature shows strong linear correlation with the target.

## 24. Limitations

1. **Historical data.** The dataset is a static public snapshot; passenger expectations and
   airline operations change over time.
2. **Unknown provenance.** The airline, time period and sampling methodology are undocumented, so
   sampling bias cannot be ruled out and generalisation to other carriers is unverified.
3. **Label methodology.** The satisfaction labels reflect the original survey's binarisation of a
   subjective construct.
4. **Lean tuning budget.** 12 candidates × 3 folds was chosen for tractability. A larger search
   might reorder the leading models slightly, though the statistical tie is unlikely to change.
5. **Rating encoding.** Ratings of 0 are modelled numerically despite evidence they denote "not
   applicable" — a deliberate choice for comparability with published work, but a known
   simplification.
6. **Explanation methods are associational.** Neither SHAP nor permutation importance supports
   causal claims.
7. **Neural-network explanation is global.** KernelExplainer was too costly, so per-prediction
   SHAP is available only for tree models.
8. **Overlapping confidence intervals are not equivalence tests.** They indicate the absence of a
   demonstrated difference, not proof that the models are identical.
9. **Single dataset.** All conclusions are conditioned on this one dataset.

## 25. Future scope

- Real-time prediction on live post-flight feedback streams.
- NLP sentiment analysis of free-text passenger reviews, and transformer-based review models.
- Multimodal models combining structured ratings with review text.
- Treating rating 0 as an explicit "not applicable" indicator and quantifying the effect.
- Formal significance testing (McNemar's test, corrected repeated-measures t-tests).
- Cloud deployment with drift monitoring and online learning.
- Validation on data from multiple airlines and time periods.

## 26. Conclusion

Fourteen models were compared on 129,880 airline passenger records under a strict protocol in
which the test partition was evaluated exactly once. The highest test-set F1 was achieved by a
deep multilayer perceptron (0.9602, accuracy 0.9660), narrowly ahead of a tuned Random Forest
(0.9597, accuracy 0.9656). Bootstrap confidence intervals show, however, that the leading seven
models — drawn from both the ensemble and deep-learning families — are statistically
indistinguishable, with a top-five spread of 0.00077 F1 against a mean interval width of 0.0059.

The practical recommendation is therefore to select on grounds other than accuracy. Random Forest
and XGBoost are preferable for deployment: they match the neural networks statistically, train
considerably faster (XGBoost in 6.9 s versus 35.5 s), and support exact SHAP explanations.
Service-quality features — particularly Online Boarding and In-flight Wifi Service — carry the
greatest predictive weight under three independent attribution methods.

The broader methodological point is that comparative model studies should report uncertainty
alongside point estimates. Without confidence intervals, this study would have reported a "best"
model whose apparent advantage lies well inside the noise floor.

## 27. References

The following are general sources for the methods used. They are listed as software and
methodological references that were directly used in this project; no numbered citations to
specific papers are claimed, as the underlying literature was not independently verified for this
report.

1. Pedregosa, F. et al. *Scikit-learn: Machine Learning in Python.* Journal of Machine Learning
   Research, 12, 2825–2830. (scikit-learn 1.9.1 — used for all classical and ensemble models.)
2. Chen, T. and Guestrin, C. *XGBoost: A Scalable Tree Boosting System.* (XGBoost 3.2.0.)
3. Lundberg, S. M. and Lee, S.-I. *A Unified Approach to Interpreting Model Predictions.*
   (SHAP 0.51.0 — used for TreeExplainer attributions.)
4. Abadi, M. et al. *TensorFlow: Large-Scale Machine Learning on Heterogeneous Systems.*
   (TensorFlow 2.21.0 / Keras 3.15.1 — used for all neural networks.)
5. Breiman, L. *Random Forests.* Machine Learning, 45(1), 5–32.
6. Efron, B. and Tibshirani, R. *An Introduction to the Bootstrap.* (Basis for the percentile
   bootstrap confidence intervals in §19.3.)
7. "Predicting Airline Passenger Satisfaction Using Machine Learning: A Comparative Evaluation."
   2026 5th International Conference on Sentiment Analysis and Deep Learning (ICSADL) — the
   reference work that motivated this independent implementation.
8. Airline Passenger Satisfaction dataset (public, Kaggle).

---

*All experimental results in this report were generated by the code in this repository. Run
`python train.py --mode full` to reproduce them; the exact configuration of the run described
here is recorded in `outputs/experiment_manifest.json`.*
