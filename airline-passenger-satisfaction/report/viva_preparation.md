# Viva Preparation — Questions and Answers

Answers reflect what this project actually did. Numbers are from
`outputs/model_results.csv` and `outputs/experiment_manifest.json`.

---

## A. Project fundamentals

**Q1. What problem does your project solve?**
Binary classification of airline passenger satisfaction — predicting whether a passenger is
*Satisfied* or *Neutral or Dissatisfied* from 22 features covering demographics, itinerary,
14 service-quality ratings and delays. The dataset has 129,880 records.

**Q2. Why is this useful?**
Airlines can identify dissatisfaction patterns and see which service dimensions carry the most
predictive weight, guiding where to invest. The model flags at-risk passengers for recovery action.

**Q3. What makes this a research project rather than a tutorial exercise?**
Three things. It compares 14 models under a controlled protocol; it quantifies *uncertainty* in
that comparison using bootstrap confidence intervals, which most comparative studies omit; and it
reaches a conclusion that contradicts the naive reading of the leaderboard — the top seven models
are statistically indistinguishable.

**Q4. What is your single most important finding?**
That there is no demonstrable winner. The top five models differ by 0.00077 F1 while the mean 95%
confidence interval is 0.0059 wide — about eight times larger. Seven models have overlapping
intervals. Ranking by point estimate alone would have produced a misleading conclusion.

---

## B. Data

**Q5. Describe your dataset.**
129,880 records, 24 columns. One identifier (`ID`, dropped), 22 predictive features, one target.
Class balance 56.55% *Neutral or Dissatisfied* to 43.45% *Satisfied*. 393 missing values, all in
`Arrival Delay` (0.30%). No duplicate rows.

**Q6. How did you handle missing values?**
Median imputation, fitted on the training partition only, inside the pipeline. Median rather than
mean because delay distributions are extremely right-skewed (median 0, maximum 1,592 minutes), so
the mean would be distorted by outliers.

**Q7. Your data is imbalanced. Why no SMOTE?**
The imbalance ratio is only 1.30:1, which is mild. Stratified splitting preserves the proportion
in every partition, and the baseline check confirms models aren't just exploiting the prior — the
majority-class dummy gets 0.0000 F1 while real models exceed 0.94. Synthesising records would add
risk without addressing a real problem.

**Q8. Tell me something surprising you found in the data.**
Service ratings are documented as 1–5 but contain 0, and 0 is not the bottom of the scale. Passengers
who rated in-flight wifi 0 are **99.7% satisfied**, versus about 25% for ratings of 1–3. The 0 means
"not applicable" — the passenger never used the service. Treating it as "worst" would be a genuine
modelling error.

**Q9. So how did you encode it?**
As plain 0–5 numeric, deliberately, for comparability with the reference literature. I documented
the finding rather than re-encoding it. Because the strong models are non-linear — trees and neural
networks — they can fit the non-monotone relationship regardless. Adding an explicit
"not applicable" indicator is listed in Future Scope as a concrete extension.

---

## C. Methodology — the questions examiners press on

**Q10. Explain your data split.**
Stratified 70/15/15: 90,916 train, 19,482 validation, 19,482 test. Train fits the models and is the
only data cross-validation and hyperparameter search ever see. Validation drives early stopping and
model selection. Test was opened exactly once, at the very end, after the final model was chosen.

**Q11. Why three splits instead of the usual train/test?**
With only train/test, using the test set for early stopping or model selection leaks information
and inflates the reported score. The validation partition absorbs all selection decisions, keeping
the test estimate genuinely unbiased.

**Q12. How do you *know* there's no data leakage?**
Four mechanisms. (1) Splitting happens in exactly one function, so nothing can re-split. (2) All
preprocessing lives inside sklearn `Pipeline`s fitted on the training fold, so scalers and imputers
never see validation or test data. (3) A test asserts the three index sets are pairwise disjoint.
(4) A test asserts a scaler fitted on train centres the training data but *not* the validation data
— if validation were also perfectly centred, that would prove the scaler had seen it.

**Q13. Why don't you scale features for tree models?**
Decision trees split on thresholds within individual features, so they're invariant to any monotone
transformation. Scaling adds computation and obscures feature values in explanations without
changing predictions. I scale only for Logistic Regression, SVM, k-NN and the neural networks,
which depend on distances or gradient magnitudes.

**Q14. Why F1 as your selection metric rather than accuracy?**
With a 1.30:1 imbalance, accuracy is slightly inflated by the majority class — the dummy scores
0.5655 accuracy while being useless. F1 is the harmonic mean of precision and recall, so it only
rewards a model that handles the positive class well. I report accuracy, precision, recall, F1 and
ROC-AUC for completeness.

**Q15. Your engineered features — could they leak?**
No, and this is tested. All three are computed row-wise from that row's own values, so no
information passes between rows or across splits. `test_engineered_features_are_row_wise` proves
this by transforming a 50-row subset and asserting the values are identical to transforming the
full frame. None of them uses the target.

---

## D. Models

**Q16. Which models did you compare, and why those?**
Fourteen across four families: a dummy baseline (the floor); classical models (Logistic Regression,
Decision Tree, SVM, k-NN); ensembles (Random Forest, Extra Trees, XGBoost, Gradient Boosting,
Voting, Stacking); and three neural networks. This covers the linear, non-linear, bagged, boosted
and deep approaches, so the comparison spans the realistic option space.

**Q17. Why start with a dummy classifier?**
To establish that the models learn something real. It predicts the majority class always — 0.5655
accuracy, 0.0000 F1. Without that reference point, "96% accuracy" is uninterpretable; against it,
the improvement is clearly genuine.

**Q18. Describe your neural network architectures.**
Three MLPs. Basic: Dense(256) → BatchNorm → ReLU → Dropout(0.3), repeated, then a sigmoid output —
43,009 parameters. Regularized: the same with L2 (1e-4) and dropout raised to 0.45. Deep: four
hidden blocks tapering 384 → 192 → 96 → 48, 112,513 parameters. Widths derive from the encoded
feature count (32) rather than being copied from an arbitrary reference.

**Q19. Why no CNN, LSTM or transformer?**
Because this is tabular data with no spatial or temporal structure. A CNN assumes local spatial
correlation among adjacent features; an LSTM assumes sequence ordering. Neither holds — column
order here is arbitrary. Using them would add complexity and compute without a defensible
rationale, and would be a red flag in a research context.

**Q20. How did you prevent overfitting in the networks?**
Batch normalisation, dropout (0.3, and 0.45 in the regularized variant), L2 regularisation, and
EarlyStopping with patience 8 restoring the best weights. All three stopped early — at 49, 49 and
44 epochs against a 60-epoch budget — which confirms the regularisation was doing real work.

**Q21. Did the Voting and Stacking ensembles help?**
Not meaningfully. Voting reached 0.9595 F1 and Stacking 0.9594, versus 0.9597 for a single tuned
Random Forest. The reason is that the base learners are all tree ensembles making correlated
errors; combining correlated models yields little. A useful negative result — it answers RQ2 with
a qualified "no".

---

## E. Results and statistics

**Q22. What were your best results?**
Deep MLP led on test F1 at 0.9602 (accuracy 0.9660, ROC-AUC 0.9959), with Random Forest at 0.9597
(accuracy 0.9656). But those are within noise of each other — see Q23.

**Q23. You say the models are "statistically indistinguishable". Justify that.**
I computed percentile bootstrap 95% confidence intervals on test F1 using 1,000 resamples. The
leader's interval is [0.9573, 0.9631], which is 0.0059 wide. The top five models span only 0.00077
F1 — about an eighth of that width — and seven models have intervals overlapping the leader's. When
the difference is far smaller than the uncertainty, the honest conclusion is that no difference has
been demonstrated.

**Q24. Isn't "overlapping confidence intervals" a weak test?**
Yes, and I state that in the report. Overlapping intervals are conservative evidence of *no
demonstrated difference* — they don't prove equivalence, and they're not a formal significance
test. A stronger approach would be McNemar's test on paired predictions or a corrected
repeated-measures t-test; both are listed in Future Scope. The point stands directionally: the gaps
are well inside the noise floor.

**Q25. Then which model would you actually deploy?**
XGBoost or Random Forest. They match the neural networks statistically, but XGBoost trains in 6.9
seconds versus 35.5 for the Deep MLP, produces a 1.4 MB artifact, and supports exact SHAP
explanations through `TreeExplainer`. When accuracy is tied, cost and interpretability should
decide — that's the practical recommendation in my conclusion.

**Q26. The paper reports 95.83% accuracy. You got 96.56% for Random Forest. Did you beat it?**
I'd avoid that framing. It's a different split, different preprocessing and independent tuning, so
the numbers aren't directly comparable. I treated the paper's figures purely as reference context,
never as a target — optimising toward a published number would be methodologically unsound. My
results happen to land slightly higher; plausible reasons are the dedicated validation partition,
the three engineered features and independent tuning.

**Q27. Why is Logistic Regression so much worse (0.8585 F1)?**
Because the decision boundary is substantially non-linear. My EDA showed no single feature has a
strong linear correlation with the target, and the service-rating relationship is non-monotone —
recall the 0 anomaly. A linear model cannot represent that; every non-linear model exceeds 0.94 F1.
The ~0.09 F1 gap is effectively a measure of how much signal is non-linear.

**Q28. Are your results reproducible?**
Yes. Seed 42 applied to Python, NumPy and TensorFlow. I ran the full pipeline three times and got
identical metrics to four decimal places. Every run writes `experiment_manifest.json` recording the
seed, split sizes, library versions, per-model timings and chosen hyperparameters.

---

## F. Explainability

**Q29. What does SHAP tell you?**
Applied via TreeExplainer to Random Forest, the top features by mean |SHAP| are Online Boarding
(0.1181), In-flight Wifi Service (0.1112), Type of Travel and Class. High ratings on the first two
push predictions toward *Satisfied*; personal travel and first-time customer status push toward
*Neutral or Dissatisfied*.

**Q30. So better wifi causes satisfaction?**
No — and this distinction matters. SHAP describes how the *fitted model* uses a feature. It's
associational, not causal. I can say the model places substantial predictive weight on wifi rating;
I cannot say improving wifi would raise satisfaction. That would need a controlled experiment.
Wifi rating may also proxy for overall service quality or cabin class.

**Q31. Why permutation importance for the neural network instead of SHAP?**
SHAP's exact TreeExplainer only works for tree models. For the network I'd need KernelExplainer,
which is model-agnostic but computationally prohibitive at this dataset size — it requires many
perturbed evaluations per prediction. Permutation importance gives a defensible global measure
cheaply. The limitation is documented rather than hidden.

**Q32. Do the two methods agree?**
Largely, which is reassuring. Both rank In-flight Wifi Service and Type of Travel near the top.
The network weights Gate Location more highly than the trees do. Convergence across two independent
methods on two different model families strengthens confidence in the attribution.

---

## G. Engineering and deployment

**Q33. Walk me through your project structure.**
`config.py` holds all configuration. `src/` has one module per concern — loading, preprocessing,
feature engineering, the model registry, ML training, DL training, evaluation, explainability,
plotting, utilities. `train.py` orchestrates end to end. `app/` is the Streamlit interface, `tests/`
the test suite, `outputs/` the generated artifacts, `report/` the documentation.

**Q34. What is the experiment registry and why does it matter?**
`src/registry.py` declares each model once as a dataclass — name, family, factory, scaling
requirement, search space, mode flags. Training iterates the registry instead of hard-coding model
lists. So adding a model is a one-line change, every model is evaluated identically, and a model
that fails is caught and reported in the results table rather than silently vanishing.

**Q35. What happens if a model crashes mid-run?**
It's caught, its traceback is recorded in the manifest, and it appears in the results table marked
FAILED with the reason. The run continues. Silently skipping a model would misrepresent the
comparison. In the final run, all 14 succeeded.

**Q36. What does your test suite cover?**
32 tests: dataset loading and column detection, split integrity (disjointness, stratification,
proportions), preprocessing and the anti-leakage assertions, feature engineering row-wise
behaviour, model training and save/load round-trips, and the exact app inference path from a raw
input dictionary to a probability. There's also a test asserting no model achieves exactly 1.0
accuracy, since that would signal leakage.

**Q37. Tell me about a bug you found and fixed.**
When a neural network won, `best_model.pkl` was never written — Keras models don't pickle reliably
— yet the log claimed it had been saved, and the test passed only because a stale file from an
earlier run was sitting there. I fixed it so `best_model.pkl` always stores a pointer to the
`.keras` file and its preprocessor, and strengthened the test to verify the pointer's targets
actually exist on disk. I confirmed the new test fails when the artifact is removed.

**Q38. Why Python 3.11 and not the latest?**
TensorFlow doesn't support Python 3.14, which was the system default. I also had to
`brew install libomp`, because XGBoost needs OpenMP and fails to load without it on macOS ARM.
Both are documented as prerequisites in the README.

**Q39. How does the app avoid drifting out of sync with the model?**
Training writes `models/feature_schema.json` describing each raw feature — type, range, median,
categories. The app generates its input widgets from that schema, so the form always matches what
the model was trained on. It also never retrains: it loads saved artifacts, cached with
`@st.cache_resource`.

**Q40. Where would you deploy this, and where couldn't you?**
Streamlit Community Cloud — free, GitHub-connected. It *cannot* go on Netlify or GitHub Pages:
those are static hosts, and Streamlit needs a persistent Python server with WebSocket support.
I checked artifact sizes for the 1 GB memory limit and applied compression, cutting the models
directory from 1.3 GB to 256 MB; the files the app actually needs total under 2 MB.

---

## H. Critical questions

**Q41. What's the weakest part of your project?**
The tuning budget — 12 candidates × 3 folds is lean. A larger search could reorder the leading
models slightly. I chose it for tractability and state it explicitly so the evidence isn't
overstated. That said, since the top seven are already statistically tied, more tuning would most
likely tighten the tie rather than break it.

**Q42. Why do all the good models plateau around 0.96?**
Most likely an information ceiling in the features themselves. Seven architecturally different
models converging on the same value suggests they've extracted the available signal, and the
remaining ~4% of cases aren't resolvable from these 22 features. More architecture won't help;
richer data — free-text reviews, booking history — might.

**Q43. If you had another month, what would you do?**
Three things. Add McNemar's test for a formal pairwise comparison instead of relying on
overlapping intervals. Run the ablation on the rating-0 encoding to quantify what the "not
applicable" indicator is worth. And test generalisation on a second dataset from a different
airline, since every conclusion here is conditioned on one dataset of unknown provenance.

**Q44. Did deep learning "win"?**
On the point estimate it ranked first, but the honest answer is no — it tied. The valuable finding
is the tie itself: it contradicts the common assumption that tree ensembles clearly dominate
tabular data, while also showing deep learning offers no practical advantage here given its higher
training cost and weaker explainability.

**Q45. What did you learn?**
That measuring uncertainty matters as much as measuring performance. My first instinct was to
report the Deep MLP as the best model. Computing confidence intervals showed the margin was an
eighth of the noise, which changed the conclusion entirely — from "deep learning wins" to "these
are tied, so choose on cost and interpretability." That's a more useful and more defensible result.
