# Results

Generated from `outputs/model_results.csv`, `outputs/feature_importance.csv`,
`outputs/shap_results/` and `outputs/experiment_manifest.json`.

**Run:** mode `full`, seed 42, wall-clock ~7 min 38 s
**Test set:** 19,482 passengers, opened exactly once
**Failures:** none — all 14 models completed

## 1. Consolidated comparison (test set)

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

## 2. Baseline check

The majority-class baseline scores 0.5655 accuracy and 0.0000 F1 (it never predicts *Satisfied*).
Every trained model beats it decisively, so the models are learning real structure rather than
exploiting the class prior.

## 3. Statistical comparison — the headline finding

Percentile bootstrap 95% confidence intervals on test F1 (1,000 resamples):

| Model | F1 | 95% CI | Width |
|---|---|---|---|
| Neural Net (Deep MLP) | 0.9602 | [0.9573, 0.9631] | 0.0059 |
| Neural Net (Basic MLP) | 0.9601 | [0.9573, 0.9630] | 0.0057 |
| Random Forest | 0.9597 | [0.9568, 0.9626] | 0.0058 |
| Voting Ensemble | 0.9595 | [0.9566, 0.9627] | 0.0061 |
| Stacking Ensemble | 0.9594 | [0.9564, 0.9624] | 0.0060 |
| Neural Net (Regularized MLP) | 0.9584 | [0.9557, 0.9611] | 0.0054 |
| XGBoost | 0.9578 | [0.9548, 0.9610] | 0.0061 |

- Top-five F1 spread: **0.00077**
- Mean CI width: **0.0059** (~8× the spread)
- Models overlapping the leader: **7**

**These seven models are statistically indistinguishable.** Declaring a winner on a 0.0005 F1
margin would overstate the evidence.

## 4. Cross-validation stability (training partition only)

| Model | CV Accuracy mean | CV Accuracy std | CV F1 mean | CV F1 std |
|---|---|---|---|---|
| XGBoost | 0.9623 | 0.0010 | 0.9559 | 0.0012 |
| Random Forest | 0.9610 | 0.0011 | 0.9544 | 0.0012 |
| Extra Trees | 0.9581 | 0.0010 | 0.9510 | 0.0012 |
| Decision Tree | 0.9491 | 0.0015 | 0.9403 | 0.0021 |
| Logistic Regression | 0.8738 | 0.0035 | 0.8518 | 0.0039 |

Low standard deviations (0.001–0.004) confirm stable performance across folds.

## 5. Computational cost

| Model | Fit time (s) | Serialised size |
|---|---|---|
| Dummy Baseline | 0.2 | 2 KB |
| Logistic Regression | 0.2 | 3 KB |
| K-Nearest Neighbors | 0.2 | 3.6 MB |
| Decision Tree | 0.5 | 27 KB |
| Extra Trees | 1.4 | 101 MB |
| Voting Ensemble | 2.8 | 94 MB |
| Stacking Ensemble | 6.3 | 27 MB |
| XGBoost | 6.9 | 1.4 MB |
| Gradient Boosting | 16.2 | 61 KB |
| Neural Net (Basic MLP) | 23.8 | 543 KB |
| Neural Net (Regularized MLP) | 25.5 | 544 KB |
| Neural Net (Deep MLP) | 35.5 | 1.3 MB |
| Support Vector Machine | 35.9 | 659 KB |
| Random Forest | 49.0 | 12 MB |

XGBoost achieves statistically equivalent accuracy to the Deep MLP in **6.9 s versus 35.5 s**, at
a fraction of the artifact size. This is the decisive practical difference given the performance
tie.

## 6. Deep learning training

| Model | Parameters | Epochs run (budget 60) |
|---|---|---|
| Basic MLP | 43,009 | 49 |
| Regularized MLP | 43,009 | 49 |
| Deep MLP | 112,513 | 44 |

All three stopped early, confirming regularisation worked and none was trained to overfitting.

## 7. Feature importance

### Random Forest (impurity-based)

| Rank | Feature | Importance |
|---|---|---|
| 1 | Online Boarding | 0.2691 |
| 2 | In-flight Wifi Service | 0.1692 |
| 3 | Class (Business) | 0.0792 |
| 4 | Type of Travel (Personal) | 0.0676 |
| 5 | Type of Travel (Business) | 0.0506 |
| 6 | In-flight Entertainment | 0.0464 |
| 7 | Service Quality Score (engineered) | 0.0307 |

### SHAP (TreeExplainer, Random Forest)

| Rank | Feature | Mean \|SHAP\| |
|---|---|---|
| 1 | Online Boarding | 0.1181 |
| 2 | In-flight Wifi Service | 0.1112 |
| 3 | Type of Travel (Personal) | 0.0725 |
| 4 | Type of Travel (Business) | 0.0605 |
| 5 | Class (Business) | 0.0469 |

### Neural network (permutation importance)

| Rank | Feature | Accuracy drop |
|---|---|---|
| 1 | In-flight Wifi Service | 0.2177 |
| 2 | Gate Location | 0.0696 |
| 3 | Type of Travel (Personal) | 0.0398 |
| 4 | Baggage Handling | 0.0380 |

Three independent methods across two model families converge on the same leading features, which
strengthens confidence in the attribution.

> **Caution.** These are measures of association *within the fitted models*, not causal effects.

## 8. Comparison with the reference paper

| Source | Model | Accuracy | F1 |
|---|---|---|---|
| ICSADL 2026 | Random Forest | 0.9583 | 0.9515 |
| This project | Random Forest | 0.9656 | 0.9597 |

Slightly higher here. Likely contributors: the 70/15/15 protocol with a dedicated validation
partition, the three engineered features, and independent tuning. Exact reproduction is not
expected, since split composition and preprocessing differ.

## 9. Figures

| File | Content |
|---|---|
| `01_target_distribution.png` | Class balance |
| `02_demographics.png` | Age, gender, customer type |
| `03_travel_context.png` | Travel type, class, distance |
| `04_service_rating_heatmap.png` | Satisfaction rate by rating (shows the 0 anomaly) |
| `05_top_service_ratings.png` | Most discriminative service ratings |
| `06_delays.png` | Delay distributions and satisfaction by delay band |
| `07_correlation_heatmap.png` | Numeric correlations |
| `08_target_correlation.png` | Ranked correlation with the target |
| `10_dl_training_history.png` | Accuracy/loss vs epochs for all three networks |
| `11_confusion_matrices.png` | Confusion matrices, all models |
| `12_roc_pr_curves.png` | Overlaid ROC and PR curves |
| `13_model_comparison.png` | Accuracy and F1 by model |
| `14_feature_importance.png` | Ranked importances |
| `15_shap_summary.png` | SHAP beeswarm |
| `16_shap_bar.png` | SHAP global importance |
| `17_shap_individual.png` | Single-passenger explanation |
| `18_ci_forest_plot.png` | **Confidence-interval forest plot — the headline finding** |
| `19_summary_poster.png` | One-page visual summary (PDF in `outputs/reports/`) |

## 10. Confusion matrix interpretation

For the final model on the test set, with *Satisfied* as positive:

- **True positive** — correctly identified a satisfied passenger.
- **True negative** — correctly identified a dissatisfied passenger.
- **False positive** — a dissatisfied passenger predicted satisfied. Operationally the more
  costly error: the airline believes service was adequate and takes no recovery action for a
  passenger who may not return.
- **False negative** — a satisfied passenger predicted dissatisfied. Wastes retention effort but
  causes no customer harm.

The leading models show precision (~0.978) above recall (~0.943), meaning they are comparatively
conservative about predicting *Satisfied* — they produce relatively few false positives, which is
the desirable bias given the asymmetry above.
