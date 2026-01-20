# Diabetes Risk Prediction

This repository contains an end-to-end machine learning workflow for predicting diabetes risk using clinical and lifestyle indicators, built as an entry to the **Kaggle Diabetes Prediction Challenge**.

The pipeline emphasizes both **predictive performance** and **interpretability**, combining:
- model-based EDA (**CatBoost + SHAP**),
- clinically motivated feature engineering,
- leakage-safe **AI-based threshold binning**,
- strong tabular models (**CatBoost + LightGBM**) with **Optuna** tuning.

---

## Notebook Sequence (and what each notebook does)

### 1) `EDA.ipynb` — Model-based EDA using CatBoost and SHAP
This notebook performs interpretability-driven EDA:
- trains a **baseline CatBoost classifier** as a strong tabular benchmark
- computes **SHAP values** to obtain:
  - global feature importance
  - SHAP dependence plots (non-linear effects + feature interaction patterns)

Rationale: SHAP-based EDA provides more useful guidance for feature engineering than correlation-only approaches, especially for non-linear clinical risk signals.

---

### 2) `clinical_indicators_baseline.ipynb` — Clinical threshold indicator features
This notebook performs clinically motivated feature engineering:
- constructs **threshold-based indicator features** (risk flags) aligned with screening logic  
  (e.g., BMI categories, hypertension ranges, lipid thresholds)
- trains baseline models:
  - Logistic Regression (LR)
  - Random Forest (RF)
  - CatBoost
  - LightGBM (LGBM)
- evaluates models using stratified CV AUC + feature importance comparison

Rationale: Many clinical variables are interpreted via thresholds; converting raw values into risk ranges can create clearer predictive signals.

---

### 3) `threshold_bins_baseline.ipynb` — Threshold binning experiments
This notebook evaluates discretisation-based feature engineering:
- **Simple quantile binning** (20 bins) as a baseline discretisation method
- **AI-based binning**:
  - fit single-feature decision trees (max_depth = 3)
  - extract split thresholds as bin edges (up to 8 bins)

To prevent leakage, decision-tree binning (which uses `y`) is performed **inside stratified CV**, ensuring bin thresholds are learned only from training folds before applying to validation folds.

---

### 4) `final_model_1.ipynb` — Final models + Optuna tuning
This notebook consolidates final model selection and tuning:
- final model family: **CatBoost + LightGBM**
- CatBoost:
  - uses **selective threshold bin features** applied only to top continuous predictors
  - tuned using Optuna (`depth`, `learning_rate`, `l2_leaf_reg`, `one_hot_max_size`)
- LightGBM:
  - uses **raw features only**
  - tuned using Optuna (`learning_rate`, `num_leaves`, `max_depth`, `min_child_samples`)

Rationale:
- CatBoost benefits slightly from selective binning on key predictors
- LightGBM performs best on raw continuous inputs (bins reduced AUC empirically)

---

## Results (Cross-Validated ROC-AUC)

| Model | Indicators AUC (mean ± std) | Threshold Bins AUC (mean ± std) |
|------|------------------------------|----------------------------------|
| LR   | 0.695020 ± 0.000367          | 0.698481 ± 0.000595              |
| RF   | 0.701064 ± 0.000347          | 0.702733 ± 0.000601              |
| CatBoost | 0.723791 ± 0.000489      | **0.724559 ± 0.000364**          |
| LightGBM | **0.725294 ± 0.000474**  | 0.722209 ± 0.000250              |

Baseline reference:
- CatBoost baseline (raw features): **0.723558**

Summary:
- Boosting models (CatBoost / LightGBM) consistently achieve the best performance (AUC ≈ 0.72–0.73)
- Threshold bins provide a small uplift for CatBoost relative to baseline
- LightGBM still performs best using raw features, suggesting that discretisation is unnecessary for LGBM on this dataset
