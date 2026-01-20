# Diabetes Risk Prediction (Clinical Indicators + SHAP-driven Feature Engineering)

This project builds an end-to-end machine learning pipeline for diabetes risk prediction using clinical and lifestyle indicators. The workflow emphasises interpretability and robust model validation by combining:

- **Model-based EDA using SHAP**
- **Clinically motivated feature engineering**
- **Leakage-safe AI-based threshold binning**
- Strong tabular models (**CatBoost + LightGBM**) with hyperparameter tuning (Optuna)

---

## Project Method Overview

### 1) Model-based EDA with CatBoost + SHAP
Instead of relying only on correlation tables or univariate plots, this project uses **model-based EDA**:

- A **baseline CatBoost classifier** is trained first as an interpretable and strong tabular benchmark.
- SHAP values are computed to:
  - rank feature importance,
  - visualize non-linear feature effects,
  - inspect potential interactions / redundancy using SHAP dependence plots.

This guides downstream feature engineering decisions.

---

## Feature Engineering Approaches

Clinical metrics often contain threshold-driven meaning (e.g. obese BMI, high BP, elevated cholesterol). This motivates creating derived features aligned with real screening logic.

This repo experiments with three main feature strategies:

### A) Clinical indicator features (domain-based)
Threshold/flag features based on established clinical cut-offs such as:
- BMI categories
- Hypertension thresholds
- Lipid panel thresholds
- Lifestyle risk flags

These features provide clearer “risk signals” and improve interpretability.

### B) Threshold binning (discretization)
Continuous features are discretized to reduce noise and help models learn stable patterns. Two binning methods are tested:

1. **Simple quantile binning**
   - Fixed **20 bins** per continuous feature (heuristic baseline)

2. **AI-based binning via Decision Trees**
   - Fit a **single-feature decision tree** to predict `y`
   - Extract split thresholds as bin edges
   - Use `max_depth = 3` → up to **8 bins (2³)**  
     (keeps bins interpretable + limits overfitting)

✅ AI-binning is **supervised** (uses `y`) → must be done inside CV to avoid leakage.

### C) Selective binning (final approach)
For CatBoost, threshold bins are applied **only to the most important features**, based on feature importance plots, since binning everything adds noise/redundancy.

---

## Model Training & Evaluation

### Baseline models
This repo trains and compares 4 core models:

- **Logistic Regression (LR)**  
  Interpretable linear baseline.
- **Random Forest (RF)**  
  Bagging trees; naturally captures thresholds and interactions.
- **CatBoost**  
  Strong tabular performance; strong with categorical/indicator/bin features.
- **LightGBM (LGBM)**  
  Efficient boosting model; excels with raw continuous features.

### Validation strategy
- **Stratified 3-Fold Cross-Validation**
- Out-of-fold (OOF) predictions to estimate generalization fairly
- Primary metric: **ROC-AUC**
  - Measures ranking quality across all decision thresholds
  - Robust to threshold selection and mild class imbalance

---

## Final Models

Based on CV performance and empirical testing:

### ✅ Final CatBoost
- Uses **selective threshold bins** (only top features)
- Binning provides a small but consistent AUC gain over baseline

### ✅ Final LightGBM
- Uses **raw features only**
- Trial-and-error showed bins reduced AUC, likely because LGBM already learns optimal thresholds from raw features

---

## Hyperparameter Tuning (Optuna)

Optuna tuning focuses on the most impactful bias–variance tradeoff parameters:

### CatBoost search space
- `depth`: tree complexity / interactions
- `learning_rate`: convergence stability
- `l2_leaf_reg`: regularization against overfitting
- `one_hot_max_size`: categorical encoding complexity

### LightGBM search space
- `learning_rate`: convergence stability
- `num_leaves`: main capacity parameter
- `max_depth`: structure constraint to avoid overfitting
- `min_child_samples`: leaf-level regularization

---

## Notebooks (Run Order)

### 1. `EDA.ipynb`
**Model-based EDA using CatBoost + SHAP**
- Train baseline CatBoost
- SHAP feature importance
- SHAP dependence plots (non-linearity + interaction patterns)
- Extract insights for feature engineering

### 2. `clinical_indicators_baseline.ipynb`
**Indicator/threshold feature engineering**
- Create clinically meaningful risk flags
- Train baseline models (LR/RF/CatBoost/LGBM) using engineered indicator features
- Compare AUC + interpret feature importance

### 3. `threshold_bins_baseline.ipynb`
**Threshold binning experiments**
- Simple quantile binning (20 bins)
- AI-based binning using decision tree thresholds (`max_depth=3`)
- Leakage-safe integration of AI-binning inside CV
- Model comparisons under binned feature representations

### 4. `final_model_1.ipynb`
**Final models + Optuna tuning**
- Final CatBoost (selective bins + Optuna tuning)
- Final LightGBM (raw features + Optuna tuning)
- Consolidated training pipeline and evaluation

---

## Repo Notes

### Data
Training and test CSVs are **not included** in the repo (excluded via `.gitignore`) to prevent:
- file size issues on GitHub
- redistribution of restricted datasets
- accidental data leakage

Place your data locally (example):
