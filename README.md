# Kaggle Diabetes Prediction: A Data Science Project

This repository documents a comprehensive data science workflow designed to predict diabetes risk using patient health records. Developed for the Kaggle Diabetes Prediction challenge (S5E12), this project moves beyond standard modeling to incorporate **clinically grounded feature engineering**, **AI-driven variable discretization**, and **advanced hyperparameter optimization**.

**Tech Stack:** Python, Pandas, Scikit-learn, CatBoost, LightGBM, SHAP, Optuna, Seaborn, Matplotlib.

## Data Access
The dataset for this project is sourced from the Kaggle Diabetes Prediction Competition.
* **Source:** https://www.kaggle.com/competitions/playground-series-s5e12
* **Setup:** Download `train.csv` and `test.csv` from Kaggle and place them in a folder named `data` in the root directory.

## Sequence of Notebooks

### 1. Exploratory Data Analysis
**Notebook:** `EDA.ipynb`


This notebook establishes a baseline understanding of the dataset through **model-based EDA**, moving beyond standard correlation checks to identify complex, non-linear relationships.

- **Baseline Modeling:** Trained an initial **CatBoostClassifier** to serve as a reference point for feature importance. CatBoost was selected for its robust handling of categorical variables and strong performance on tabular data without extensive preprocessing.
- **SHAP Value Analysis:** Utilized **SHAP (SHapley Additive exPlanations)** to interpret the model's predictions.
    - Generated **SHAP feature importance plots** to rank predictors by their marginal contribution to diabetes risk.
    - Created **SHAP dependence plots** (e.g., for `screen_time_hours_per_day`) to visualize non-linear trends and identify interaction effects with other variables like physical activity.

**Rationale:**
Standard correlation matrices often miss non-linear dependencies (common in healthcare data, e.g., risk thresholds for BMI or sleep). By training a gradient boosting model first and analyzing it with SHAP, we uncover *how* the model uses features to make predictions. This reveals critical insights, such as specific cut-off points for binning or interaction candidates, that inform the feature engineering strategies in subsequent steps.


### 2. Feature Engineering by Clincial Indicators
**Notebook:** `clinical_indicators_baseline.ipynb`

This notebook focuses on transforming raw continuous data into medically relevant risk indicators and establishing a reliable performance benchmark.

- **Threshold-Based Feature Engineering:** Converted continuous variables into diagnostic categories based on clinical guidelines and SHAP insights from the previous step.
    - **Metabolic Indicators:** Created flags for `obesity` (BMI > 30), `hypertension` (BP > 130/90), and cholesterol risks (e.g., `high_ldl`, `low_hdl`).
    - **Lifestyle Flags:** Engineered composite features like `sedentary_risk` (combining high screen time with low physical activity) and sleep abnormalities (`short_sleep` / `long_sleep`).
    - **Composite Risk Factors:** Synthesized interaction terms such as `diabetic_dyslipidemia` (High Triglycerides + Low HDL) to capture complex risk profiles.
- **Baseline Model Benchmarking:** Established a rigorous testing pipeline using Stratified K-Fold Cross-Validation (3 folds) to compare four distinct model architectures:
    - **Logistic Regression & Random Forest:** To test linear and standard tree-based performance.
    - **CatBoost & LightGBM:** To evaluate gradient boosting capabilities.
- **Performance Evaluation:** Validated that gradient boosting models (LightGBM and CatBoost) outperformed the linear and bagging baselines (Mean AUC ~0.725 vs 0.695), identifying them as the primary candidates for tuning.

**Rationale:**
In medical diagnosis, risk is often defined by specific thresholds (e.g., "High Blood Pressure") rather than a linear continuum. Explicitly engineering these "risk flags" simplifies the learning task for the models, allowing them to capture non-linear step changes in diabetes risk. Testing multiple model families early ensures that development efforts are focused on the architecture best suited for this specific data structure.


### 3. Feature Engineering by Discretization of Continuous Features
**Notebook:** `threshold_bins_baseline.ipynb`

This notebook explores **variable discretization** (binning) to handle non-linearity and reduce noise in continuous features, comparing traditional statistical methods against supervised machine learning techniques.

- **Binning Strategies:**
    - **Simple Quantile Binning:** Partitioned continuous features (e.g., Age, BMI) into fixed quantiles (e.g., deciles) to smooth out outliers and capture broad trends.
    - **AI-Driven Binning:** Utilized shallow **Decision Trees** (max_depth=3) to mathematically identify optimal split points that maximize information gain. This effectively treats the "optimal cut-off" finding as a supervised learning task.
- **Model Comparison:** Retrained baselines using these new binned features alongside raw data.
- **Key Findings:**
    - **Logistic Regression** showed improvement from the addition of binned features, suggesting that binning helps linear models by introducing structured non-linearity.
    - **LightGBM** showed a strong preference for **Simple Bins**, suggesting that smoothing continuous noise helped the gradient boosting algorithm find cleaner splits.
    - **CatBoost** maintained a balanced reliance on **Raw Features** (Family History, Activity) while effectively utilizing **AI-Bins** for specific metabolic markers (e.g., triglycerides), proving that binned features serve as powerful non-linear representations.
    - 

**Rationale:**
While tree-based models can naturally handle continuous data, they often waste splits trying to approximate simple step functions or overfit to minor fluctuations in the data. By explicitly providing "AI-optimized" bins (e.g., grouping Age into clusters where diabetes risk is statistically similar), we inject domain-specific non-linearity directly into the dataset, allowing the model to focus on higher-order interactions rather than finding basic thresholds.

---

### 4. Hyperparameter Tuning & Final Prediction
**Notebook:** `final_model.ipynb`

This notebook represents the culmination of the project, combining the most effective engineered features with rigorous hyperparameter optimization to build the final predictive model.

- **Feature Consolidation:** Aggregated the best-performing feature set identified in previous steps, merging "Clinical Indicators" (e.g., `diabetic_dyslipidemia`, `sedentary_risk`) with "AI-Driven Bins" (e.g., optimal splits for Age and Triglycerides).
- **Bayesian Optimization (Optuna):** Implemented **Optuna** to perform an automated, efficient search for the optimal model hyperparameters.
    - **Search Space:** Tuned critical parameters for **CatBoost** including `tree_depth`, `learning_rate`, `l2_leaf_reg`, and stochastic settings like `random_strength` and `bagging_temperature`.
    - **Optimization Goal:** Maximized the 3-Fold Stratified Cross-Validation AUC to ensure the model generalizes well to unseen data.
- **Final Model Architecture:** Trained a fully tuned **CatBoostClassifier** on the complete training dataset.
    - *Note:* While LightGBM was also evaluated, the final submission file was generated using the optimized CatBoost predictions due to its superior handling of the categorical risk flags created in earlier steps.

**Rationale:**
Default model parameters rarely yield optimal results. Manual tuning is time-consuming and often misses complex interactions between parameters (e.g., how tree depth affects the optimal learning rate). By using Optuna's Bayesian optimization, we efficiently explored the hyperparameter space to squeeze the maximum performance out of our feature engineering efforts without overfitting.

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
