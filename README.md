# Telco Customer Churn — Prediction & Business Impact Analysis

An end-to-end churn analytics project that goes beyond model accuracy to answer the question that actually matters to a business: **which customers should we spend our retention budget on, and how much money does that save us?**

The project moves through four stages — exploratory analysis, leakage-safe preprocessing, model development, and a cost-based business impact simulation — using the classic [Telco Customer Churn dataset](https://www.kaggle.com/datasets/jazidesigns/telecom-dataset) (7,043 customers, 21 features).

---

## 📌 Project Objective

Predict which telecom customers are likely to churn, understand *why* they churn, and translate model predictions into a concrete, budget-constrained retention strategy — rather than stopping at a ROC-AUC score.

---

## 🗂️ Repository Structure

| Notebook | Purpose |
|---|---|
| `01-eda.ipynb` | Exploratory data analysis — data quality checks, univariate/bivariate analysis, and interaction effects between tenure, contract type, and payment behavior |
| `02-preprocessing.ipynb` | Leakage-safe feature engineering pipeline using `ColumnTransformer` + `Pipeline`, stratified train/test split |
| `03-modeling-evaluation.ipynb` | Model training (Logistic Regression, Random Forest, XGBoost), hyperparameter tuning, and decision-threshold optimization |
| `04-business-impact-analysis.ipynb` | Cost-sensitive evaluation, threshold-vs-cost sensitivity analysis, and budget-constrained retention targeting |

---

## 🔍 1. Exploratory Data Analysis

Key patterns uncovered before any modeling was done:

- **Churn is a lifecycle problem.** Churn rate exceeds 50% within the first 3 months of tenure and drops below 10% after 4 years. Median tenure for churned customers is 10 months vs. 38 months for retained customers.
- **Contract type dominates.** Month-to-month customers churn at far higher rates than one- or two-year contract holders, who show minimal churn regardless of tenure.
- **Payment friction matters.** Customers on electronic checks churn at ~45%, compared to ~15–17% for customers on automatic payment methods.
- **Value-added services reduce churn.** Customers without online security or tech support churn at ~42%, vs. ~15% for those with these services — even after controlling for internet service type.
- **The riskiest segment** is the intersection of these factors: month-to-month + electronic check customers, where churn exceeds 50%.
- **Weak signals:** gender, phone service, and entertainment add-ons (streaming TV/movies) show little standalone relationship with churn.
- A single data quality issue was found and resolved: 11 records had blank `TotalCharges` values (all zero-tenure customers), which were removed as a negligible, low-information edge case.

---

## 🛠️ 2. Preprocessing

- Target (`Churn`) mapped to binary; identifier column dropped.
- `TotalCharges` corrected from string to numeric.
- `SeniorCitizen` re-classified as categorical (it's a binary flag, not a continuous number).
- Built a `ColumnTransformer` pipeline:
  - **Numeric features** → median imputation + standard scaling
  - **Categorical features** → most-frequent imputation + one-hot encoding (unseen categories handled gracefully)
- Stratified 80/20 train-test split, with the pipeline **fit only on training data** to prevent leakage.
- Processed arrays and feature names are serialized (`.npy`) for direct use in modeling.

---

## 🤖 3. Modeling & Evaluation

Three model families were evaluated to check whether additional model complexity actually helps:

| Model | Test ROC-AUC | Notes |
|---|---|---|
| **Logistic Regression** (baseline) | ~0.84 | Strong, stable, interpretable |
| Random Forest | Overfit train, underperformed on test | Bagging alone didn't generalize better |
| XGBoost (tuned via `GridSearchCV`) | Marginal gain over LR | Gain too small to justify added complexity |

**Final model: Logistic Regression** — selected for its interpretability and near-equivalent performance to the tuned boosting model.

**Key churn drivers identified by the model** (consistent with EDA): month-to-month contracts, fiber optic internet, absence of security/tech support, and electronic check payments increase churn risk; longer tenure and long-term contracts reduce it.

### Decision Threshold Optimization
Since churn prediction is a **cost-sensitive** problem (missing a churner is costlier than a false alarm), the default 0.5 threshold was replaced. A threshold of **0.35** was selected, capturing ~71% of churners while keeping precision at an operationally reasonable level.

Artifacts produced: `final_logistic_model.pkl`, `final_threshold.json`.

---

## 💰 4. Business Impact Analysis

The final stage reframes the model's output as a **cost-minimization decision problem** rather than a classification metric.

**Cost framework:**
- Cost of losing a churned customer (no action taken): high
- Cost of a retention incentive offered to a predicted churner: low

**Results:**
- Applying the model at the optimized threshold reduces total expected churn-related cost by **~45%** vs. taking no action.
- A **threshold-vs-cost sensitivity analysis** (across multiple churn-to-retention cost ratios) confirmed the strategy is robust: lower thresholds consistently conserve more cost, and this holds even as cost assumptions are varied.
- A **budget-constrained targeting simulation** was run to reflect real-world retention campaigns with fixed capacity: customers are ranked by predicted churn probability, and the top-N highest-risk customers are contacted until the budget is exhausted.
  - This ranking-based approach outperforms a fixed-probability cutoff — the optimal strategy targets the top-ranked customers directly rather than relying on a single decision threshold.
  - Implied probability cutoff at the optimal budget allocation: **~0.178**.

**Bottom line:** the real business value came less from squeezing out marginal ROC-AUC gains and more from optimizing *how the model's output is used* — threshold selection and ranked, budget-aware targeting.

---

## 🧰 Tech Stack

`Python` · `pandas` / `numpy` · `scikit-learn` · `XGBoost` · `matplotlib` / `seaborn` · `missingno` · `joblib`

---

## 🚀 Reproducing the Pipeline

```bash
# 1. EDA
jupyter notebook 01-eda.ipynb

# 2. Preprocessing (produces processed arrays + feature names)
jupyter notebook 02-preprocessing.ipynb

# 3. Modeling (trains LR/RF/XGBoost, tunes, saves final model + threshold)
jupyter notebook 03-modeling-evaluation.ipynb

# 4. Business impact (loads saved model, runs cost/budget simulations)
jupyter notebook 04-business-impact-analysis.ipynb
```

> Note: file paths in the notebooks (`/kaggle/input/...`, `/kaggle/working/...`) reflect a Kaggle environment. Update these to local paths if running outside Kaggle.

---

## 📈 Key Takeaways

1. Churn is concentrated at the intersection of **low commitment** (month-to-month), **high friction** (electronic check), and **early lifecycle stage** (low tenure).
2. Simpler, interpretable models (Logistic Regression) matched or beat more complex ones — the ceiling here is set by the feature set, not model capacity.
3. The largest gains in *business value* came from **decision-layer optimization** — choosing the right threshold and targeting policy — not from further model tuning.