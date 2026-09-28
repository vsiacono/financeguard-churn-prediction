# FinanceGuard — Customer Churn Prediction

Machine Learning project focused on predicting customer churn for a digital banking scenario and translating model performance into business decisions.

## Project objective

The project analyzes a dataset of **10,000 customers** to identify clients with a higher probability of churn. The workflow combines supervised learning, hyperparameter optimization, unsupervised learning and a business-oriented decision-threshold analysis.

## Workflow

1. **Exploratory Data Analysis & baseline model**  
   Data quality checks, class imbalance analysis, feature exploration, preprocessing and Logistic Regression as an interpretable baseline.

2. **Advanced supervised models**  
   Comparison of Random Forest, XGBoost, LightGBM and CatBoost, including cross-validation and hyperparameter optimization with Grid Search and Optuna. A Stacking ensemble is also evaluated.

3. **Customer segmentation**  
   K-Means and DBSCAN clustering, together with PCA and t-SNE, are used to explore customer segments and churn-risk patterns.

4. **Business threshold optimization**  
   A custom cost matrix is used to optimize the classification threshold according to business impact rather than relying on the default 0.5 cutoff.

## Key results

- The dataset contains approximately **20% churn**, making class-sensitive metrics especially relevant.
- Boosting models outperform the Logistic Regression baseline on the main discrimination metrics used in the project.
- CatBoost achieved the strongest individual-model results used for the final business-threshold exercise.
- The business-oriented threshold moved from **0.50 to 0.09** under the illustrative cost assumptions defined in the project.
- Estimated business value increased from **13,120 to 26,690**, an improvement of **103.4%**, while substantially reducing false negatives.
- The cost matrix is illustrative: in a production environment, its values should be calibrated using real customer lifetime value and retention-campaign costs.

## Technologies

**Python · Pandas · NumPy · Matplotlib · Seaborn · Scikit-learn · Statsmodels · XGBoost · LightGBM · CatBoost · Optuna · Jupyter Notebook**

## Repository structure

```text
financeguard-churn-prediction/
├── README.md
├── requirements.txt
├── .gitignore
├── data/
│   └── Churn_Modelling.csv
├── notebooks/
│   ├── 01_eda_logistic_regression.ipynb
│   ├── 02_gradient_boosting_optimization.ipynb
│   ├── 03_unsupervised_learning.ipynb
│   └── 04_business_threshold_optimization.ipynb
└── docs/
    └── model_report.pdf
```

## Notebook guide

- `01_eda_logistic_regression.ipynb` — EDA, preprocessing and Logistic Regression baseline.
- `02_gradient_boosting_optimization.ipynb` — Random Forest, XGBoost, LightGBM, CatBoost, Grid Search, Optuna and Stacking.
- `03_unsupervised_learning.ipynb` — K-Means, DBSCAN, PCA and t-SNE.
- `04_business_threshold_optimization.ipynb` — cost-sensitive threshold optimization and business-impact comparison.

## How to run

Create a Python environment and install the dependencies:

```bash
pip install -r requirements.txt
```

The notebooks are organized to load the dataset from `../data/Churn_Modelling.csv` when they are run from the `notebooks/` directory.

## Business interpretation

The project illustrates why selecting a model only by accuracy can be insufficient in churn problems. When the cost of failing to identify a customer who will leave is higher than the cost of contacting a customer unnecessarily, the decision threshold becomes a business parameter. The final exercise quantifies that trade-off explicitly.

## Documentation

A consolidated project report is available in `docs/model_report.pdf`.

---

*Data Science — Henry, Module 4 project (2026).*
