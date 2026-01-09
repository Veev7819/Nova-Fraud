## Transaction Fraud Detection & Scoring API
An end-to-end fraud detection system that transforms raw transaction data into explainable, production-ready fraud scores, complete with model evaluation, SHAP-based interpretability, and a local FastAPI scoring service.

## Project Overview
This project builds and deploys machine learning models to detect potentially fraudulent financial transactions.
It focuses not only on predictive performance, but also on interpretability, auditability, and real-world usability.
Key capabilities include:
Fraud-aware feature engineering
Robust model training with imbalance handling
Clear performance evaluation
SHAP-based explainability (local & global)
A local REST API for real-time scoring

## Objectives
- Prepare a clean, consistent transaction dataset for fraud modeling and analytics
- Explore and visualize fraud patterns across channels, corridors, and time
- Train and evaluate strong baseline models (Logistic Regression and Random Forest) using time-aware validation
- Compare advanced models (XGBoost, LightGBM) with imbalance handling
- Generate reproducible metrics, plots, and artifacts
- Build a local API for real-time fraud scoring
- Provide explainability to turn model outputs into auditable decisions



🗂️ Project Structure
Nova_Fraud/
│
├── data/
│   └── Nova_pay_features.csv
│
├── notebooks/
│   ├── 01_Data_cleaning.ipynb
│   ├── 02_EDA.ipynb
│   ├── 03_train_baseline_model.ipynb
│   ├── 04_advanced_models.ipynb
│   ├── 05_model_summary_explainability.ipynb
│   └── 06_api_test.ipynb
│
├── models/
│   ├── model_log_reg.joblib
│   ├── model_rf.joblib
│   ├── model_xgb.joblib
│   └── model_lgbm.joblib
│
├── plots/
│   ├── confusion_matrices.png
│   ├── roc_pr_curves.png
│   ├── roc_curves.png
│   └── pr_curves.png
│
├── API/
│   ├── main.py
│   ├── requirements.txt
│
├── .gitignore
├── README.md



## Notebook Flow Overview
01_Data_cleaning: Prepare a clean, consistent transaction dataset
02_EDA: Analyze fraud patterns by channel, corridor, and time
03_train_baseline_model: Train Logistic Regression and Random Forest baselines
04_advanced_models: Train XGBoost and LightGBM with imbalance handling
05_model_summary_explainability: Compare models and explain predictions using SHAP
06_api_test: Test the FastAPI fraud scoring service locally

## Modeling Approach
Baseline Models
Logistic Regression (class-weighted)
Random Forest (balanced subsampling)

Advanced Models
XGBoost + Random Undersampling
LightGBM + Random Undersampling

Evaluation Metrics
Precision
Recall
F1-score
ROC-AUC
False Positive Rate
Confusion Matrix


Model selection prioritizes precision–recall trade-offs, reflecting real-world fraud costs.

## Explainability with SHAP
This project uses SHAP (SHapley Additive exPlanations) to interpret model decisions.
What SHAP Provides
Local explanations: why a specific transaction was flagged
Global explanations: which features drive fraud risk overall

Example Insight
High transaction velocity, young accounts, and elevated IP risk increase fraud probability —
while trusted devices and stable behavior reduce it.
This turns the model from a black box into an auditable decision system.

## Fraud Scoring API
A local FastAPI service exposes the trained model for real-time scoring.
Start the API
cd API
uvicorn main:app --reload

Health Check
GET http://localhost:8000/health

Score Transactions
POST http://localhost:8000/score

Sample Request
 {
  "items": [
    {
      "transaction_id": "demo-1",
      "home_country": "US",
      "source_currency": "USD",
      "dest_currency": "MXN",
      "channel": "mobile",
      "kyc_tier": "standard",
      "ip_country": "US",
      "new_device": false,
      "location_mismatch": false,
      "ip_country_missing": false,
      "amount_src": 120.0,
      "amount_usd": 120.0,
      "fee": 2.1,
      "ip_risk_score": 0.2,
      "device_trust_score": 0.7,
      "account_age_days": 240,
      "txn_velocity_1h": 0,
      "txn_velocity_24h": 1,
      "corridor_risk": 0.05,
      "risk_score_internal": 0.2,
      "hour": 14,
      "dayofweek": 2
    }
  ]
}

Sample Response
{
  "results": [
    {
      "transaction_id": "demo-1",
      "score": 0.0033333333333333335,
      "decision": "allow"
    }
  ]
}


## Key Takeaway
This project demonstrates how to move from raw transaction data to explainable, deployable fraud decisions, balancing accuracy, trust, and operational realism.

## Tech Stack
Python, Pandas, NumPy
Scikit-learn, XGBoost, LightGBM
Imbalanced-learn
SHAP
FastAPI, Uvicorn
Matplotlib, Seaborn




