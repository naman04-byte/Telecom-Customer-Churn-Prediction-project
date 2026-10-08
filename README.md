# Telecom Customer Churn Prediction
Churn prediction on an imbalanced telecom dataset (73% stay / 27% churn) using Logistic Regression, Random
Forest and XGBoost, tuned with GridSearchCV (5-fold CV) optimizing F1-score. Best model achieved ROC-AUC ≈ 0.85,
catching ~79% of churners; F1 improved 0.60 → 0.62 via decision-threshold tuning (0.50 → 0.56).
Customers ranked by churn probability so retention offers target the highest-risk segment first.
**Tech:** Python, pandas, scikit-learn, XGBoost
