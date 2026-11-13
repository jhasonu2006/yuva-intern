# Final Data Science Capstone – Customer Churn Prediction (Yuva Intern, Task 6)

End-to-end machine learning project predicting telecom customer churn (TeleConnect India, simulated data, 7,000 customers).

## Contents
- `capstone.py` – full pipeline: data generation, cleaning, feature engineering, EDA, model training, evaluation, threshold optimisation, error analysis, charts
- `teleconnect_churn_dataset.csv` – cleaned dataset
- `metrics.json` – all results used in the report
- `g1_eda_churn.png` … `g6_importance.png` – charts
- `Task6_Capstone_Report.docx` – full project report

## Run
```
pip install pandas numpy matplotlib scikit-learn
python capstone.py
```

## Method
Logistic Regression, Random Forest and Gradient Boosting compared with 5-fold stratified cross-validation; best model chosen by ROC-AUC; decision threshold tuned on training data for business net benefit; permutation importance and error analysis.

## Key results
- Churn rate 38.3%; test ROC-AUC about 0.83 (Logistic Regression)
- Top drivers: contract type, tenure, internet service, support calls
- Targeting by model risk is profitable; offering to everyone is not
