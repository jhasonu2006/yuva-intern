# Data Science Tool Mastery Project

A hands-on portfolio project demonstrating proficiency with the core Python data science stack:
**NumPy, pandas, Matplotlib, Seaborn, SciPy and scikit-learn**, run in Jupyter Notebook.

## Dataset
Breast Cancer Wisconsin (Diagnostic) - 569 samples, 30 numeric features. It ships with
scikit-learn (`sklearn.datasets.load_breast_cancer`), so no download is required.

## What is covered
| Tool | What is demonstrated |
|---|---|
| NumPy | random arrays, percentiles, matrix maths, vectorised z-score |
| pandas | inspection, missing/duplicate checks, groupby, qcut binning, crosstab |
| Matplotlib / Seaborn | histogram + KDE, boxplot, correlation heatmap, scatter, confusion matrix, bar chart |
| SciPy | Welch t-test, Pearson correlation |
| scikit-learn | train/test split, pipelines, cross-validation, Logistic Regression, Random Forest, metrics |

## How to run
```bash
pip install -r requirements.txt
jupyter notebook Data_Science_Tool_Mastery.ipynb
# or run the plain script:
python tool_mastery.py
```

## Files
- `Data_Science_Tool_Mastery.ipynb` - main notebook
- `tool_mastery.py` - same code as a script (saves charts to `figures/`)
- `run_output.txt` - captured console output
- `model_results.csv` - model comparison table
- `figures/` - generated charts

## Key results
- Logistic Regression (scaled): 98.25% test accuracy, F1 0.986
- Random Forest: 95.61% test accuracy, F1 0.966
- Mean radius differs strongly between malignant and benign tumours (Welch t = 22.21, p < 0.001)
