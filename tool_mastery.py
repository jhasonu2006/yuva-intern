# %% [markdown]
# # Data Science Tool Mastery Project
# Tools covered: **NumPy, pandas, Matplotlib, Seaborn, SciPy, scikit-learn** (run inside Jupyter).
# Dataset: Breast Cancer Wisconsin (Diagnostic) - bundled with scikit-learn, so no download is needed.

# %%
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, classification_report)

sns.set_theme(style="whitegrid")
print("NumPy", np.__version__, "| pandas", pd.__version__, "| seaborn", sns.__version__)

# %% [markdown]
# ## 1. NumPy - array operations

# %%
rng = np.random.default_rng(42)
a = rng.normal(loc=50, scale=10, size=1000)
print("mean:", a.mean().round(2), "std:", a.std().round(2))
print("percentiles 25/50/75:", np.percentile(a, [25, 50, 75]).round(2))
m = rng.integers(1, 10, size=(3, 3))
print("matrix:\n", m)
print("determinant:", round(np.linalg.det(m), 2))
print("column means:", m.mean(axis=0))
z = (a - a.mean()) / a.std()            # vectorised z-score, no loops
print("z-score mean ~0:", z.mean().round(6), "std ~1:", z.std().round(6))

# %% [markdown]
# ## 2. pandas - loading, inspecting, cleaning, grouping

# %%
data = load_breast_cancer(as_frame=True)
df = data.frame.copy()
df["diagnosis"] = df["target"].map({0: "malignant", 1: "benign"})
print("shape:", df.shape)
print("missing values:", int(df.isna().sum().sum()))
print("duplicates:", int(df.duplicated().sum()))
print(df["diagnosis"].value_counts())
print(df[["mean radius", "mean texture", "mean area"]].describe().round(2))

# %%
grouped = (df.groupby("diagnosis")[["mean radius", "mean texture", "mean area", "mean smoothness"]]
             .agg(["mean", "median"]).round(3))
print(grouped)

df["size_band"] = pd.qcut(df["mean radius"], 3, labels=["small", "medium", "large"])
pivot = pd.crosstab(df["size_band"], df["diagnosis"], normalize="index").round(3)
print(pivot)

# %% [markdown]
# ## 3. Matplotlib + Seaborn - visualisation

# %%
fig, ax = plt.subplots(figsize=(7, 4.5))
sns.histplot(data=df, x="mean radius", hue="diagnosis", kde=True, ax=ax)
ax.set_title("Distribution of Mean Radius by Diagnosis")
fig.tight_layout(); fig.savefig("figures/fig1_hist.png", dpi=150); plt.close(fig)

fig, ax = plt.subplots(figsize=(7, 4.5))
sns.boxplot(data=df, x="diagnosis", y="mean area", ax=ax)
ax.set_title("Mean Area by Diagnosis")
fig.tight_layout(); fig.savefig("figures/fig2_box.png", dpi=150); plt.close(fig)

cols = ["mean radius", "mean texture", "mean perimeter", "mean area",
        "mean smoothness", "mean compactness", "mean concavity", "mean symmetry"]
fig, ax = plt.subplots(figsize=(8, 6.5))
sns.heatmap(df[cols].corr(), annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
ax.set_title("Correlation Heatmap (selected features)")
fig.tight_layout(); fig.savefig("figures/fig3_heatmap.png", dpi=150); plt.close(fig)

fig, ax = plt.subplots(figsize=(7, 4.5))
sns.scatterplot(data=df, x="mean radius", y="mean concavity", hue="diagnosis", alpha=0.7, ax=ax)
ax.set_title("Mean Radius vs Mean Concavity")
fig.tight_layout(); fig.savefig("figures/fig4_scatter.png", dpi=150); plt.close(fig)
print("saved 4 figures")

# %% [markdown]
# ## 4. SciPy - hypothesis testing

# %%
mal = df.loc[df["diagnosis"] == "malignant", "mean radius"]
ben = df.loc[df["diagnosis"] == "benign", "mean radius"]
t, p = stats.ttest_ind(mal, ben, equal_var=False)
print(f"Welch t-test on mean radius: t = {t:.2f}, p = {p:.3e}")
r, p_r = stats.pearsonr(df["mean radius"], df["mean area"])
print(f"Pearson r (radius vs area) = {r:.3f}, p = {p_r:.2e}")

# %% [markdown]
# ## 5. scikit-learn - modelling pipeline

# %%
X = data.data
y = data.target
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)
print("train:", X_train.shape, "test:", X_test.shape)

models = {
    "Logistic Regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)),
    "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
}
rows = []
fitted = {}
for name, model in models.items():
    cv = cross_val_score(model, X_train, y_train, cv=5, scoring="accuracy")
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    fitted[name] = (model, pred)
    rows.append({"Model": name, "CV accuracy": cv.mean(),
                 "Test accuracy": accuracy_score(y_test, pred),
                 "Precision": precision_score(y_test, pred),
                 "Recall": recall_score(y_test, pred),
                 "F1": f1_score(y_test, pred)})
results = pd.DataFrame(rows).round(4)
print(results.to_string(index=False))
results.to_csv("model_results.csv", index=False)

# %%
best_name = results.sort_values("F1", ascending=False).iloc[0]["Model"]
best_model, best_pred = fitted[best_name]
print("Best model:", best_name)
print(classification_report(y_test, best_pred, target_names=data.target_names))
cm = confusion_matrix(y_test, best_pred)
fig, ax = plt.subplots(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
            xticklabels=data.target_names, yticklabels=data.target_names)
ax.set_xlabel("Predicted"); ax.set_ylabel("Actual"); ax.set_title(f"Confusion Matrix - {best_name}")
fig.tight_layout(); fig.savefig("figures/fig5_confusion.png", dpi=150); plt.close(fig)

rf = fitted["Random Forest"][0]
imp = pd.Series(rf.feature_importances_, index=data.feature_names).sort_values(ascending=False).head(10)
print(imp.round(4))
fig, ax = plt.subplots(figsize=(7, 4.5))
sns.barplot(x=imp.values, y=imp.index, ax=ax, color="#3b7dd8")
ax.set_title("Top 10 Feature Importances (Random Forest)")
fig.tight_layout(); fig.savefig("figures/fig6_importance.png", dpi=150); plt.close(fig)
