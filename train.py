import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix,
                             ConfusionMatrixDisplay, roc_curve)
from xgboost import XGBClassifier
from lifelines import KaplanMeierFitter

os.makedirs("models",  exist_ok=True)
os.makedirs("outputs", exist_ok=True)

# ── 1. LOAD DATA ──────────────────────────────────────────────────────────────
df = pd.read_excel("data/E Commerce Dataset.xlsx", sheet_name="E Comm")
print(f"Loaded data: {df.shape[0]} rows, {df.shape[1]} columns")

# ── 2. DROP IDENTIFIER COLUMN ─────────────────────────────────────────────────
df.drop(columns=["CustomerID"], inplace=True)

# ── 3. HANDLE MISSING VALUES ──────────────────────────────────────────────────
for col in df.columns:
    if df[col].dtype == "object":
        df[col] = df[col].fillna(df[col].mode()[0])
    else:
        df[col] = df[col].fillna(df[col].median())

print(f"Missing values after imputation: {df.isnull().sum().sum()}")

# ── 3b. SAVE CLEANED DATASET ──────────────────────────────────────────────────
df.to_csv("data/cleaned_dataset.csv", index=False)
print("Cleaned dataset saved to data/cleaned_dataset.csv")
print(f"Cleaned dataset shape: {df.shape}")
print(f"Missing values after cleaning: {df.isnull().sum().sum()}")

# ── 3c. PROXY LTV CALCULATION ─────────────────────────────────────────────────
# Formula: Proxy LTV = CashbackAmount × OrderCount × (1 + Tenure / 50)
# NOTE: This is an estimated proxy only. The dataset contains no Revenue,
#       TotalSpend, or PurchaseValue column. This formula uses available
#       engagement signals as a reasonable approximation.
df["ProxyLTV"] = df["CashbackAmount"] * df["OrderCount"] * (1 + df["Tenure"] / 50)
print(f"\nProxy LTV — min: {df['ProxyLTV'].min():.2f}, "
      f"max: {df['ProxyLTV'].max():.2f}, "
      f"mean: {df['ProxyLTV'].mean():.2f}")

# ── 4. ONE-HOT ENCODE CATEGORICAL COLUMNS ────────────────────────────────────
df_encoded = pd.get_dummies(df.drop(columns=["ProxyLTV"]), drop_first=True)
print(f"Columns after encoding: {df_encoded.shape[1]}")

# ── 5. SEPARATE FEATURES AND TARGET ──────────────────────────────────────────
X = df_encoded.drop(columns=["Churn"])
y = df_encoded["Churn"]

# ── 6. TRAIN-TEST SPLIT ───────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"Train size: {X_train.shape[0]}, Test size: {X_test.shape[0]}")

X_train.to_csv("data/X_train.csv", index=False)
X_test.to_csv("data/X_test.csv",   index=False)
y_train.to_csv("data/y_train.csv", index=False)
y_test.to_csv("data/y_test.csv",   index=False)

# ── 7. TRAIN MODELS ───────────────────────────────────────────────────────────
models = {
    "Logistic Regression": LogisticRegression(max_iter=5000, random_state=42),
    "Random Forest":       RandomForestClassifier(n_estimators=100, random_state=42),
    "XGBoost":             XGBClassifier(n_estimators=100, random_state=42,
                                         use_label_encoder=False,
                                         eval_metric="logloss",
                                         verbosity=0),
}

for name, model in models.items():
    model.fit(X_train, y_train)
    print(f"Trained: {name}")

# ── 8. EVALUATE MODELS ───────────────────────────────────────────────────────
print("\n========== MODEL EVALUATION ==========")
results = []
for name, model in models.items():
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    acc  = round(accuracy_score(y_test,  y_pred), 4)
    prec = round(precision_score(y_test, y_pred), 4)
    rec  = round(recall_score(y_test,    y_pred), 4)
    f1   = round(f1_score(y_test,        y_pred), 4)
    auc  = round(roc_auc_score(y_test,   y_prob), 4)

    print(f"\n--- {name} ---")
    print(f"  Accuracy  : {acc}")
    print(f"  Precision : {prec}")
    print(f"  Recall    : {rec}")
    print(f"  F1-Score  : {f1}")
    print(f"  ROC-AUC   : {auc}")

    results.append({"Model": name, "Accuracy": acc, "Precision": prec,
                    "Recall": rec, "F1-Score": f1, "ROC-AUC": auc})

    # Confusion matrix
    cm   = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                  display_labels=["No Churn", "Churn"])
    disp.plot(cmap="Blues")
    plt.title(f"Confusion Matrix — {name}")
    plt.tight_layout()
    plt.savefig(f"models/confusion_matrix_{name.replace(' ', '_')}.png", dpi=150)
    plt.close()
    print(f"  Confusion matrix saved.")

results_df = pd.DataFrame(results)

# ── 9. SAVE MODELS ────────────────────────────────────────────────────────────
joblib.dump(models["Logistic Regression"], "models/logistic_regression.pkl")
joblib.dump(models["Random Forest"],       "models/random_forest.pkl")
joblib.dump(models["XGBoost"],             "models/xgboost.pkl")
joblib.dump(list(X.columns),              "models/feature_columns.pkl")
print("\nAll models saved to models/")

# ── 10. CHURN DISTRIBUTION ────────────────────────────────────────────────────
churn_counts = y.value_counts().sort_index()
fig, ax = plt.subplots(figsize=(7, 5))
bars = ax.bar(["No Churn (0)", "Churn (1)"], churn_counts.values,
              color=["steelblue", "tomato"], edgecolor="white", width=0.5)
for bar, count in zip(bars, churn_counts.values):
    pct = count / len(y) * 100
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 30,
            f"{count}\n({pct:.1f}%)", ha="center", va="bottom", fontsize=11)
ax.set_title("Customer Churn Distribution", fontsize=14, fontweight="bold", pad=15)
ax.set_xlabel("Churn Status", fontsize=12)
ax.set_ylabel("Number of Customers", fontsize=12)
ax.set_ylim(0, churn_counts.max() * 1.2)
sns.despine()
plt.tight_layout()
plt.savefig("outputs/churn_distribution.png", dpi=150)
plt.close()
print("Saved: outputs/churn_distribution.png")

# ── 11. MODEL PERFORMANCE COMPARISON (3 models) ───────────────────────────────
metrics   = ["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"]
x         = np.arange(len(metrics))
bar_width = 0.25
bar_colors = ["steelblue", "tomato", "seagreen"]

fig, ax = plt.subplots(figsize=(12, 6))
for i, row in results_df.iterrows():
    vals = [row[m] for m in metrics]
    bars = ax.bar(x + i * bar_width, vals, bar_width,
                  label=row["Model"], color=bar_colors[i], edgecolor="white")
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f"{val:.2f}", ha="center", va="bottom", fontsize=7)

ax.set_title("Model Performance Comparison", fontsize=14, fontweight="bold", pad=15)
ax.set_xlabel("Metric", fontsize=12)
ax.set_ylabel("Score", fontsize=12)
ax.set_xticks(x + bar_width)
ax.set_xticklabels(metrics, fontsize=11)
ax.set_ylim(0, 1.18)
ax.legend(fontsize=11)
sns.despine()
plt.tight_layout()
plt.savefig("outputs/model_performance_comparison.png", dpi=150)
plt.close()
print("Saved: outputs/model_performance_comparison.png")

# ── 12. RANDOM FOREST FEATURE IMPORTANCE ─────────────────────────────────────
rf_imp  = models["Random Forest"].feature_importances_
feat_rf = (pd.DataFrame({"Feature": X.columns, "Importance": rf_imp})
           .sort_values("Importance", ascending=False).head(10))

fig, ax = plt.subplots(figsize=(9, 6))
sns.barplot(x="Importance", y="Feature", data=feat_rf, palette="Blues_r", ax=ax, hue="Feature", legend=False)
ax.set_title("Random Forest — Top 10 Feature Importances",
             fontsize=14, fontweight="bold", pad=15)
ax.set_xlabel("Importance Score", fontsize=12)
ax.set_ylabel("Feature", fontsize=12)
sns.despine()
plt.tight_layout()
plt.savefig("outputs/random_forest_feature_importance.png", dpi=150)
plt.close()
print("Saved: outputs/random_forest_feature_importance.png")

# ── 13. XGBOOST FEATURE IMPORTANCE ───────────────────────────────────────────
xgb_imp  = models["XGBoost"].feature_importances_
feat_xgb = (pd.DataFrame({"Feature": X.columns, "Importance": xgb_imp})
            .sort_values("Importance", ascending=False).head(10))

fig, ax = plt.subplots(figsize=(9, 6))
sns.barplot(x="Importance", y="Feature", data=feat_xgb, palette="Greens_r", ax=ax, hue="Feature", legend=False)
ax.set_title("XGBoost — Top 10 Feature Importances",
             fontsize=14, fontweight="bold", pad=15)
ax.set_xlabel("Importance Score", fontsize=12)
ax.set_ylabel("Feature", fontsize=12)
sns.despine()
plt.tight_layout()
plt.savefig("outputs/xgboost_feature_importance.png", dpi=150)
plt.close()
print("Saved: outputs/xgboost_feature_importance.png")

# ── 14. ROC CURVE (3 models) ──────────────────────────────────────────────────
roc_colors = {"Logistic Regression": "steelblue",
              "Random Forest":       "tomato",
              "XGBoost":             "seagreen"}

fig, ax = plt.subplots(figsize=(8, 6))
for name, model in models.items():
    y_prob      = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    auc_val     = roc_auc_score(y_test, y_prob)
    ax.plot(fpr, tpr, label=f"{name} (AUC = {auc_val:.4f})",
            color=roc_colors[name], linewidth=2)

ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Random Classifier")
ax.set_title("ROC Curve — Model Comparison", fontsize=14, fontweight="bold", pad=15)
ax.set_xlabel("False Positive Rate", fontsize=12)
ax.set_ylabel("True Positive Rate", fontsize=12)
ax.legend(fontsize=10)
sns.despine()
plt.tight_layout()
plt.savefig("outputs/roc_curve.png", dpi=150)
plt.close()
print("Saved: outputs/roc_curve.png")

# ── 15. KAPLAN-MEIER SURVIVAL ANALYSIS ───────────────────────────────────────
# Tenure = duration on platform (months), Churn = event (1 = churned)
kmf = KaplanMeierFitter()

fig, ax = plt.subplots(figsize=(10, 6))

# Overall survival curve
kmf.fit(durations=df["Tenure"], event_observed=df["Churn"], label="All Customers")
kmf.plot_survival_function(ax=ax, ci_show=True, color="steelblue")

# Survival by Complain group
for complain_val, color, label in [(0, "seagreen", "No Complaint"),
                                    (1, "tomato",   "Complained")]:
    mask = df["Complain"] == complain_val
    kmf.fit(durations=df.loc[mask, "Tenure"],
            event_observed=df.loc[mask, "Churn"],
            label=label)
    kmf.plot_survival_function(ax=ax, ci_show=False, color=color)

ax.set_title("Kaplan-Meier Survival Curve — Customer Retention",
             fontsize=14, fontweight="bold", pad=15)
ax.set_xlabel("Tenure (Months on Platform)", fontsize=12)
ax.set_ylabel("Survival Probability (Retention Rate)", fontsize=12)
ax.legend(fontsize=11)
ax.set_ylim(0, 1.05)
sns.despine()
plt.tight_layout()
plt.savefig("outputs/kaplan_meier_survival_curve.png", dpi=150)
plt.close()
print("Saved: outputs/kaplan_meier_survival_curve.png")

# ── 16. PROXY LTV DISTRIBUTION ───────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Histogram of Proxy LTV
axes[0].hist(df["ProxyLTV"], bins=40, color="steelblue", edgecolor="white")
axes[0].set_title("Estimated Proxy LTV Distribution",
                  fontsize=13, fontweight="bold")
axes[0].set_xlabel("Estimated Proxy LTV", fontsize=11)
axes[0].set_ylabel("Number of Customers", fontsize=11)
sns.despine(ax=axes[0])

# Proxy LTV by Churn status (box plot)
churn_labels = df["Churn"].map({0: "No Churn", 1: "Churned"})
axes[1].boxplot(
    [df.loc[df["Churn"] == 0, "ProxyLTV"], df.loc[df["Churn"] == 1, "ProxyLTV"]],
    tick_labels=["No Churn", "Churned"],
    patch_artist=True,
    boxprops=dict(facecolor="steelblue", color="navy"),
    medianprops=dict(color="white", linewidth=2),
)
axes[1].set_title("Estimated Proxy LTV by Churn Status",
                  fontsize=13, fontweight="bold")
axes[1].set_xlabel("Churn Status", fontsize=11)
axes[1].set_ylabel("Estimated Proxy LTV", fontsize=11)
sns.despine(ax=axes[1])

plt.suptitle(
    "Note: Proxy LTV = CashbackAmount × OrderCount × (1 + Tenure/50)\n"
    "No revenue/spend data available in dataset.",
    fontsize=9, color="gray", y=0.01
)
plt.tight_layout()
plt.savefig("outputs/ltv_distribution.png", dpi=150)
plt.close()
print("Saved: outputs/ltv_distribution.png")

# ── 17. SAVE EVALUATION RESULTS CSV ──────────────────────────────────────────
results_df.to_csv("outputs/model_results.csv", index=False)
print("Saved: outputs/model_results.csv")

print("\n========== TRAINING COMPLETE ==========")
print("Models  -> models/")
print("Outputs -> outputs/")
