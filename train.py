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
                             ConfusionMatrixDisplay)

# ── 1. LOAD DATA ─────────────────────────────────────────────────────────────
df = pd.read_excel("data/E Commerce Dataset.xlsx", sheet_name="E Comm")
print(f"Loaded data: {df.shape[0]} rows, {df.shape[1]} columns")

# ── 2. DROP IDENTIFIER COLUMN ─────────────────────────────────────────────────
df.drop(columns=["CustomerID"], inplace=True)

# ── 3. HANDLE MISSING VALUES ──────────────────────────────────────────────────
for col in df.columns:
    if df[col].dtype == "object":
        df[col] = df[col].fillna(df[col].mode()[0])   # categorical → mode
    else:
        df[col] = df[col].fillna(df[col].median())    # numerical   → median

print(f"Missing values after imputation: {df.isnull().sum().sum()}")

# ── 3b. SAVE CLEANED DATASET ─────────────────────────────────────────────────
df.to_csv("data/cleaned_dataset.csv", index=False)
print("Cleaned dataset saved to data/cleaned_dataset.csv")
print(f"Cleaned dataset shape: {df.shape}")
print(f"Missing values after cleaning: {df.isnull().sum().sum()}")

# ── 4. ONE-HOT ENCODE CATEGORICAL COLUMNS ────────────────────────────────────
df = pd.get_dummies(df, drop_first=True)
print(f"Columns after encoding: {df.shape[1]}")

# ── 5. SEPARATE FEATURES AND TARGET ──────────────────────────────────────────
X = df.drop(columns=["Churn"])
y = df["Churn"]

# ── 6. TRAIN-TEST SPLIT ───────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"Train size: {X_train.shape[0]}, Test size: {X_test.shape[0]}")

# Save preprocessed splits for reference
X_train.to_csv("data/X_train.csv", index=False)
X_test.to_csv("data/X_test.csv", index=False)
y_train.to_csv("data/y_train.csv", index=False)
y_test.to_csv("data/y_test.csv", index=False)

# ── 7. TRAIN MODELS ───────────────────────────────────────────────────────────
models = {
    "Logistic Regression": LogisticRegression(max_iter=5000, random_state=42),
    "Random Forest":       RandomForestClassifier(n_estimators=100, random_state=42)
}

for name, model in models.items():
    model.fit(X_train, y_train)

# ── 8. EVALUATE MODELS ───────────────────────────────────────────────────────
print("\n========== MODEL EVALUATION ==========")
for name, model in models.items():
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    print(f"\n--- {name} ---")
    print(f"  Accuracy  : {accuracy_score(y_test, y_pred):.4f}")
    print(f"  Precision : {precision_score(y_test, y_pred):.4f}")
    print(f"  Recall    : {recall_score(y_test, y_pred):.4f}")
    print(f"  F1-Score  : {f1_score(y_test, y_pred):.4f}")
    print(f"  ROC-AUC   : {roc_auc_score(y_test, y_prob):.4f}")

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["No Churn", "Churn"])
    disp.plot(cmap="Blues")
    plt.title(f"Confusion Matrix — {name}")
    plt.tight_layout()
    plt.savefig(f"models/confusion_matrix_{name.replace(' ', '_')}.png")
    plt.close()
    print(f"  Confusion matrix saved.")

# ── 9. SAVE MODELS ────────────────────────────────────────────────────────────
joblib.dump(models["Logistic Regression"], "models/logistic_regression.pkl")
joblib.dump(models["Random Forest"],       "models/random_forest.pkl")

# Save feature column names (needed by Streamlit app)
joblib.dump(list(X.columns), "models/feature_columns.pkl")

print("\nModels saved to models/")
print("Training complete!")
