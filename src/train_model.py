"""
Machine Learning Model Training for Netflix Churn Prediction
------------------------------------------------------------
Trains and evaluates multiple models using processed data.
Excludes target-leaking features like 'status'.
"""

import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import joblib

# ---------------------------- 📁 Paths
PROCESSED_DIR = "data/processed"
REPORTS_DIR = "data/reports"
MODELS_DIR = "models"
os.makedirs(REPORTS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

# ---------------------------- 📥 Load Data
df = pd.read_csv(os.path.join(PROCESSED_DIR, "analysis_ready.csv"))
print(f"✅ Loaded dataset with {df.shape[0]} rows and {df.shape[1]} columns")

# ---------------------------- 🧹 Preprocessing
target = 'is_churned'

# Drop target-leaking features (like subscription status)
leak_features = ['status']  # Add any other features that directly reveal churn
X = df.drop(columns=[target] + [f for f in leak_features if f in df.columns])
y = df[target]

# Encode categorical features
cat_cols = X.select_dtypes(include=['object']).columns
for col in cat_cols:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col].astype(str))

# Handle missing values
X.fillna(0, inplace=True)

# Save the feature list for feature importance
features_used = X.columns.tolist()
joblib.dump(features_used, os.path.join(MODELS_DIR, "gradient_boosting_features.pkl"))

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Scale numeric features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ---------------------------- 🤖 Train Models
models = {
    "Logistic Regression": LogisticRegression(max_iter=500),
    "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(random_state=42)
}

results = []

for name, model in models.items():
    print(f"\n🚀 Training {name}...")
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else preds

    acc = accuracy_score(y_test, preds)
    prec = precision_score(y_test, preds, zero_division=0)
    rec = recall_score(y_test, preds, zero_division=0)
    f1 = f1_score(y_test, preds, zero_division=0)
    auc = roc_auc_score(y_test, probs)

    results.append({
        "model": name,
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "roc_auc": auc
    })

    # Save model
    model_file = os.path.join(MODELS_DIR, f"{name.replace(' ', '_').lower()}.pkl")
    joblib.dump(model, model_file)
    print(f"✅ {name} saved to {model_file}")

# ---------------------------- 📊 Save Results
results_df = pd.DataFrame(results).sort_values(by='roc_auc', ascending=False)
results_path = os.path.join(REPORTS_DIR, "model_performance.csv")
results_df.to_csv(results_path, index=False)

print(f"\n📈 Model Performance Summary:\n{results_df}")
print(f"\n✅ Results saved to {results_path}")
