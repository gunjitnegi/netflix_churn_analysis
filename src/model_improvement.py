"""
Model Improvement Script for Netflix Churn Prediction
------------------------------------------------------
Balances dataset using SMOTE, tunes Gradient Boosting via GridSearchCV,
evaluates improved model, and saves tuned model and feature list.
"""

import pandas as pd
import os
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report, roc_auc_score
from imblearn.over_sampling import SMOTE
import joblib

# ---------------------------- 📁 Define directories
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
MODEL_DIR = os.path.join(BASE_DIR, "models")
REPORTS_DIR = os.path.join(BASE_DIR, "data", "reports")
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)

print(">>\n🚀 Starting Model Improvement Phase...\n")

# ---------------------------- 📥 Load dataset
df = pd.read_csv(os.path.join(PROCESSED_DIR, "analysis_ready.csv"))

# ---------------------------- 🛠 Select features for model
features_used = [
    'total_watch_time', 'avg_watch_time_per_session',
    'days_since_last_activity', 'total_payments',
    'failed_payment_ratio', 'support_ticket_count',
    'subscription_duration_days', 'subscription_duration_months',
    'payment_per_month', 'watch_time_per_day'
]

X = df[features_used]
y = df['is_churned']

# ---------------------------- ⚖️ Balance classes with SMOTE
print("🔄 Applying SMOTE to balance classes...")
smote = SMOTE(random_state=42)
X_res, y_res = smote.fit_resample(X, y)
print(f"✅ After SMOTE: {y_res.value_counts().to_dict()}")

# ---------------------------- ✂️ Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.2, random_state=42)

# ---------------------------- 🔍 GridSearchCV for Gradient Boosting
print("⚙️ Starting hyperparameter tuning (this may take a few minutes)...")

param_grid = {
    'n_estimators': [100, 150],
    'learning_rate': [0.05, 0.1],
    'max_depth': [3, 4],
    'subsample': [0.8, 1.0]
}

gb = GradientBoostingClassifier(random_state=42)
grid_search = GridSearchCV(
    gb, param_grid, cv=3, scoring='roc_auc', n_jobs=-1, verbose=1
)
grid_search.fit(X_train, y_train)

print(f"\n✅ Best Parameters: {grid_search.best_params_}")
best_model = grid_search.best_estimator_

# ---------------------------- 📈 Evaluate improved model
y_pred = best_model.predict(X_test)
y_proba = best_model.predict_proba(X_test)[:, 1]

roc = roc_auc_score(y_test, y_proba)
print("\n📊 Classification Report:\n", classification_report(y_test, y_pred))
print(f"ROC-AUC: {roc:.4f}")

# ---------------------------- 💾 Save improved model and feature list
model_path = os.path.join(MODEL_DIR, "gradient_boosting_tuned.pkl")
features_path = os.path.join(MODEL_DIR, "gradient_boosting_features.pkl")
joblib.dump(best_model, model_path)
joblib.dump(features_used, features_path)

print(f"✅ Improved Gradient Boosting model saved at: {model_path}")
print(f"✅ Feature list saved at: {features_path}")
