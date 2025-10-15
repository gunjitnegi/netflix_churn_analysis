"""
Netflix Churn Prediction for New Users
--------------------------------------
Loads the trained Gradient Boosting model, predicts churn for new users,
and optionally evaluates model performance if true labels exist.
"""

import pandas as pd
import os
import joblib
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# ---------------------------- 📁 Paths
PROCESSED_DIR = "data/processed"
MODELS_DIR = "models"
REPORTS_DIR = "data/reports"
NEW_DATA_FILE = "data/new_users.csv"

os.makedirs(REPORTS_DIR, exist_ok=True)

MODEL_FILE = os.path.join(MODELS_DIR, "gradient_boosting_tuned.pkl")
FEATURES_FILE = os.path.join(MODELS_DIR, "gradient_boosting_features.pkl")

# ---------------------------- 📥 Load model and features
if not os.path.exists(MODEL_FILE):
    raise FileNotFoundError(f"❌ Model not found: {MODEL_FILE}. Train models first!")
if not os.path.exists(FEATURES_FILE):
    raise FileNotFoundError(f"❌ Feature list not found: {FEATURES_FILE}. Retrain model and save features!")
if not os.path.exists(NEW_DATA_FILE):
    raise FileNotFoundError(f"❌ New data not found: {NEW_DATA_FILE}")

model = joblib.load(MODEL_FILE)
features_used = joblib.load(FEATURES_FILE)
new_df = pd.read_csv(NEW_DATA_FILE)

print(f"✅ Loaded model from: {MODEL_FILE}")
print(f"✅ New data shape: {new_df.shape}")

# ---------------------------- 🛠 Ensure required features
# Fill derived features if missing
derived_features = ['subscription_duration_months', 'payment_per_month', 'watch_time_per_day']
for f in derived_features:
    if f not in new_df.columns:
        if f == 'subscription_duration_months':
            new_df[f] = new_df.get('subscription_duration_days', 0) / 30
        elif f == 'payment_per_month':
            new_df[f] = new_df.get('total_payments', 0) / new_df.get('subscription_duration_months', 1)
        elif f == 'watch_time_per_day':
            new_df[f] = new_df.get('total_watch_time', 0) / new_df.get('subscription_duration_days', 1)

# Check for missing columns
missing_features = [f for f in features_used if f not in new_df.columns]
if missing_features:
    raise KeyError(f"❌ Missing required features in new data: {missing_features}")

X_new = new_df[features_used]

# ---------------------------- 🔮 Predict churn
preds = model.predict(X_new)
probs = model.predict_proba(X_new)[:, 1] if hasattr(model, "predict_proba") else None

new_df['predicted_churn'] = preds
if probs is not None:
    new_df['predicted_churn_proba'] = probs

# Save predictions
predictions_file = os.path.join(REPORTS_DIR, "new_users_predictions.csv")
new_df.to_csv(predictions_file, index=False)
print(f"✅ Predictions saved to: {predictions_file}")

# ---------------------------- 📊 Optional: Evaluate if true labels exist
if 'is_churned' in new_df.columns:
    y_true = new_df['is_churned']
    y_pred = new_df['predicted_churn']
    y_prob = new_df.get('predicted_churn_proba', None)

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    roc = roc_auc_score(y_true, y_prob) if y_prob is not None else None

    metrics = {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1_score": f1,
        "roc_auc": roc
    }
    metrics_file = os.path.join(REPORTS_DIR, "new_users_prediction_metrics.csv")
    pd.DataFrame([metrics]).to_csv(metrics_file, index=False)
    print(f"✅ Performance metrics saved to: {metrics_file}")
    print(f"\n📊 Performance:\n{metrics}")
else:
    print("⚠️ No actual labels in new data. Only predictions saved.")
