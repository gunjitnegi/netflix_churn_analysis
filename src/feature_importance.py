import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# ---------------------------- Paths
PROCESSED_DIR = "data/processed"
MODELS_DIR = "models"
REPORTS_DIR = "data/reports"
os.makedirs(REPORTS_DIR, exist_ok=True)

MODEL_FILE = os.path.join(MODELS_DIR, "gradient_boosting_tuned.pkl")
FEATURES_FILE = os.path.join(MODELS_DIR, "gradient_boosting_features.pkl")
DATA_FILE = os.path.join(PROCESSED_DIR, "analysis_ready.csv")

# ---------------------------- Load model, features, dataset
model = joblib.load(MODEL_FILE)
features_used = joblib.load(FEATURES_FILE)  # this must be the list of features actually used in training
df = pd.read_csv(DATA_FILE)

print(f"✅ Loaded tuned model from: {MODEL_FILE}")
print(f"✅ Dataset shape: {df.shape}")

# ---------------------------- Align dataset with model features
missing_features = [f for f in features_used if f not in df.columns]
if missing_features:
    raise KeyError(f"❌ Missing required features in dataset: {missing_features}")

X = df[features_used]  # select only the columns the model was trained on

# ---------------------------- Extract feature importances
if hasattr(model, "feature_importances_"):
    importances = model.feature_importances_
else:
    # If model is a pipeline, try to find the GradientBoostingClassifier
    for step in model.named_steps.values():
        if hasattr(step, "feature_importances_"):
            importances = step.feature_importances_
            break
    else:
        raise AttributeError("❌ No estimator with feature_importances_ found in pipeline.")

if len(importances) != len(features_used):
    raise ValueError(f"❌ Feature importance length ({len(importances)}) does not match number of training features ({len(features_used)})")

# ---------------------------- Save feature importance
feat_importance_df = pd.DataFrame({
    "feature": features_used,
    "importance": importances
}).sort_values(by="importance", ascending=False)

csv_path = os.path.join(REPORTS_DIR, "feature_importance.csv")
feat_importance_df.to_csv(csv_path, index=False)
print(f"✅ Feature importance CSV saved: {csv_path}")

# ---------------------------- Plot
sns.set(style="whitegrid")
plt.figure(figsize=(10, 6))
sns.barplot(x="importance", y="feature", data=feat_importance_df, palette="viridis")
plt.title("Gradient Boosting Feature Importance")
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.tight_layout()
plt.savefig(os.path.join(REPORTS_DIR, "feature_importance.png"))
plt.close()
print(f"✅ Feature importance plot saved: {os.path.join(REPORTS_DIR, 'feature_importance.png')}")
