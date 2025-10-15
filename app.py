import streamlit as st
import pandas as pd
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns

# ----------------------------
# 📁 Paths
# ----------------------------
MODEL_PATH = "models/gradient_boosting_tuned.pkl"
FEATURES_PATH = "models/gradient_boosting_features.pkl"
FEATURE_IMPORTANCE_PATH = "data/reports/feature_importance.csv"

# ----------------------------
# 📥 Load model & features
# ----------------------------
if not os.path.exists(MODEL_PATH) or not os.path.exists(FEATURES_PATH):
    st.error("Model or features file not found. Train the model first!")
    st.stop()

model = joblib.load(MODEL_PATH)
features_used = joblib.load(FEATURES_PATH)

st.set_page_config(page_title="Netflix Churn Dashboard", layout="wide")
st.title("📊 Netflix Churn Prediction Dashboard")

# ----------------------------
# 1️⃣ Upload New User Data
# ----------------------------
uploaded_file = st.file_uploader("Upload new user data (CSV)", type="csv")
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write(f"Uploaded dataset shape: {df.shape}")

    # Check missing features
    missing_features = [f for f in features_used if f not in df.columns]
    if missing_features:
        st.error(f"❌ Missing required features: {missing_features}")
    else:
        X = df[features_used]
        preds = model.predict(X)
        probs = model.predict_proba(X)[:, 1]

        df["churn_prediction"] = preds
        df["churn_probability"] = probs

        st.success("✅ Predictions completed!")
        st.subheader("Top 10 Predictions")
        st.dataframe(df.head(10))

        # ----------------------------
        # 2️⃣ Download Predictions
        # ----------------------------
        csv = df.to_csv(index=False).encode()
        st.download_button(
            label="Download Predictions CSV",
            data=csv,
            file_name="new_users_predictions.csv",
            mime="text/csv"
        )

        # ----------------------------
        # 3️⃣ Feature Importance
        # ----------------------------
        if os.path.exists(FEATURE_IMPORTANCE_PATH):
            feat_df = pd.read_csv(FEATURE_IMPORTANCE_PATH)
            st.subheader("Feature Importance")
            plt.figure(figsize=(8, 6))
            sns.barplot(x="importance", y="feature", data=feat_df.sort_values(by="importance", ascending=False), palette="viridis")
            plt.xlabel("Importance")
            plt.ylabel("Feature")
            plt.title("Gradient Boosting Feature Importance")
            st.pyplot(plt)
            plt.clf()
        else:
            st.warning("Feature importance CSV not found.")

        # ----------------------------
        # 4️⃣ Churn Probability Distribution
        # ----------------------------
        st.subheader("Churn Probability Distribution")
        plt.figure(figsize=(8, 4))
        sns.histplot(df["churn_probability"], bins=20, kde=True, color="skyblue")
        plt.xlabel("Churn Probability")
        plt.ylabel("Count")
        st.pyplot(plt)
        plt.clf()

        # ----------------------------
        # 5️⃣ Churned vs Active Users Pie Chart
        # ----------------------------
        st.subheader("Churned vs Active Users")
        counts = df["churn_prediction"].value_counts().rename({0: "Active", 1: "Churned"})
        plt.figure(figsize=(5, 5))
        plt.pie(counts, labels=counts.index, autopct="%1.1f%%", colors=["#4CAF50", "#F44336"])
        plt.title("Churned vs Active Users")
        st.pyplot(plt)
        plt.clf()
