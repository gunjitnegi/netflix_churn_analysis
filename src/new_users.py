"""
Generate synthetic new users data for Netflix Churn Prediction
--------------------------------------------------------------
Creates a sample of 1000 new users with all features required for
prediction, including derived features like payment per month,
watch time per day, and subscription duration in months.
"""

import pandas as pd
import numpy as np
import os

# ----------------------------
# 📁 Paths
# ----------------------------
NEW_DATA_DIR = "data"
os.makedirs(NEW_DATA_DIR, exist_ok=True)
NEW_DATA_FILE = os.path.join(NEW_DATA_DIR, "new_users.csv")

# ----------------------------
# 🔢 Generate synthetic user data
# ----------------------------
np.random.seed(42)
n_users = 1000

df = pd.DataFrame({
    "user_id": range(100001, 100001 + n_users),
    "signup_date": pd.date_range(start="2025-01-01", periods=n_users, freq="D"),
    "country": np.random.choice(["USA", "UK", "India", "Canada", "Germany"], size=n_users),
    "region": np.random.choice(["North", "South", "East", "West"], size=n_users),
    "plan_type": np.random.choice(["Basic", "Standard", "Premium"], size=n_users),
    "signup_channel": np.random.choice(["Web", "Mobile", "Tablet"], size=n_users),
    "age_group": np.random.choice(["18-25", "26-35", "36-45", "46-55", "55+"], size=n_users),
    "subscription_duration_days": np.random.randint(30, 365, size=n_users),
    "total_payments": np.round(np.random.uniform(10, 500, size=n_users), 2),
    "total_watch_time": np.random.randint(1000, 100000, size=n_users),  # in seconds
    "avg_watch_time_per_session": np.random.randint(100, 2000, size=n_users),  # in seconds
    "days_since_last_activity": np.random.randint(0, 60, size=n_users),
    "support_ticket_count": np.random.randint(0, 10, size=n_users),
    "failed_payment_ratio": np.round(np.random.uniform(0, 0.5, size=n_users), 2)
})

# ----------------------------
# 🛠 Derived features
# ----------------------------
df['subscription_duration_months'] = df['subscription_duration_days'] / 30
df['payment_per_month'] = df['total_payments'] / df['subscription_duration_months']
df['watch_time_per_day'] = df['total_watch_time'] / df['subscription_duration_days']

# ----------------------------
# 💾 Save CSV
# ----------------------------
df.to_csv(NEW_DATA_FILE, index=False)
print(f"✅ Synthetic new users data saved to: {NEW_DATA_FILE}")
print(f"✅ Dataset shape: {df.shape}")
