"""
Feature Engineering Script for Netflix Churn Project
----------------------------------------------------
Merges cleaned datasets, computes key behavioral features,
calculates subscription duration and churn label, and saves
analysis-ready dataset.
"""

import pandas as pd
import os
import numpy as np

# ----------------------------
# 📁 Define directory structure
# ----------------------------
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
INTERIM_DIR = os.path.join(BASE_DIR, "data", "interim")
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
os.makedirs(PROCESSED_DIR, exist_ok=True)

print(">>\n🚀 Starting Feature Engineering Phase...\n")

# ----------------------------
# 📥 Load cleaned datasets
# ----------------------------
users = pd.read_csv(os.path.join(INTERIM_DIR, "users_cleaned.csv"))
subscriptions = pd.read_csv(os.path.join(INTERIM_DIR, "subscriptions_cleaned.csv"))
payments = pd.read_csv(os.path.join(INTERIM_DIR, "payments_cleaned.csv"))
viewing = pd.read_csv(os.path.join(INTERIM_DIR, "viewing_events_cleaned.csv"))
tickets = pd.read_csv(os.path.join(INTERIM_DIR, "support_tickets_cleaned.csv"))

# ----------------------------
# 🕒 Handle date columns safely
# ----------------------------
# Payments
if 'date' in payments.columns:
    payments['payment_date'] = pd.to_datetime(payments['date'], errors='coerce')
elif 'payment_date' in payments.columns:
    payments['payment_date'] = pd.to_datetime(payments['payment_date'], errors='coerce')
else:
    raise KeyError("❌ No valid date column found in payments.csv (expected 'date' or 'payment_date').")

# Viewing timestamps
viewing['timestamp'] = pd.to_datetime(viewing['timestamp'], errors='coerce', infer_datetime_format=True)

# Subscriptions: compute duration and churn
subscriptions['start_date'] = pd.to_datetime(subscriptions['start_date'], errors='coerce')
subscriptions['end_date'] = pd.to_datetime(subscriptions['end_date'], errors='coerce')
subscriptions['end_date'] = subscriptions['end_date'].fillna(pd.Timestamp.today())
subscriptions['subscription_duration_days'] = (subscriptions['end_date'] - subscriptions['start_date']).dt.days

# Churn: 1 if subscription canceled, 0 if still active
if 'status' in subscriptions.columns:
    subscriptions['is_churned'] = (subscriptions['status'] == 'Canceled').astype(int)
else:
    subscriptions['is_churned'] = (subscriptions['end_date'] < pd.Timestamp.today()).astype(int)

# ----------------------------
# 💰 Aggregate payment features
# ----------------------------
payment_features = payments.groupby('user_id').agg(
    total_payments=('amount', 'sum'),
    failed_payment_ratio=('status', lambda x: (x == 'Failed').mean())
).reset_index()

# ----------------------------
# 🎬 Aggregate viewing activity
# ----------------------------
viewing_features = viewing.groupby('user_id').agg(
    total_watch_time=('duration_watched_seconds', 'sum'),
    avg_watch_time_per_session=('duration_watched_seconds', 'mean'),
    days_since_last_activity=('timestamp', lambda x: (pd.Timestamp.today() - x.max()).days)
).reset_index()

# ----------------------------
# 🎟️ Aggregate support tickets
# ----------------------------
ticket_features = tickets.groupby('user_id').size().reset_index(name='support_ticket_count')

# ----------------------------
# 🔗 Merge all features together
# ----------------------------
df = subscriptions.merge(
    users[['user_id', 'country', 'region', 'signup_channel', 'age_group']],  # exclude plan_type from users
    on='user_id', how='left'
)

df = df.merge(payment_features, on='user_id', how='left')
df = df.merge(viewing_features, on='user_id', how='left')
df = df.merge(ticket_features, on='user_id', how='left')

# Rename subscription plan_type if exists
if 'plan_type' not in df.columns and 'plan_type' in subscriptions.columns:
    df['plan_type'] = subscriptions['plan_type']

# ----------------------------
# 🧹 Handle missing values
# ----------------------------
# Categorical columns
for col in ['plan_type', 'country', 'region', 'signup_channel', 'age_group']:
    if col in df.columns:
        df[col] = df[col].fillna('Unknown')

# Numeric columns
numeric_cols = [
    'total_watch_time', 'avg_watch_time_per_session',
    'days_since_last_activity', 'total_payments',
    'failed_payment_ratio', 'support_ticket_count',
    'subscription_duration_days'
]

for col in numeric_cols:
    if col in df.columns:
        df[col] = df[col].fillna(df[col].median())

# Derived features
if 'subscription_duration_days' in df.columns:
    df['subscription_duration_months'] = df['subscription_duration_days'] / 30
    df['payment_per_month'] = df['total_payments'] / df['subscription_duration_months'].replace(0, np.nan)
    df['watch_time_per_day'] = df['total_watch_time'] / df['subscription_duration_days'].replace(0, np.nan)

# Fill remaining NaNs
df.fillna(0, inplace=True)

# ----------------------------
# 💾 Save processed dataset
# ----------------------------
output_path = os.path.join(PROCESSED_DIR, "analysis_ready.csv")
df.to_csv(output_path, index=False)

print(f"✅ Analysis-ready dataset saved successfully at: {output_path}")
