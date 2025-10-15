import numpy as np
import pandas as pd
from faker import Faker
import random
import os

fake = Faker()
np.random.seed(42)
random.seed(42)

# Paths
RAW_DIR = "data/raw"
os.makedirs(RAW_DIR, exist_ok=True)

# Number of users
N_USERS = 50000

# ------------------------
# 1️⃣ Generate Users
# ------------------------
def generate_users(n=N_USERS):
    print("Generating users.csv ...")

    # Realistic countries manually (exclude Antarctica)
    valid_countries = [
        "United States", "Canada", "United Kingdom", "Germany", "France", "Italy",
        "Spain", "Australia", "Brazil", "India", "Japan", "China", "South Korea",
        "Mexico", "Russia", "Netherlands", "Sweden", "Norway", "South Africa", "Egypt"
    ]

    users = pd.DataFrame({
        "user_id": range(1, n + 1),
        "signup_date": [fake.date_between(start_date='-2y', end_date='today') for _ in range(n)],
        "country": np.random.choice(valid_countries, n),
        "region": np.random.choice(["North America", "Europe", "Asia", "South America", "Africa"], n),
        "plan_type": np.random.choice(["Basic", "Standard", "Premium"], n, p=[0.4, 0.4, 0.2]),
        "signup_channel": np.random.choice(["Web", "Mobile App", "Smart TV"], n, p=[0.5, 0.4, 0.1]),
        "age_group": np.random.choice(["18-25", "26-35", "36-45", "46-60", "60+"], n)
    })

    # Inject missing regions (~2%)
    users.loc[np.random.choice(users.index, size=int(0.02*n), replace=False), 'region'] = np.nan
    users.to_csv(f"{RAW_DIR}/users.csv", index=False)
    return users

# ------------------------
# 2️⃣ Generate Subscriptions
# ------------------------
def generate_subscriptions(users):
    print("Generating subscriptions.csv ...")
    subs = pd.DataFrame({
        "user_id": users["user_id"],
        "plan_type": users["plan_type"],
        "start_date": users["signup_date"],
        "end_date": [fake.date_between(start_date='+30d', end_date='+400d') for _ in range(len(users))],
        "status": np.random.choice(["Active", "Canceled", "Expired"], len(users), p=[0.75, 0.15, 0.1])
    })
    subs.to_csv(f"{RAW_DIR}/subscriptions.csv", index=False)
    return subs

# ------------------------
# 3️⃣ Generate Payments
# ------------------------
def generate_payments(users):
    print("Generating payments.csv ...")
    payments = []
    for uid in users["user_id"]:
        for i in range(np.random.randint(8, 14)):
            payments.append({
                "user_id": uid,
                "payment_date": fake.date_between(start_date='-1y', end_date='today'),
                "amount": round(np.random.choice([9.99, 15.99, 19.99], p=[0.4, 0.4, 0.2]), 2),
                "status": np.random.choice(["Success", "Failed"], p=[0.92, 0.08]),
                "failure_reason": np.random.choice(["Card Declined", "Insufficient Funds", None], p=[0.4, 0.3, 0.3]),
                "retry_count": np.random.randint(0, 3)
            })
    payments = pd.DataFrame(payments)
    payments.to_csv(f"{RAW_DIR}/payments.csv", index=False)
    return payments

# ------------------------
# 4️⃣ Generate Viewing Events
# ------------------------
def generate_viewing_events(users):
    print("Generating viewing_events.csv ...")
    events = []
    for uid in users["user_id"]:
        for i in range(np.random.randint(5, 20)):
            events.append({
                "user_id": uid,
                "timestamp": fake.date_time_between(start_date='-1y', end_date='now'),
                "duration_watched_seconds": abs(np.random.normal(1800, 1200))
            })

    df = pd.DataFrame(events)
    # Inject small fraction of invalid durations
    invalid_idx = np.random.choice(df.index, size=int(0.01 * len(df)), replace=False)
    df.loc[invalid_idx, "duration_watched_seconds"] = -1

    df.to_csv(f"{RAW_DIR}/viewing_events.csv", index=False)
    return df

# ------------------------
# 5️⃣ Generate Support Tickets
# ------------------------
def generate_support_tickets(users):
    print("Generating support_tickets.csv ...")
    tickets = []
    sampled_users = np.random.choice(users["user_id"], size=int(0.15 * len(users)), replace=False)
    for uid in sampled_users:
        for _ in range(np.random.randint(1, 4)):
            tickets.append({
                "user_id": uid,
                "ticket_id": fake.uuid4(),
                "issue_type": np.random.choice(["Billing", "Streaming", "Account", "Technical"]),
                "created_date": fake.date_between(start_date='-6M', end_date='today'),
                "resolved_date": fake.date_between(start_date='-5M', end_date='today')
            })
    df = pd.DataFrame(tickets)
    df.to_csv(f"{RAW_DIR}/support_tickets.csv", index=False)
    return df

# ------------------------
# 6️⃣ Inject Churn Correlation
# ------------------------
def inject_churn_correlation(users, payments, viewing, tickets):
    print("Injecting churn correlation ...")

    # Aggregate payments
    pay_agg = payments.groupby("user_id").agg(
        total_payments=('amount', 'sum'),
        failed_payment_ratio=('status', lambda x: (x == 'Failed').mean())
    ).reset_index()

    # Aggregate viewing
    view_agg = viewing.groupby("user_id").agg(
        total_watch_time=('duration_watched_seconds', 'sum'),
        avg_watch_time_per_session=('duration_watched_seconds', 'mean'),
        days_since_last_activity=('timestamp', lambda x: (pd.Timestamp.today() - pd.to_datetime(x).max()).days)
    ).reset_index()

    # Aggregate tickets
    ticket_agg = tickets.groupby("user_id").size().reset_index(name="support_ticket_count")

    # Merge features
    features = users.merge(pay_agg, on='user_id', how='left') \
                    .merge(view_agg, on='user_id', how='left') \
                    .merge(ticket_agg, on='user_id', how='left')

    # Fill missing values safely
    features['avg_watch_time_per_session'] = features['avg_watch_time_per_session'].fillna(features['avg_watch_time_per_session'].median())
    features['failed_payment_ratio'] = features['failed_payment_ratio'].fillna(0)
    features['support_ticket_count'] = features['support_ticket_count'].fillna(0)
    features['days_since_last_activity'] = features['days_since_last_activity'].fillna(features['days_since_last_activity'].median())

    # ------------------------
    # Churn logic (more balanced)
    # ------------------------
    churn_prob = np.zeros(len(features))
    churn_prob += np.where(features['avg_watch_time_per_session'] < 800, 0.3, 0.0)
    churn_prob += np.where((features['avg_watch_time_per_session'] >= 800) &
                           (features['avg_watch_time_per_session'] < 1200), 0.15, 0.0)
    churn_prob += np.clip(features['days_since_last_activity'] / 90, 0, 0.3)
    churn_prob += np.clip(features['failed_payment_ratio'] * 1.5, 0, 0.2)
    churn_prob += np.clip(features['support_ticket_count'] / 3, 0, 0.1)
    churn_prob += np.random.uniform(0, 0.05, size=len(features))
    churn_prob = churn_prob.clip(0, 1)

    features['is_churned'] = [np.random.choice([0,1], p=[1-p, p]) for p in churn_prob]

    # Save
    features.to_csv(f"{RAW_DIR}/user_churn_features.csv", index=False)
    print(f"✅ Created user_churn_features.csv with balanced churn correlation")
    return features

# ------------------------
# Main
# ------------------------
def main():
    users = generate_users()
    subs = generate_subscriptions(users)
    payments = generate_payments(users)
    viewing = generate_viewing_events(users)
    tickets = generate_support_tickets(users)
    inject_churn_correlation(users, payments, viewing, tickets)
    print("✅ Synthetic Netflix dataset generated successfully!")
    print("Files saved in 'data/raw/' folder.")

if __name__ == "__main__":
    main()
