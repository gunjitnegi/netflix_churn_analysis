import pandas as pd
import numpy as np
import os

# -----------------------
# CONFIGURATION
# -----------------------
RAW_DIR = "data/raw"
INTERIM_DIR = "data/interim"
os.makedirs(INTERIM_DIR, exist_ok=True)


# -----------------------
# CLEANING FUNCTIONS
# -----------------------

def clean_users(df):
    print("🧍 Cleaning users.csv...")
    df = df.copy()

    if "country" in df.columns:
        df["country"] = df["country"].fillna(df["country"].mode()[0])
    if "age_group" in df.columns:
        df["age_group"] = df["age_group"].replace("Unknown", np.nan)
    if "plan_type" in df.columns:
        df["plan_type"] = df["plan_type"].astype(str).str.title()
    if "signup_channel" in df.columns:
        df["signup_channel"] = df["signup_channel"].astype(str).str.title()
    if "region" in df.columns:
        df["region"] = df["region"].fillna("Unknown")

    df = df.drop_duplicates()
    return df


def clean_subscriptions(df):
    print("💳 Cleaning subscriptions.csv...")
    df = df.copy()

    if "status" in df.columns:
        df["status"] = df["status"].replace("Unknown", "Cancelled")
    if "payment_method" in df.columns:
        df["payment_method"] = df["payment_method"].fillna("Unknown")

    if "start_date" in df.columns:
        df["start_date"] = pd.to_datetime(df["start_date"], errors="coerce")
    if "end_date" in df.columns:
        df["end_date"] = pd.to_datetime(df["end_date"], errors="coerce")

    if "start_date" in df.columns and "end_date" in df.columns:
        invalid_dates = df["end_date"] < df["start_date"]
        df.loc[invalid_dates, "end_date"] = pd.NaT

    df = df.drop_duplicates()
    return df


def clean_payments(df):
    print("💰 Cleaning payments.csv...")
    df = df.copy()

    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
    if "amount" in df.columns:
        df["amount"] = pd.to_numeric(df["amount"], errors="coerce")

    # Remove invalid/negative amounts
    df = df[df["amount"] > 0].copy()

    # Fill missing values safely
    if "status" in df.columns:
        df["status"] = df["status"].fillna("Pending")
    if "failure_reason" in df.columns:
        df["failure_reason"] = df["failure_reason"].fillna("None")
    if "retry_count" in df.columns:
        df["retry_count"] = df["retry_count"].fillna(0).astype(int)

    df = df.drop_duplicates()
    return df


def clean_viewing_events(df):
    print("🎬 Cleaning viewing_events.csv...")
    df = df.copy()

    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        df = df.dropna(subset=["timestamp"]).copy()

    if "duration_watched_seconds" in df.columns:
        df["duration_watched_seconds"] = pd.to_numeric(df["duration_watched_seconds"], errors="coerce")

        # Replace missing durations with mean per user
        df["duration_watched_seconds"] = (
            df.groupby("user_id")["duration_watched_seconds"]
            .transform(lambda x: x.fillna(x.mean()))
        )
        # Fill remaining NaNs with global mean
        df["duration_watched_seconds"] = df["duration_watched_seconds"].fillna(
            df["duration_watched_seconds"].mean()
        )

        # Keep only positive durations
        df = df[df["duration_watched_seconds"] > 0].copy()

    df = df.drop_duplicates()
    return df


def clean_support_tickets(df):
    print("🧾 Cleaning support_tickets.csv...")
    df = df.copy()

    if "created_at" in df.columns:
        df["created_at"] = pd.to_datetime(df["created_at"], errors="coerce")
    if "resolved_at" in df.columns:
        df["resolved_at"] = pd.to_datetime(df["resolved_at"], errors="coerce")

    # Fix invalid date order
    if "created_at" in df.columns and "resolved_at" in df.columns:
        invalid_order = df["resolved_at"] < df["created_at"]
        df.loc[invalid_order, "resolved_at"] = pd.NaT

    if "sentiment" in df.columns:
        df["sentiment"] = df["sentiment"].fillna("Neutral")

    if "topic" in df.columns:
        df["topic"] = df["topic"].astype(str).str.title()

    df = df.drop_duplicates()
    return df


# -----------------------
# MAIN PIPELINE
# -----------------------

def main():
    print("🚀 Starting Data Cleaning Process...\n")

    datasets = {
        "users": (clean_users, os.path.join(RAW_DIR, "users.csv")),
        "subscriptions": (clean_subscriptions, os.path.join(RAW_DIR, "subscriptions.csv")),
        "payments": (clean_payments, os.path.join(RAW_DIR, "payments.csv")),
        "viewing_events": (clean_viewing_events, os.path.join(RAW_DIR, "viewing_events.csv")),
        "support_tickets": (clean_support_tickets, os.path.join(RAW_DIR, "support_tickets.csv")),
    }

    for name, (clean_func, path) in datasets.items():
        if os.path.exists(path):
            print(f"\n➡️ Cleaning {name}.csv ...")
            df = pd.read_csv(path)
            cleaned_df = clean_func(df)
            output_path = os.path.join(INTERIM_DIR, f"{name}_cleaned.csv")
            cleaned_df.to_csv(output_path, index=False)
            print(f"✅ Saved cleaned file: {output_path}")
        else:
            print(f"⚠️ File not found: {path}")

    print("\n🎉 Data Cleaning Complete! Cleaned datasets are in /data/interim/")


if __name__ == "__main__":
    main()
