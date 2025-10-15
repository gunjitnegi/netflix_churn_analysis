import pandas as pd
import numpy as np
import os

# -----------------------
# CONFIGURATION
# -----------------------
RAW_DIR = "data/raw"
REPORTS_DIR = "data/reports"
os.makedirs(REPORTS_DIR, exist_ok=True)
REPORT_PATH = os.path.join(REPORTS_DIR, "data_quality_report.txt")

# -----------------------
# HELPER FUNCTION
# -----------------------
def analyze_dataset(name, df):
    report = []
    report.append(f"\n{'='*80}")
    report.append(f"📊 DATASET: {name.upper()}")
    report.append(f"{'='*80}")
    report.append(f"Shape: {df.shape[0]} rows × {df.shape[1]} columns")
    report.append("\nColumns:\n" + ", ".join(df.columns))

    # -----------------------
    # Missing values
    # -----------------------
    null_counts = df.isnull().sum()
    null_report = null_counts[null_counts > 0].sort_values(ascending=False)
    if not null_report.empty:
        report.append("\n🔸 Missing Values:\n" + str(null_report))
    else:
        report.append("\n✅ No Missing Values Found.")

    # -----------------------
    # Duplicates
    # -----------------------
    duplicate_count = df.duplicated().sum()
    if duplicate_count > 0:
        report.append(f"\n⚠️ Duplicates Found: {duplicate_count}")
    else:
        report.append("\n✅ No Duplicate Rows Found.")

    # -----------------------
    # Descriptive statistics
    # -----------------------
    try:
        desc = df.describe(include='all', datetime_is_numeric=True)
    except TypeError:
        desc = df.describe(include='all')
    report.append("\n📈 Descriptive Statistics (first 10 columns):\n" +
                  str(desc.transpose().head(10)))

    # -----------------------
    # Dataset-specific checks
    # -----------------------
    if name == "payments":
        if "amount" in df.columns:
            invalid_amounts = (df["amount"] <= 0).sum()
            report.append(f"\n💰 Invalid Payment Amounts (<=0): {invalid_amounts}")
        if "status" in df.columns:
            failed = df[df["status"].isin(["Failed", "Pending"])].shape[0]
            report.append(f"Failed/Pending Transactions: {failed}")
        if "failure_reason" in df.columns:
            missing_reasons = df["failure_reason"].isnull().sum()
            report.append(f"Missing Failure Reasons: {missing_reasons}")

    if name == "subscriptions":
        if "status" in df.columns:
            unknown_status = df["status"].isin(["Unknown"]).sum()
            report.append(f"\n🔖 Unknown Subscription Statuses: {unknown_status}")
        if "payment_method" in df.columns:
            null_payment_method = df["payment_method"].isnull().sum()
            report.append(f"Missing Payment Methods: {null_payment_method}")

    if name == "viewing_events":
        if "duration_watched_seconds" in df.columns:
            invalid_durations = (df["duration_watched_seconds"] <= 0).sum()
            report.append(f"\n🎬 Invalid Durations (<=0): {invalid_durations}")
        if "timestamp" in df.columns:
            missing_timestamps = df["timestamp"].isnull().sum()
            report.append(f"Missing Timestamps: {missing_timestamps}")

    if name == "support_tickets":
        created_col = "created_date" if "created_date" in df.columns else "created_at"
        resolved_col = "resolved_date" if "resolved_date" in df.columns else "resolved_at"

        if created_col in df.columns and resolved_col in df.columns:
            df_tmp = df.copy()
            df_tmp[created_col] = pd.to_datetime(df_tmp[created_col], errors="coerce")
            df_tmp[resolved_col] = pd.to_datetime(df_tmp[resolved_col], errors="coerce")
            reversed_dates = (df_tmp[resolved_col] < df_tmp[created_col]).sum()
            report.append(f"\n📅 Tickets with Invalid Date Order: {reversed_dates}")

        if "sentiment" in df.columns:
            missing_sentiments = df["sentiment"].isnull().sum()
            report.append(f"Missing Sentiments: {missing_sentiments}")

    if name == "users":
        if "age_group" in df.columns:
            unknown_ages = df["age_group"].isin(["Unknown"]).sum()
            report.append(f"\n👤 Unknown Age Groups: {unknown_ages}")
        if "country" in df.columns:
            missing_country = df["country"].isnull().sum()
            report.append(f"Missing Country Values: {missing_country}")

    return "\n".join(report)

# -----------------------
# MAIN SCRIPT
# -----------------------
def main():
    print("🔍 Running Data Quality Checks...\n")

    datasets = {
        "users": os.path.join(RAW_DIR, "users.csv"),
        "subscriptions": os.path.join(RAW_DIR, "subscriptions.csv"),
        "payments": os.path.join(RAW_DIR, "payments.csv"),
        "viewing_events": os.path.join(RAW_DIR, "viewing_events.csv"),
        "support_tickets": os.path.join(RAW_DIR, "support_tickets.csv"),
    }

    full_report = []
    for name, path in datasets.items():
        if os.path.exists(path):
            try:
                df = pd.read_csv(path)
                full_report.append(analyze_dataset(name, df))
            except Exception as e:
                full_report.append(f"\n❌ ERROR reading {name}: {e}")
        else:
            full_report.append(f"\n⚠️ FILE MISSING: {path}")

    # Write combined report
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write("\n\n".join(full_report))

    print(f"\n✅ Data Quality Report Generated Successfully: {REPORT_PATH}")

if __name__ == "__main__":
    main()
