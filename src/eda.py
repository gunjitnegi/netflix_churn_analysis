"""
Exploratory Data Analysis (EDA) for Netflix Churn Project
---------------------------------------------------------
Generates visual summaries and relationships between churn
and user behaviors, saving plots and summary tables.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import squarify  # for treemap
import os

# -----------------------
# CONFIGURATION
# -----------------------
PROCESSED_DIR = "data/processed"
REPORTS_DIR = "data/reports"
os.makedirs(REPORTS_DIR, exist_ok=True)

# -----------------------
# LOAD DATA
# -----------------------
df = pd.read_csv(os.path.join(PROCESSED_DIR, "analysis_ready.csv"))

# Ensure required columns exist
required_cols = ['plan_type', 'country', 'is_churned',
                 'avg_watch_time_per_session', 'failed_payment_ratio',
                 'support_ticket_count', 'days_since_last_activity']
for col in required_cols:
    if col not in df.columns:
        df[col] = 0
        print(f"⚠️ Column '{col}' not found. Created placeholder with zeros.")

# -----------------------
# CLIP EXTREME VALUES (95th percentile)
# -----------------------
numeric_cols = ['avg_watch_time_per_session', 'days_since_last_activity',
                'support_ticket_count', 'failed_payment_ratio']

for col in numeric_cols:
    upper = df[col].quantile(0.95)
    df[col + '_clipped'] = np.clip(df[col], 0, upper)

# -----------------------
# SET PLOT STYLE
# -----------------------
sns.set(style="whitegrid")
plt.rcParams["figure.figsize"] = (10, 6)

# -----------------------
# 1️⃣ Overall Churn Rate
# -----------------------
churn_rate = df['is_churned'].mean()
print(f"Overall Churn Rate: {churn_rate:.2%}")

# -----------------------
# 2️⃣ Churn by Plan Type (Donut Chart)
# -----------------------
plan_churn = df.groupby('plan_type')['is_churned'].mean().sort_values(ascending=False)

plt.figure()
sizes = plan_churn.values
labels = plan_churn.index
colors = sns.color_palette("pastel")[0:len(labels)]

wedges, texts, autotexts = plt.pie(
    sizes,
    labels=labels,
    autopct="%1.1f%%",
    startangle=90,
    colors=colors,
    wedgeprops=dict(width=0.4)
)
plt.title("Churn Rate by Plan Type (Donut Chart)")
plt.tight_layout()
plt.savefig(os.path.join(REPORTS_DIR, "churn_by_plan_type_donut.png"))
plt.close()

# -----------------------
# 3️⃣ Churn by Country (Treemap)
# -----------------------
country_churn = df.groupby('country')['is_churned'].mean().sort_values(ascending=False).head(10)
plt.figure(figsize=(12, 8))
squarify.plot(
    sizes=country_churn.values,
    label=[f"{c}\n{r:.2%}" for c, r in zip(country_churn.index, country_churn.values)],
    color=sns.color_palette("Set3"),
    alpha=0.8
)
plt.title("Top 10 Countries by Churn Rate (Treemap)")
plt.axis("off")
plt.tight_layout()
plt.savefig(os.path.join(REPORTS_DIR, "churn_by_country_treemap.png"))
plt.close()

# -----------------------
# 4️⃣ Avg Watch Time vs Churn (Strip Plot, Swarm Alternative)
# -----------------------
plt.figure()
sample_df = df.sample(n=min(1000, len(df)), random_state=42)
sns.stripplot(
    data=sample_df,
    x='is_churned',
    y='avg_watch_time_per_session_clipped',
    hue='is_churned',
    dodge=False,
    jitter=0.25,
    alpha=0.7,
    size=4,
    palette="Set2",
    legend=False
)
plt.xlabel("Churned (0 = No, 1 = Yes)")
plt.ylabel("Avg Watch Time per Session (clipped)")
plt.title("Avg Watch Time vs Churn (Strip Plot)")
plt.tight_layout()
plt.savefig(os.path.join(REPORTS_DIR, "watch_time_vs_churn_strip.png"))
plt.close()

# -----------------------
# 5️⃣ Days Since Last Activity vs Churn (KDE Plot)
# -----------------------
plt.figure()
sns.kdeplot(
    data=df,
    x='days_since_last_activity_clipped',
    hue='is_churned',
    fill=True,
    common_norm=False,
    palette="coolwarm",
    alpha=0.5
)
plt.xlabel("Days Since Last Activity (clipped)")
plt.ylabel("Density")
plt.title("Days Since Last Activity vs Churn (KDE Plot)")
plt.tight_layout()
plt.savefig(os.path.join(REPORTS_DIR, "days_vs_churn_kde.png"))
plt.close()

# -----------------------
# 6️⃣ Support Ticket Count vs Churn (Strip Plot)
# -----------------------
plt.figure()
sns.stripplot(
    data=sample_df,
    x='is_churned',
    y='support_ticket_count_clipped',
    hue='is_churned',
    dodge=False,
    jitter=0.3,
    alpha=0.7,
    size=4,
    palette="muted",
    legend=False
)
plt.xlabel("Churned")
plt.ylabel("Support Ticket Count (clipped)")
plt.title("Support Tickets vs Churn (Strip Plot)")
plt.tight_layout()
plt.savefig(os.path.join(REPORTS_DIR, "tickets_vs_churn_strip.png"))
plt.close()

# -----------------------
# 7️⃣ Failed Payment Ratio vs Churn (Violin Plot)
# -----------------------
plt.figure()
sns.violinplot(
    data=df,
    x='is_churned',
    y='failed_payment_ratio_clipped',
    palette="Set3",
    inner="quartile",
    cut=0
)
plt.xlabel("Churned")
plt.ylabel("Failed Payment Ratio (clipped)")
plt.title("Failed Payment Ratio vs Churn (Violin Plot)")
plt.tight_layout()
plt.savefig(os.path.join(REPORTS_DIR, "failed_payment_vs_churn_violin.png"))
plt.close()

# -----------------------
# SAVE SUMMARY TABLES
# -----------------------
summary = {
    "overall_churn_rate": [churn_rate],
    "churn_by_plan": [plan_churn.to_dict()],
    "top_countries_by_churn": [country_churn.to_dict()]
}
summary_df = pd.DataFrame(summary)
summary_df.to_csv(os.path.join(REPORTS_DIR, "eda_summary.csv"), index=False)

print(f"\n✅ EDA completed successfully.")
print(f"📊 Charts & summary saved in '{REPORTS_DIR}' folder.")
