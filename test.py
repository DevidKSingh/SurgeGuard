import pandas as pd
import numpy as np

# ============================================================
# 1. LOAD DATASET
# ============================================================

train = pd.read_csv("train.csv")

print("Original dataset shape:")
print(train.shape)

print("\nFirst 5 rows:")
print(train.head())


# ============================================================
# 2. SORT BY TIME
# ============================================================
# Time-based features must be calculated chronologically.
# This prevents future transactions from being used as
# historical information.

train = train.sort_values("Time").reset_index(drop=True)


# ============================================================
# 3. BASIC FEATURE ENGINEERING
# ============================================================

# Approximate elapsed-hour bucket
train["Time_Hour"] = (train["Time"] // 3600) % 24

# Log transformation of transaction amount
# Helps reduce the effect of extremely large amounts.
train["Log_Amount"] = np.log1p(train["Amount"])

# Time difference from the previous transaction
train["Time_Since_Previous"] = train["Time"].diff().fillna(0)


# ============================================================
# 4. VELOCITY FEATURES
# ============================================================
# These measure GLOBAL transaction density.
#
# IMPORTANT:
# The current dataset does not contain Account_ID.
# Therefore these are NOT per-account velocity features.
#
# They measure how many transactions occurred recently
# across the entire dataset.


time_values = train["Time"].values


def transactions_in_window(times, window):
    """
    Count previous transactions occurring within
    the specified time window.

    window:
        10  -> previous 10 seconds
        60  -> previous 60 seconds
        300 -> previous 5 minutes
    """

    counts = []

    for i, current_time in enumerate(times):

        start_time = current_time - window

        # Find the first transaction inside the window
        left = times.searchsorted(start_time, side="left")

        # Exclude the current transaction itself
        count = i - left

        counts.append(count)

    return counts


# Previous transactions in last 10 seconds
train["Transactions_Last_10s"] = transactions_in_window(
    time_values,
    10
)

# Previous transactions in last 60 seconds
train["Transactions_Last_60s"] = transactions_in_window(
    time_values,
    60
)

# Previous transactions in last 300 seconds (5 minutes)
train["Transactions_Last_300s"] = transactions_in_window(
    time_values,
    300
)


# ============================================================
# 5. DISPLAY ENGINEERED FEATURES
# ============================================================

print("\n" + "=" * 70)
print("ENGINEERED FEATURES")
print("=" * 70)

print(
    train[
        [
            "Time",
            "Amount",
            "Class",
            "Time_Hour",
            "Log_Amount",
            "Time_Since_Previous",
            "Transactions_Last_10s",
            "Transactions_Last_60s",
            "Transactions_Last_300s"
        ]
    ].head(30)
)


# ============================================================
# 6. CHECK AVERAGE VELOCITY BY CLASS
# ============================================================

print("\n" + "=" * 70)
print("AVERAGE VELOCITY BY CLASS")
print("=" * 70)

velocity_columns = [
    "Time_Since_Previous",
    "Transactions_Last_10s",
    "Transactions_Last_60s",
    "Transactions_Last_300s"
]

print(
    train.groupby("Class")[velocity_columns].mean()
)


# ============================================================
# 7. CHECK MEDIAN VELOCITY BY CLASS
# ============================================================

print("\n" + "=" * 70)
print("MEDIAN VELOCITY BY CLASS")
print("=" * 70)

print(
    train.groupby("Class")[velocity_columns].median()
)


# ============================================================
# 8. DETAILED STATISTICS FOR FRAUD TRANSACTIONS
# ============================================================

print("\n" + "=" * 70)
print("FRAUD TRANSACTION VELOCITY STATISTICS")
print("=" * 70)

fraud_data = train[train["Class"] == 1]

print(
    fraud_data[
        [
            "Time",
            "Amount",
            "Time_Since_Previous",
            "Transactions_Last_10s",
            "Transactions_Last_60s",
            "Transactions_Last_300s"
        ]
    ].describe()
)


# ============================================================
# 9. DATASET CLASS DISTRIBUTION
# ============================================================

print("\n" + "=" * 70)
print("CLASS DISTRIBUTION")
print("=" * 70)

print(train["Class"].value_counts())

print("\nClass percentages:")
print(
    train["Class"].value_counts(normalize=True) * 100
)


# ============================================================
# 10. FINAL FEATURE LIST
# ============================================================

print("\n" + "=" * 70)
print("FINAL COLUMNS")
print("=" * 70)

print(train.columns.tolist())


# ============================================================
# 11. CHECK FOR MISSING VALUES
# ============================================================

print("\n" + "=" * 70)
print("MISSING VALUES")
print("=" * 70)

print(
    train.isnull().sum()
)