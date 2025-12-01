import pandas as pd
import numpy as np
import os


def validate_data(df):
    print("  Running Data Validation Checks...")

    # 1. Schema Check
    required_cols = ['user_id', 'item_id', 'timestamp', 'event_type']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f" Missing required column: {col}")
    print("    Schema check passed.")

    # 2. Null Check
    null_counts = df[required_cols].isnull().sum()
    if null_counts.any():
        print(f"    Warning: Found nulls in critical columns:\n{null_counts[null_counts > 0]}")
    else:
        print("    No nulls in critical columns.")

    # 3. Treatment Balance Check (if variant exists)
    if 'variant' in df.columns:
        balance = df['variant'].value_counts(normalize=True)
        print(f"   ⚖️  Treatment Balance:\n{balance}")
        if abs(balance['control'] - 0.5) > 0.1:
            print("   ️  Warning: Significant treatment imbalance detected!")
        else:
            print("    Treatment groups are balanced.")

    # 4. Funnel Logic Check
    # Purchases should not exceed Views
    n_views = len(df[df['event_type'] == 'view'])
    n_purchases = len(df[df['event_type'] == 'transaction'])
    if n_purchases > n_views:
        raise ValueError(" Logic Error: More purchases than views!")
    print(f"    Funnel Logic valid (Views: {n_views}, Purchases: {n_purchases})")

    print("  Validation Complete.\n")


if __name__ == "__main__":
    # Test run
    DATA_PATH = "data/raw/events.csv"
    if os.path.exists(DATA_PATH):
        df = pd.read_csv(DATA_PATH)
        validate_data(df)