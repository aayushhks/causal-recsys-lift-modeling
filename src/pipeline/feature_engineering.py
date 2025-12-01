# src/pipeline/feature_engineering.py
import pandas as pd
import numpy as np
import os

INPUT_PATH = "data/processed/impressions.parquet"
OUTPUT_PATH = "data/features/training_set.parquet"


def load_processed_data():
    if not os.path.exists(INPUT_PATH):
        raise FileNotFoundError(f"{INPUT_PATH} not found. Run pipeline/data_pipeline.py first.")
    return pd.read_parquet(INPUT_PATH)


def engineer_features(df):
    print(" Engineering Features...")

    # 1. Temporal Features (Time of Day, Weekend)
    df['hour'] = df['impression_time'].dt.hour
    df['day_of_week'] = df['impression_time'].dt.dayofweek
    df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)

    # 2. User History Features
    # (Assuming the dataframe is sorted by time)
    df['user_view_count'] = df.groupby('user_id').cumcount()

    # 3. Item Popularity (Rolling window simulation)
    # Global popularity so far
    df['item_global_views'] = df.groupby('item_id').cumcount()

    # 4. Interaction Features
    df['user_item_log_views'] = np.log1p(df['user_view_count'])

    # 5. Clean / Fill NAs
    df = df.fillna(0)

    print(f" Generated {df.shape[1]} features.")
    return df


def save_features(df):
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    df.to_parquet(OUTPUT_PATH, index=False)
    print(f" Saved features to {OUTPUT_PATH}")


if __name__ == "__main__":
    df = load_processed_data()
    df_features = engineer_features(df)
    save_features(df_features)