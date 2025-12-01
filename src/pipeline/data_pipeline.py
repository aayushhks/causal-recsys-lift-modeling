# src/pipeline/data_pipeline.py
import pandas as pd
import numpy as np
import os

# CONFIGURATION
RAW_EVENTS_PATH = "data/raw/events.csv"
OUTPUT_PATH = "data/processed/impressions.parquet"
ATTRIBUTION_WINDOW = pd.Timedelta(minutes=60)


def load_data():
    if not os.path.exists(RAW_EVENTS_PATH):
        raise FileNotFoundError(f"File not found: {RAW_EVENTS_PATH}")
    df = pd.read_csv(RAW_EVENTS_PATH)
    # Handle timestamp
    if pd.api.types.is_numeric_dtype(df['timestamp']):
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    else:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
    return df.sort_values('timestamp')


def create_impressions(df):
    # Filter for views (impressions)
    impressions = df[df['event_type'] == 'view'].copy()
    impressions = impressions.rename(columns={'timestamp': 'impression_time'})
    impressions['impression_id'] = (
            impressions['user_id'].astype(str) + "_" +
            impressions['item_id'].astype(str) + "_" +
            impressions['impression_time'].astype(np.int64).astype(str)
    )
    # Initialize targets
    for col in ['clicked', 'added_to_cart', 'purchased']:
        impressions[col] = 0
    return impressions


def attribute_events(impressions, all_events, event_type, target_col):
    target_events = all_events[all_events['event_type'] == event_type].copy()
    target_events = target_events.sort_values('timestamp')

    matched = pd.merge_asof(
        target_events,
        impressions[['impression_id', 'user_id', 'item_id', 'impression_time']],
        left_on='timestamp',
        right_on='impression_time',
        by=['user_id', 'item_id'],
        direction='backward',
        tolerance=ATTRIBUTION_WINDOW
    )
    successful_ids = matched['impression_id'].dropna().unique()
    impressions.loc[impressions['impression_id'].isin(successful_ids), target_col] = 1
    return impressions


def run_pipeline():
    print("⏳ Running Pipeline...")
    df = load_data()
    impressions = create_impressions(df)

    # Attribute outcomes
    impressions = attribute_events(impressions, df, 'click', 'clicked')
    impressions = attribute_events(impressions, df, 'addtocart', 'added_to_cart')
    impressions = attribute_events(impressions, df, 'transaction', 'purchased')

    # --- NEW: Assign A/B Test Variant (The Missing Step) ---
    print("🎲 Assigning A/B Test Variants...")
    np.random.seed(42)
    # 50/50 Split: Control vs Treatment
    impressions['variant'] = np.where(np.random.rand(len(impressions)) > 0.5, 'Treatment', 'Control')

    # Simple Feature: Hour of day (as a confounder example)
    impressions['hour_of_day'] = impressions['impression_time'].dt.hour

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    impressions.to_parquet(OUTPUT_PATH, index=False)
    print(f"✅ Saved {len(impressions):,} rows with 'variant' column to {OUTPUT_PATH}")


if __name__ == "__main__":
    run_pipeline()