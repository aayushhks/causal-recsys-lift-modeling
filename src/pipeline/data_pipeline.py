# src/data_pipeline.py
import pandas as pd
import numpy as np
import os

# CONFIGURATION
RAW_EVENTS_PATH = "data/raw/events.csv"
ITEMS_PATH = "data/raw/item_properties.csv"  # Optional
OUTPUT_PATH = "data/processed/impressions.parquet"

# Attribution window: How long after a view do we count a click? (e.g., 60 mins)
ATTRIBUTION_WINDOW = pd.Timedelta(minutes=60)


def load_data():
    """
    Loads raw events and ensures correct types.
    """
    print(f"⏳ Loading data from {RAW_EVENTS_PATH}...")
    if not os.path.exists(RAW_EVENTS_PATH):
        raise FileNotFoundError(f" File not found: {RAW_EVENTS_PATH}. Run generate_synthetic_data.py first!")

    df = pd.read_csv(RAW_EVENTS_PATH)

    # Convert timestamps to datetime (handle Unix ms or standard strings)
    if pd.api.types.is_numeric_dtype(df['timestamp']):
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    else:
        df['timestamp'] = pd.to_datetime(df['timestamp'])

    # Sort by time is REQUIRED for merge_asof
    df = df.sort_values('timestamp')
    print(f" Loaded {len(df):,} events.")
    return df


def create_impressions(df):
    """
    1. Filter for 'view' events (these are our potential recommendations).
    2. Create a unique ID for every impression.
    """
    print("  Creating base impression table...")

    # In this dataset, a 'view' is a proxy for a recommendation impression
    impressions = df[df['event_type'] == 'view'].copy()

    # Rename timestamp to impression_time for clarity
    impressions = impressions.rename(columns={'timestamp': 'impression_time'})

    # Create a unique Impression ID
    impressions['impression_id'] = (
            impressions['user_id'].astype(str) + "_" +
            impressions['item_id'].astype(str) + "_" +
            impressions['impression_time'].view('int64').astype(str)
    )

    # Initialize target variables (all 0 for now)
    impressions['clicked'] = 0
    impressions['added_to_cart'] = 0
    impressions['purchased'] = 0

    return impressions


def attribute_events(impressions, all_events, event_type, target_col):
    """
    Matches downstream events (clicks/carts) to the most recent view
    using a backward-looking 'asof' merge.
    """
    print(f" Attributing '{event_type}' to impressions...")

    # Filter for the specific event type (e.g., 'click')
    target_events = all_events[all_events['event_type'] == event_type].copy()
    target_events = target_events.sort_values('timestamp')

    # Merge AsOf: Find the last 'impression_time' that is BEFORE the 'timestamp'
    # for the same user_id and item_id
    matched = pd.merge_asof(
        target_events,  # Left table (Clicks)
        impressions[['impression_id', 'user_id', 'item_id', 'impression_time']],  # Right table (Views)
        left_on='timestamp',  # Match timestamp...
        right_on='impression_time',  # ...to impression_time
        by=['user_id', 'item_id'],  # Group by User and Item
        direction='backward',  # Look backwards in time
        tolerance=ATTRIBUTION_WINDOW  # Only match if within 1 hour
    )

    # Get the IDs of impressions that successfully "caught" a click
    successful_ids = matched['impression_id'].dropna().unique()

    # Update the main table
    impressions.loc[impressions['impression_id'].isin(successful_ids), target_col] = 1

    count = len(successful_ids)
    print(f"   -> Matched {count:,} {event_type} events.")
    return impressions


def run_pipeline():
    # 1. Load Raw Data
    df = load_data()

    # 2. Create Base Table
    impressions = create_impressions(df)

    # 3. Attribute Targets
    impressions = attribute_events(impressions, df, 'click', 'clicked')
    impressions = attribute_events(impressions, df, 'addtocart', 'added_to_cart')
    impressions = attribute_events(impressions, df, 'transaction', 'purchased')

    # 4. Basic Feature Engineering (Session-level)
    print(" Engineering features...")
    # Example: How many times has this user seen this item before?
    impressions['user_item_view_count'] = impressions.groupby(['user_id', 'item_id']).cumcount()

    # 5. Save
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    impressions.to_parquet(OUTPUT_PATH, index=False)
    print(f" Success! Saved {len(impressions):,} rows to {OUTPUT_PATH}")


if __name__ == "__main__":
    run_pipeline()