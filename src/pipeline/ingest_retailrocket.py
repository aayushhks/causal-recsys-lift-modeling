import pandas as pd
import numpy as np
import os

# CONFIG
RAW_INPUT_PATH = "data/raw/retailrocket_events.csv"  # The file you downloaded
CANONICAL_OUTPUT_PATH = "data/raw/events.csv"  # The standard format your pipeline expects


def ingest_retailrocket():
    print(f" Starting Ingestion from {RAW_INPUT_PATH}...")

    if not os.path.exists(RAW_INPUT_PATH):
        raise FileNotFoundError(f" Could not find {RAW_INPUT_PATH}. Did you download it from Kaggle?")

    # 1. Load Data
    # RetailRocket timestamps are in Unix Milliseconds
    df = pd.read_csv(RAW_INPUT_PATH)
    print(f"   Loaded {len(df):,} raw rows.")

    # 2. Standardize Schema (The Adapter Step)
    print(" Adapting Schema...")
    df = df.rename(columns={
        'visitorid': 'user_id',
        'itemid': 'item_id',
        'event': 'event_type',
        'transactionid': 'transaction_id'  # Optional, but good to keep
    })

    # 3. Standardize Timestamps
    # Convert ms integer to datetime object
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')

    # 4. Standardize Event Names
    # RetailRocket uses 'view', 'addtocart', 'transaction'
    # Your synthetic data used 'view', 'click', 'addtocart', 'purchase'
    # We need to map them to what your Feature Engineering expects
    event_map = {
        'view': 'view',
        'addtocart': 'addtocart',
        'transaction': 'transaction'  # We can treat this as 'purchase' or keep as 'transaction'
    }
    # Note: RetailRocket has no explicit "click" event (View is the proxy for click usually)
    # We will trust the raw names but ensure they are lowercase
    df['event_type'] = df['event_type'].map(event_map)

    # 5. Data Quality Checks (Simple)
    print(" Running Quality Checks...")
    # Drop rows with missing critical IDs
    original_len = len(df)
    df = df.dropna(subset=['user_id', 'item_id', 'timestamp'])
    if len(df) < original_len:
        print(f"   ️ Dropped {original_len - len(df)} rows with null keys.")

    # 6. Sort by Time (Critical for sessionization)
    df = df.sort_values('timestamp')

    # 7. Save to Canonical Path
    print(f" Saving Standardized Data to {CANONICAL_OUTPUT_PATH}...")
    df.to_csv(CANONICAL_OUTPUT_PATH, index=False)
    print(" Ingestion Complete. The main pipeline is now ready to run.")


if __name__ == "__main__":
    ingest_retailrocket()