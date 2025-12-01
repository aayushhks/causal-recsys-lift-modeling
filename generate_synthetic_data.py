import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta


def generate():
    print(" Generating SIGNAL-RICH synthetic data...")
    n_views = 50_000
    n_users = 2000
    n_items = 500

    # 1. Setup Item Popularity (Power Law)
    # Items 0-50 are "Viral", 51-500 are "Long Tail"
    item_ids = np.arange(n_items)
    item_probs = 1 / (item_ids + 1)  # Zipfian distribution
    item_probs /= item_probs.sum()

    # 2. Setup User Activity Levels
    # Users 0-200 are "Power Users"
    user_ids = np.arange(n_users)
    user_activity = 1 / (user_ids + 1)
    user_activity /= user_activity.sum()

    # 3. Generate Views (Biased towards popular items and active users)
    base_time = datetime.now() - timedelta(days=30)

    df = pd.DataFrame({
        'timestamp': [base_time + timedelta(seconds=np.random.randint(0, 30 * 24 * 3600)) for _ in range(n_views)],
        'user_id': np.random.choice(user_ids, size=n_views, p=user_activity),
        'item_id': np.random.choice(item_ids, size=n_views, p=item_probs),
        'event_type': 'view'
    })

    # 4. Generate Clicks based on FEATURE SIGNAL
    # Formula: P(Click) increases if Item is Popular OR User is Power User OR Weekend

    # Calculate "Intrinsic Quality" of the view
    # Normalized rank (0 to 1, where 1 is most popular)
    item_quality = 1 - (df['item_id'] / n_items)
    user_quality = 1 - (df['user_id'] / n_users)

    # Time Signal: Weekends get a boost
    df['is_weekend'] = df['timestamp'].dt.dayofweek.isin([5, 6]).astype(int)

    # Base Probability = 5%
    # + up to 15% for popular items
    # + up to 10% for power users
    # + 5% flat boost for weekends
    df['click_prob'] = 0.05 + (0.15 * item_quality) + (0.10 * user_quality) + (0.05 * df['is_weekend'])

    # Clip to max 80% to add some noise
    df['click_prob'] = df['click_prob'].clip(0, 0.8)

    # Roll the dice
    df['clicked'] = np.random.rand(len(df)) < df['click_prob']

    # Create Click Events
    clicks = df[df['clicked'] == True].copy()
    clicks['event_type'] = 'click'
    clicks['timestamp'] += pd.to_timedelta(np.random.randint(30, 300, size=len(clicks)), unit='s')

    # 5. Generate Purchases (Strong signal: Very popular items get bought)
    clicks['purchase_prob'] = 0.05 + (0.2 * item_quality[clicks.index])
    clicks['purchased'] = np.random.rand(len(clicks)) < clicks['purchase_prob']

    purchases = clicks[clicks['purchased'] == True].copy()
    purchases['event_type'] = 'transaction'
    purchases['timestamp'] += pd.to_timedelta(np.random.randint(60, 600, size=len(purchases)), unit='s')

    # 6. Cleanup & Save
    final_df = pd.concat([
        df[['timestamp', 'user_id', 'item_id', 'event_type']],
        clicks[['timestamp', 'user_id', 'item_id', 'event_type']],
        purchases[['timestamp', 'user_id', 'item_id', 'event_type']]
    ]).sort_values('timestamp')

    os.makedirs('data/raw', exist_ok=True)
    final_df.to_csv('data/raw/events.csv', index=False)

    print(f" Generated {len(final_df):,} events with SIGNAL.")
    print(f"   Views: {len(df)}")
    print(f"   Clicks: {len(clicks)} (Derived from Popularity + User Activity)")
    print(f"   Purchases: {len(purchases)}")


if __name__ == "__main__":
    generate()