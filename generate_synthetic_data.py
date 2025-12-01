# generate_synthetic_data.py
import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta


def generate():
    print(" Generating synthetic e-commerce data...")
    n_rows = 100_000
    n_users = 2000
    n_items = 500

    # Probabilities for funnel: View (85%), Click (10%), Cart (4%), Purchase (1%)
    event_types = ['view', 'click', 'addtocart', 'transaction']
    probs = [0.85, 0.10, 0.04, 0.01]

    data = {
        'timestamp': [datetime.now() - timedelta(minutes=np.random.randint(1, 10000)) for _ in range(n_rows)],
        'user_id': np.random.randint(1000, 1000 + n_users, size=n_rows),
        'item_id': np.random.randint(1, n_items, size=n_rows),
        'event_type': np.random.choice(event_types, size=n_rows, p=probs)
    }

    df = pd.DataFrame(data)
    df = df.sort_values('timestamp')

    os.makedirs('data/raw', exist_ok=True)
    df.to_csv('data/raw/events.csv', index=False)
    print(" Created data/raw/events.csv")


if __name__ == "__main__":
    generate()