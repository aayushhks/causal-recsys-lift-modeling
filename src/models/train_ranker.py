import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import os

# CONFIG
DATA_PATH = "data/features/training_set.parquet"
MODEL_DIR = "models/ranking"
MODEL_PATH = os.path.join(MODEL_DIR, "xgb_ranker.json")


def train_ranker():
    print(" Loading Feature Data...")
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f" Data not found at {DATA_PATH}.")

    df = pd.read_parquet(DATA_PATH)

    # FOR RETAILROCKET DATA
    # RetailRocket doesn't have 'click' events (view IS the click).
    # We must predict a real outcome: 'purchased' (or 'added_to_cart').

    # Check if we have any positive labels for 'clicked'
    if df['clicked'].sum() == 0:
        print(" Warning: 'clicked' column has 0 positives (expected for RetailRocket).")
        print("   -> Switching target to 'purchased' (Conversion).")
        target = 'purchased'
    else:
        target = 'clicked'

    print(f" Target Variable: {target}")
    print(f"   Positive Rate: {df[target].mean():.4%}")

    # Drop IDs, leaks, and the target itself
    drop_cols = ['impression_id', 'user_id', 'item_id', 'impression_time',
                 'clicked', 'added_to_cart', 'purchased', 'variant', 'event_type',
                 'transaction_id']  # transaction_id is a leak if present

    features = [c for c in df.columns if c not in drop_cols]

    print(f"   Training on {len(features)} features: {features}")

    X = df[features]
    y = df[target]

    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train XGBoost
    # Removed 'use_label_encoder' to fix warning
    print(" Training XGBoost Ranker...")
    model = xgb.XGBClassifier(
        objective='binary:logistic',
        eval_metric='auc',
        n_estimators=100,
        learning_rate=0.1,
        max_depth=5
    )

    model.fit(X_train, y_train)

    # Evaluate
    preds = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, preds)
    print(f" Model Trained. Test AUC: {auc:.4f}")

    # Save
    os.makedirs(MODEL_DIR, exist_ok=True)
    model.save_model(MODEL_PATH)
    print(f" Model saved to {MODEL_PATH}")


if __name__ == "__main__":
    train_ranker()