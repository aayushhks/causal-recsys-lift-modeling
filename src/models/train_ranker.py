# src/models/train_ranker.py
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
        raise FileNotFoundError(f" Data not found at {DATA_PATH}. Run src/pipeline/feature_engineering.py first.")

    df = pd.read_parquet(DATA_PATH)

    # --- FIX IS HERE ---
    # We added 'event_type' to the drop list because it is text and always equals "view"
    drop_cols = ['impression_id', 'user_id', 'item_id', 'impression_time',
                 'clicked', 'added_to_cart', 'purchased', 'variant', 'event_type']

    features = [c for c in df.columns if c not in drop_cols]
    target = 'clicked'

    print(f"   Training on {len(features)} features: {features}")

    X = df[features]
    y = df[target]

    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train XGBoost
    print(" Training XGBoost Ranker...")
    model = xgb.XGBClassifier(
        objective='binary:logistic',
        eval_metric='auc',
        n_estimators=100,
        learning_rate=0.1,
        max_depth=5,
        use_label_encoder=False
    )

    model.fit(X_train, y_train)

    # Evaluate
    preds = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, preds)
    print(f" Model Trained. Test AUC: {auc:.4f}")

    # Save Artifact
    os.makedirs(MODEL_DIR, exist_ok=True)
    model.save_model(MODEL_PATH)
    print(f" Model saved to {MODEL_PATH}")


if __name__ == "__main__":
    train_ranker()