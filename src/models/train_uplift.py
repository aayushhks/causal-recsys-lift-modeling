# src/models/train_uplift.py
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
import joblib
import os

# CONFIG
DATA_PATH = "data/features/training_set.parquet"
MODEL_DIR = "models/uplift"
MODEL_PATH = os.path.join(MODEL_DIR, "uplift_meta_learner.pkl")


class TLearnerUplift:
    """
    Simple T-Learner implementation using two XGBoost models.
    Model 0: Predicts outcome given Control.
    Model 1: Predicts outcome given Treatment.
    Lift = Model 1 - Model 0
    """

    def __init__(self):
        self.m0 = xgb.XGBClassifier(objective='binary:logistic', n_estimators=50)
        self.m1 = xgb.XGBClassifier(objective='binary:logistic', n_estimators=50)

    def fit(self, X, y, t):
        # Train Control Model (t=0)
        print("   Training Control Model (T=0)...")
        self.m0.fit(X[t == 0], y[t == 0])

        # Train Treatment Model (t=1)
        print("   Training Treatment Model (T=1)...")
        self.m1.fit(X[t == 1], y[t == 1])

    def predict_lift(self, X):
        # Predict Prob(Conversion | Control)
        p0 = self.m0.predict_proba(X)[:, 1]
        # Predict Prob(Conversion | Treatment)
        p1 = self.m1.predict_proba(X)[:, 1]
        # Uplift = P(Treatment) - P(Control)
        return p1 - p0


def train_uplift_model():
    print(" Loading Data for Uplift Modeling...")
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f" Data not found at {DATA_PATH}")

    df = pd.read_parquet(DATA_PATH)

    # We need a 'treatment' column.
    # In our pipeline, we had 'variant'. Let's convert it.
    # Control = 0, Treatment = 1
    if 'variant' in df.columns:
        df['is_treated'] = (df['variant'] == 'Treatment').astype(int)
    else:
        # Fallback for synthetic data if 'variant' is missing
        import numpy as np
        df['is_treated'] = np.random.randint(0, 2, size=len(df))

    drop_cols = ['impression_id', 'user_id', 'item_id', 'impression_time',
                 'clicked', 'added_to_cart', 'purchased', 'variant', 'is_treated', 'event_type']

    features = [c for c in df.columns if c not in drop_cols]

    X = df[features]
    y = df['clicked']  # Target: Click
    t = df['is_treated']  # Treatment Indicator

    print(f"   Training T-Learner on {len(X)} samples...")
    learner = TLearnerUplift()
    learner.fit(X, y, t)

    # Sanity Check
    sample_lift = learner.predict_lift(X.head())
    print(f" Training Complete. Sample Lift Predictions: {sample_lift}")

    # Save Artifact
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(learner, MODEL_PATH)
    print(f" Uplift Model saved to {MODEL_PATH}")


if __name__ == "__main__":
    train_uplift_model()