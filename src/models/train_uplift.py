# src/models/train_uplift.py
import pandas as pd
import xgboost as xgb
import joblib
import os
import numpy as np

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
        # We use slightly shallower trees for Uplift to prevent overfitting on the treatment effect
        self.m0 = xgb.XGBClassifier(objective='binary:logistic', n_estimators=50, max_depth=3)
        self.m1 = xgb.XGBClassifier(objective='binary:logistic', n_estimators=50, max_depth=3)

    def fit(self, X, y, t):
        # Train Control Model (t=0)
        print(f"   Training Control Model (T=0) on {sum(t == 0)} samples...")
        if y[t == 0].nunique() < 2:
            print("   ️ Warning: Control group has constant outcome. Model will predict constant.")
            # Dummy fit or skip - for robust code we let XGBoost handle it or handle specific edge case
            # But swapping target to 'purchased' usually fixes this.

        self.m0.fit(X[t == 0], y[t == 0])

        # Train Treatment Model (t=1)
        print(f"   Training Treatment Model (T=1) on {sum(t == 1)} samples...")
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

    # 1. Handle Treatment Column
    if 'variant' in df.columns:
        # Assuming synthetic pipeline set 'Treatment' / 'Control'
        # Or if we ran ingest_retailrocket, we might not have 'variant' yet.
        # Let's verify string matching
        df['is_treated'] = df['variant'].apply(lambda x: 1 if str(x).lower() == 'treatment' else 0)
    else:
        # If we are running on pure RetailRocket data without the synthetic pipeline's AB assignment,
        # we need to simulate a Randomized Control Trial (RCT) for training purposes.
        print("    No 'variant' column found. Simulating 50/50 RCT assignment.")
        np.random.seed(42)
        df['is_treated'] = np.random.randint(0, 2, size=len(df))

    # 2. Handle Target Variable (Fix for RetailRocket)
    # If clicks are empty, use purchases
    if df['clicked'].sum() == 0:
        print("  Warning: 'clicked' has 0 positives. Switching target to 'purchased'.")
        target = 'purchased'
    else:
        target = 'clicked'

    print(f" Target Variable: {target}")
    print(f"   Positive Rate: {df[target].mean():.4%}")

    # 3. Define Features
    # Drop IDs, leaks, targets, and text columns
    drop_cols = ['impression_id', 'user_id', 'item_id', 'impression_time',
                 'clicked', 'added_to_cart', 'purchased', 'variant', 'is_treated',
                 'event_type', 'transaction_id']

    features = [c for c in df.columns if c not in drop_cols]

    X = df[features]
    y = df[target]  # Target
    t = df['is_treated']  # Treatment Indicator

    print(f"   Training T-Learner on {len(X)} samples using features: {features}")

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