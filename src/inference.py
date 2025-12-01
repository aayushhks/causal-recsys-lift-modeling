import pandas as pd
import xgboost as xgb
import joblib
import os
import numpy as np

# CONFIG
RANKER_MODEL_PATH = "models/ranking/xgb_ranker.json"
UPLIFT_MODEL_PATH = "models/uplift/uplift_meta_learner.pkl"
FEATURE_DATA_PATH = "data/features/training_set.parquet"  # In prod, this would be a live feature store


class RecommendationServingEngine:
    def __init__(self):
        self.ranker = None
        self.uplift_model = None
        self.load_models()

    def load_models(self):
        print(" Loading Production Models...")

        # Load Ranker
        if os.path.exists(RANKER_MODEL_PATH):
            self.ranker = xgb.Booster()
            self.ranker.load_model(RANKER_MODEL_PATH)
            print("    Ranker loaded.")
        else:
            raise FileNotFoundError(f"Ranker model not found at {RANKER_MODEL_PATH}")

        # Load Uplift
        if os.path.exists(UPLIFT_MODEL_PATH):
            self.uplift_model = joblib.load(UPLIFT_MODEL_PATH)
            print("    Uplift Model loaded.")
        else:
            raise FileNotFoundError(f"Uplift model not found at {UPLIFT_MODEL_PATH}")

    def predict(self, user_features_df):
        """
        Scoring logic:
        1. Predict Probability of Click (CTR)
        2. Predict Causal Lift (Incremental impact of recommendation)
        3. Final Score = Hybrid(CTR, Lift)
        """
        # Prepare data for XGBoost (DMatrix)
        dtest = xgb.DMatrix(user_features_df)

        # 1. CTR Prediction
        ctr_scores = self.ranker.predict(dtest)

        # 2. Uplift Prediction (T-Learner)
        # Note: The T-Learner class expects a DataFrame input
        lift_scores = self.uplift_model.predict_lift(user_features_df)

        # 3. Combine Results
        results = user_features_df.copy()
        results['predicted_ctr'] = ctr_scores
        results['predicted_uplift'] = lift_scores

        # Strategy: Target "Persuadables" (High Lift) + High Quality Items (High CTR)
        # Simple weight: 70% Lift + 30% CTR
        results['final_score'] = (0.7 * results['predicted_uplift']) + (0.3 * results['predicted_ctr'])

        return results.sort_values('final_score', ascending=False)


if __name__ == "__main__":
    # Simulation: Load some users from our feature store to score
    print(" Starting Inference Service...")
    features = pd.read_parquet(FEATURE_DATA_PATH).sample(10)  # Score 10 random users

    # Drop non-feature columns that might be in the parquet
    drop_cols = ['impression_id', 'user_id', 'item_id', 'impression_time',
                 'clicked', 'added_to_cart', 'purchased', 'variant',
                 'event_type', 'is_treated']

    # Keep IDs for display, separate features for prediction
    ids = features[['user_id', 'item_id']].reset_index(drop=True)
    scoring_data = features.drop(columns=[c for c in drop_cols if c in features.columns])

    engine = RecommendationServingEngine()
    scored_users = engine.predict(scoring_data)

    # Attach IDs back for display
    final_output = pd.concat([ids, scored_users.reset_index(drop=True)], axis=1)

    print("\n Top Recommended Users/Items:")
    print(final_output[['user_id', 'item_id', 'predicted_ctr', 'predicted_uplift', 'final_score']].head())