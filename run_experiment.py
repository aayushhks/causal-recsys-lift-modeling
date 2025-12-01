import pandas as pd
import joblib
from src.evaluation.metrics import calculate_qini
import os


def evaluate_model():
    print(" Running Comprehensive Model Evaluation...")

    # Load Test Data (Simulation: re-using training data for demo)
    # In real prod, you'd load a holdout set
    df = pd.read_parquet("data/features/training_set.parquet")

    # Load Uplift Model
    model_path = "models/uplift/uplift_meta_learner.pkl"
    if not os.path.exists(model_path):
        print(" Model not found. Run 'make train' first.")
        return

    learner = joblib.load(model_path)

    # Prepare Features
    drop_cols = ['impression_id', 'user_id', 'item_id', 'impression_time',
                 'clicked', 'added_to_cart', 'purchased', 'variant', 'event_type', 'is_treated']
    features = [c for c in df.columns if c not in drop_cols]

    # Predict Uplift
    print("    Scoring users...")
    X = df[features]
    uplift_preds = learner.predict_lift(X)

    # Calculate Qini
    # We need: Outcome (clicked), Treatment (is_treated)
    # Ensure 'is_treated' exists (created in feature engineering or synthetic gen)
    if 'variant' in df.columns:
        is_treated = (df['variant'] == 'Treatment').astype(int)
    else:
        # Fallback for synthetic
        is_treated = np.random.randint(0, 2, len(df))

    calculate_qini(
        y_true=df['clicked'],
        uplift_score=uplift_preds,
        treatment=is_treated,
        plot=True
    )


if __name__ == "__main__":
    evaluate_model()