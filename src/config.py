from pathlib import Path

# Project Root (calculated relative to this file)
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Data Paths
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_PATH = DATA_DIR / "raw" / "events.csv"
PROCESSED_DATA_PATH = DATA_DIR / "processed" / "impressions.parquet"
FEATURE_DATA_PATH = DATA_DIR / "features" / "training_set.parquet"

# Model Artifacts
MODEL_DIR = PROJECT_ROOT / "models"
RANKER_MODEL_PATH = MODEL_DIR / "ranking" / "xgb_ranker.json"
UPLIFT_MODEL_PATH = MODEL_DIR / "uplift" / "uplift_meta_learner.pkl"

# Experiment Settings
EXPERIMENT_CONFIG = {
    "n_variants": 2,
    "confidence_level": 0.95,
    "min_sample_size": 1000,
    "uplift_threshold": 0.01  # Minimum 1% lift to declare winner
}

# Columns to exclude from training
DROP_COLS = [
    'impression_id', 'user_id', 'item_id', 'impression_time',
    'clicked', 'added_to_cart', 'purchased', 'variant',
    'event_type', 'is_treated', 'transaction_id'
]