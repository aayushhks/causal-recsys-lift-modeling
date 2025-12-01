# Configuration
PYTHON := python3
PIP := pip

# Commands
.PHONY: setup data-synth data-real pipeline train infer clean help

# Default target (what happens if you just type 'make')
help:
	@echo "Available commands:"
	@echo "  make setup       - Install dependencies"
	@echo "  make data-synth  - Generate SYNTHETIC data (for testing/demo)"
	@echo "  make data-real   - Ingest RETAILROCKET data (for production)"
	@echo "  make pipeline    - Run feature engineering & processing"
	@echo "  make train       - Train XGBoost Ranker & Uplift models"
	@echo "  make infer       - Run inference prediction"
	@echo "  make all-synth   - Run full loop with Synthetic Data"
	@echo "  make all-real    - Run full loop with Real Data"

setup:
	$(PIP) install -r requirements.txt

#  Option A: Synthetic Data
data-synth:
	@echo " Generating Synthetic Data..."
	$(PYTHON) generate_synthetic_data.py
	$(PYTHON) src/pipeline/validation.py

#  Option B: Real Data
data-real:
	@echo " Ingesting RetailRocket Data..."
	$(PYTHON) src/pipeline/ingest_retailrocket.py
	$(PYTHON) src/pipeline/validation.py

#  Core Pipeline (Works for BOTH)
pipeline:
	@echo "  Running ETL Pipeline..."
	$(PYTHON) src/pipeline/data_pipeline.py
	@echo " Engineering Features..."
	$(PYTHON) src/pipeline/feature_engineering.py

train:
	@echo "  Training Ranker..."
	$(PYTHON) src/models/train_ranker.py
	@echo "  Training Uplift Model..."
	$(PYTHON) src/models/train_uplift.py

infer:
	@echo " Running Inference..."
	$(PYTHON) src/inference.py

clean:
	rm -rf data/processed/*.parquet
	rm -rf data/features/*.parquet
	rm -rf models/ranking/*.json
	rm -rf models/uplift/*.pkl

#  Meta Commands
all-synth: clean data-synth pipeline train infer
	@echo " Full Synthetic Run Complete."

all-real: clean data-real pipeline train infer
	@echo " Full Real-Data Run Complete."