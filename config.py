import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

RAW_DATA_PATH = Path("data/raw/creditcard.csv")
BATCHES_DIR = Path("data/batches")
FROZEN_HOLDOUT_PATH = Path("data/raw/frozen_holdout.parquet")
MANIFEST_PATH = Path("data/batches/manifest.json")

SEED = 42
N_BATCHES = 100
HOLDOUT_FRACTION = 0.15

# feature drift: persistent shift from this batch onward
FEATURE_DRIFT_START_BATCH = 30
FEATURE_DRIFT_FEATURES = ["V1", "V2", "V3"]
FEATURE_DRIFT_SHIFT = 2.0
FEATURE_DRIFT_SCALE = 1.5

# concept drift: fraud rows pulled toward legit centroid, temporary window
CONCEPT_DRIFT_BATCHES = list(range(45, 55))
CONCEPT_DRIFT_ALPHA = 0.6

# bootstrap: first N batches are pretrain material, not replayed as live traffic
PRETRAIN_BATCHES = 10
STREAM_START_BATCH = PRETRAIN_BATCHES

LABEL_DELAY_BATCHES = 5

MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "")
MODEL_NAME = "fraud-xgb"

FEATURE_COLUMNS = [f"V{i}" for i in range(1, 29)] + ["Amount"]
PREDICTION_THRESHOLD = 0.5

FEATURE_DRIFT_SHARE_THRESHOLD = 0.3
PERFORMANCE_WINDOW_BATCHES = 10
MIN_FRAUD_COUNT_FOR_PERF_CHECK = 20
RECALL_RELATIVE_DROP_THRESHOLD = 0.15
PRECISION_RELATIVE_DROP_THRESHOLD = 0.15
HOLDOUT_TOLERANCE = 0.05

GROQ_MODEL = "openai/gpt-oss-120b"

RECENT_SAMPLE_WEIGHT = 3.0

REPLAY_CHUNK_SIZE = 500