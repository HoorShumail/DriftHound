from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DATA_DIR = PROJECT_ROOT / "data_store" / "raw"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data_store" / "processed"
ARTIFACTS_MODELS_DIR = PROJECT_ROOT / "artifacts" / "models"
ARTIFACTS_METRICS_DIR = PROJECT_ROOT / "artifacts" / "metrics"
DRIFT_REPORTS_DIR = PROJECT_ROOT / "reports" / "drift_reports"
NUM_SAMPLES = 10000
FEATURE_COLUMNS = [f"feature_{i}" for i in range(1, 7)]
TARGET_COLUMN = "target"
DRIFT_MAGNITUDE = 0.3
RANDOM_SEED = 42
