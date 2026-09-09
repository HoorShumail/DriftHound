"""Train, evaluate, and persist a baseline classification model."""

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from config.settings import FEATURE_COLUMNS, TARGET_COLUMN
from models.evaluator import compute_metrics
from models.trainer import prepare_features, save_model, train_model


def capture_baseline(metrics_dict, output_path) -> str:
    """Save metrics and a UTC capture timestamp as JSON."""
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "metrics": metrics_dict,
    }
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return str(output)


def load_baseline(path) -> dict:
    """Load a baseline JSON document and return it as a dictionary."""
    return json.loads(Path(path).read_text(encoding="utf-8"))


def run_baseline_pipeline(data_path, config) -> dict:
    """Train, evaluate, and persist a baseline model from a parquet dataset."""
    data = pd.read_parquet(data_path)
    feature_columns = config.get("feature_columns", FEATURE_COLUMNS)
    target_column = config.get("target_column", TARGET_COLUMN)
    model_type = config.get("model_type", "random_forest")
    model_path = Path(config["model_path"])
    metrics_path = Path(config["metrics_path"])

    X_train, X_test, y_train, y_test, scaler = prepare_features(
        data,
        feature_columns,
        target_column,
        test_size=config.get("test_size", 0.2),
    )
    model = train_model(X_train, y_train, model_type=model_type)
    metrics = compute_metrics(model, X_test, y_test)
    save_model(model, scaler, model_path)
    capture_baseline(metrics, metrics_path)
    return {
        "model_path": str(model_path),
        "metrics_path": str(metrics_path),
        "metrics_summary": metrics,
    }
