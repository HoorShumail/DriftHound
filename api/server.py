"""FastAPI endpoints for the DriftHound pipeline."""

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from agent.diagnostician import DriftDiagnostician
from config.settings import (
    ARTIFACTS_METRICS_DIR,
    ARTIFACTS_MODELS_DIR,
    DRIFT_MAGNITUDE,
    DRIFT_REPORTS_DIR,
    FEATURE_COLUMNS,
    NUM_SAMPLES,
    PROCESSED_DATA_DIR,
    RAW_DATA_DIR,
    TARGET_COLUMN,
)
from data.generators.synthetic_data import generate_baseline_data, inject_drift
from data.ingestion.pipeline import ingest_data
from drift.monitor import DriftMonitor
from models.baseline import run_baseline_pipeline
from models.evaluator import compare_metrics, compute_metrics
from models.trainer import load_model


app = FastAPI(title="DriftHound API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class DiagnosisRequest(BaseModel):
    drift_report: dict | None = None
    metrics: dict | None = None


@app.get("/")
def root():
    """Provide a small welcome response for the API base URL."""
    return {"name": "DriftHound API", "status": "ok", "docs": "/docs"}


def _latest_file(directory, pattern):
    files = sorted(Path(directory).glob(pattern), key=lambda path: path.stat().st_mtime)
    return files[-1] if files else None


def _read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def _latest_report():
    path = _latest_file(DRIFT_REPORTS_DIR, "drift_report_*.json")
    if path is None:
        raise HTTPException(status_code=404, detail="No drift reports found")
    return _read_json(path)


def _baseline_metrics():
    path = _latest_file(ARTIFACTS_METRICS_DIR, "*.json")
    if path is None:
        raise HTTPException(status_code=404, detail="Baseline metrics not found")
    payload = _read_json(path)
    return payload.get("metrics", payload)


@app.get("/api/status")
def status():
    """Return the completed pipeline phases based on persisted artifacts."""
    phases = ["phase_1"] if (RAW_DATA_DIR / "baseline.parquet").exists() else []
    if (ARTIFACTS_MODELS_DIR / "baseline.joblib").exists():
        phases.append("phase_2")
    if _latest_file(DRIFT_REPORTS_DIR, "drift_report_*.json"):
        phases.append("phase_3")
    if _latest_file(DRIFT_REPORTS_DIR, "diagnosis_*.txt"):
        phases.append("phase_4")
    return {"status": "ok", "phases_completed": phases}


@app.get("/api/drift-report")
def drift_report():
    """Return the most recent persisted drift report."""
    return _latest_report()


@app.get("/api/metrics")
def metrics():
    """Return the persisted baseline metrics."""
    return _baseline_metrics()


@app.get("/api/metrics/comparison")
def metrics_comparison():
    """Evaluate the saved model on drifted data and compare its metrics."""
    baseline = _baseline_metrics()
    model_path = ARTIFACTS_MODELS_DIR / "baseline.joblib"
    drifted_path = PROCESSED_DATA_DIR / "drifted.parquet"
    if not model_path.exists() or not drifted_path.exists():
        raise HTTPException(status_code=404, detail="Baseline model or drifted data not found")

    model, scaler = load_model(model_path)
    data = pd.read_parquet(drifted_path)
    current_metrics = compute_metrics(
        model,
        scaler.transform(data[FEATURE_COLUMNS]),
        data[TARGET_COLUMN],
    )
    return compare_metrics(baseline, current_metrics)


@app.post("/api/diagnose")
def diagnose(request: DiagnosisRequest | None = None):
    """Generate an AI diagnosis from supplied or latest persisted artifacts."""
    request = request or DiagnosisRequest()
    report = request.drift_report or _latest_report()
    try:
        diagnosis = DriftDiagnostician().diagnose(report, request.metrics)
    except (EnvironmentError, RuntimeError) as error:
        raise HTTPException(status_code=503, detail=str(error)) from error
    except Exception as error:
        raise HTTPException(
            status_code=502,
            detail=f"OpenAI diagnosis request failed: {error}",
        ) from error
    return {"diagnosis": diagnosis, "timestamp": datetime.now(timezone.utc).isoformat()}


@app.post("/api/run-pipeline")
def run_pipeline():
    """Run Phases 1–3 and return the generated artifact summary."""
    baseline = generate_baseline_data(NUM_SAMPLES)
    drifted = inject_drift(
        baseline, ["feature_1", "feature_3"], DRIFT_MAGNITUDE, "gradual"
    )
    baseline_path = ingest_data(
        baseline, RAW_DATA_DIR / "baseline.parquet", expected_columns=FEATURE_COLUMNS
    )
    drifted_path = ingest_data(
        drifted,
        PROCESSED_DATA_DIR / "drifted.parquet",
        expected_columns=FEATURE_COLUMNS,
    )
    model_result = run_baseline_pipeline(
        baseline_path,
        {
            "model_path": ARTIFACTS_MODELS_DIR / "baseline.joblib",
            "metrics_path": ARTIFACTS_METRICS_DIR / "baseline.json",
        },
    )
    report = DriftMonitor(baseline, FEATURE_COLUMNS).run_full_check(drifted)
    return {
        "baseline_path": baseline_path,
        "drifted_path": drifted_path,
        "model_path": model_result["model_path"],
        "metrics_path": model_result["metrics_path"],
        "drift_report_path": report["report_path"],
    }


@app.get("/api/feature-distributions")
def feature_distributions():
    """Return up to 500 values per feature from baseline and drifted data."""
    baseline_path = RAW_DATA_DIR / "baseline.parquet"
    drifted_path = PROCESSED_DATA_DIR / "drifted.parquet"
    if not baseline_path.exists() or not drifted_path.exists():
        raise HTTPException(status_code=404, detail="Baseline or drifted data not found")
    baseline = pd.read_parquet(baseline_path)
    drifted = pd.read_parquet(drifted_path)
    step = max(1, len(baseline) // 500)
    return {
        feature: {
            "baseline": baseline[feature].iloc[::step].head(500).tolist(),
            "current": drifted[feature].iloc[::step].head(500).tolist(),
        }
        for feature in FEATURE_COLUMNS
    }
