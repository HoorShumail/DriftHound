"""Generate and store the Phase 1 baseline and drifted datasets."""

from config.settings import (
    ARTIFACTS_METRICS_DIR,
    ARTIFACTS_MODELS_DIR,
    DRIFT_MAGNITUDE,
    DRIFT_REPORTS_DIR,
    FEATURE_COLUMNS,
    NUM_SAMPLES,
    PROCESSED_DATA_DIR,
    RAW_DATA_DIR,
)
import pandas as pd

from data.generators.synthetic_data import generate_baseline_data, inject_drift
from data.ingestion.pipeline import ingest_data
from drift.alerts import format_alert, log_alert, should_alert
from drift.monitor import DriftMonitor
from agent.diagnostician import DriftDiagnostician
from models.baseline import run_baseline_pipeline


def main() -> None:
    baseline = generate_baseline_data(NUM_SAMPLES)
    drifted = inject_drift(
        baseline,
        features_to_drift=["feature_1", "feature_3"],
        magnitude=DRIFT_MAGNITUDE,
        drift_type="gradual",
    )

    baseline_path = ingest_data(
        baseline, RAW_DATA_DIR / "baseline.parquet", expected_columns=FEATURE_COLUMNS
    )
    drifted_path = ingest_data(
        drifted,
        PROCESSED_DATA_DIR / "drifted.parquet",
        expected_columns=FEATURE_COLUMNS,
    )

    print(f"Baseline: {baseline_path}")
    print(f"Drifted: {drifted_path}")
    print(drifted[FEATURE_COLUMNS].describe().round(3))

    baseline_result = run_baseline_pipeline(
        baseline_path,
        {
            "model_path": ARTIFACTS_MODELS_DIR / "baseline.joblib",
            "metrics_path": ARTIFACTS_METRICS_DIR / "baseline.json",
        },
    )
    print("Baseline model metrics:")
    print(baseline_result["metrics_summary"])

    reference_data = pd.read_parquet(baseline_path)
    current_data = pd.read_parquet(drifted_path)
    drift_monitor = DriftMonitor(reference_data, FEATURE_COLUMNS)
    drift_report = drift_monitor.run_full_check(current_data)
    alert_text = format_alert(drift_report)
    print(alert_text.encode("ascii", errors="replace").decode("ascii"))
    if should_alert(drift_report):
        alert_path = log_alert(alert_text, DRIFT_REPORTS_DIR)
        print(f"Alert: {alert_path}")
    print(f"Drift report: {drift_report['report_path']}")

    try:
        diagnostician = DriftDiagnostician()
        diagnosis_result = diagnostician.diagnose_and_save(
            drift_report, DRIFT_REPORTS_DIR
        )
        print("Diagnosis:")
        print(diagnosis_result["diagnosis"])
        print(f"Diagnosis file: {diagnosis_result['file_path']}")
    except (EnvironmentError, RuntimeError):
        print("Skipping AI diagnosis - set OPENAI_API_KEY in .env")


if __name__ == "__main__":
    main()
