"""Coordinate feature-level drift checks and report generation."""

import json
from datetime import datetime, timezone
from pathlib import Path

from drift.detectors import ks_test, mean_shift_test, psi_test, wasserstein_test


class DriftMonitor:
    """Run configured statistical drift checks for dataframe columns."""

    def __init__(self, reference_data, feature_columns, methods=None):
        self.reference_data = reference_data
        self.feature_columns = list(feature_columns)
        self.methods = methods or ["ks", "psi", "wasserstein"]

    def check_drift(self, current_data):
        """Return detector results grouped by feature and method."""
        detector_map = {
            "ks": ks_test,
            "psi": psi_test,
            "wasserstein": wasserstein_test,
            "mean_shift": mean_shift_test,
        }
        unknown_methods = set(self.methods) - set(detector_map)
        if unknown_methods:
            raise ValueError(f"Unknown drift methods: {sorted(unknown_methods)}")

        results = {}
        for feature in self.feature_columns:
            results[feature] = {}
            for method in self.methods:
                results[feature][method] = detector_map[method](
                    self.reference_data[feature], current_data[feature]
                )
        return results

    def generate_report(self, drift_results):
        """Summarize results and save a timestamped JSON drift report."""
        drifted_features = [
            feature
            for feature, methods in drift_results.items()
            if any(result["drift_detected"] for result in methods.values())
        ]
        report = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "summary": {
                "features_checked": len(drift_results),
                "drifted_features": drifted_features,
                "per_feature": {
                    feature: {
                        "drift_detected": feature in drifted_features,
                        "methods": methods,
                    }
                    for feature, methods in drift_results.items()
                },
            },
            "results": drift_results,
        }
        output_dir = Path(__file__).resolve().parent.parent / "reports" / "drift_reports"
        output_dir.mkdir(parents=True, exist_ok=True)
        filename = datetime.now(timezone.utc).strftime("drift_report_%Y%m%dT%H%M%S%fZ.json")
        report_path = output_dir / filename
        report["report_path"] = str(report_path)
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        return report

    def run_full_check(self, current_data):
        """Run all checks and immediately generate the persisted report."""
        return self.generate_report(self.check_drift(current_data))
