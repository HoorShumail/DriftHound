import pandas as pd

from drift.alerts import format_alert, should_alert
from drift.monitor import DriftMonitor


def make_data():
    reference = pd.DataFrame({"feature_1": range(100), "feature_2": range(100)})
    current = reference.copy()
    current["feature_1"] = current["feature_1"] + 100
    return reference, current


def test_detects_drift_on_drifted_data(tmp_path, monkeypatch):
    reference, current = make_data()
    monitor = DriftMonitor(reference, ["feature_1", "feature_2"], methods=["ks", "psi"])
    monkeypatch.setattr("drift.monitor.Path", lambda *args: tmp_path)

    report = monitor.run_full_check(current)

    assert "feature_1" in report["summary"]["drifted_features"]
    assert should_alert(report)
    assert "Drift detected" in format_alert(report)


def test_no_drift_on_clean_baseline(tmp_path, monkeypatch):
    reference, _ = make_data()
    monitor = DriftMonitor(reference, ["feature_1", "feature_2"], methods=["ks", "psi"])
    monkeypatch.setattr("drift.monitor.Path", lambda *args: tmp_path)

    report = monitor.run_full_check(reference.copy())

    assert report["summary"]["drifted_features"] == []
    assert should_alert(report) is False


def test_report_has_required_fields(tmp_path, monkeypatch):
    reference, current = make_data()
    monitor = DriftMonitor(reference, ["feature_1", "feature_2"])
    monkeypatch.setattr("drift.monitor.Path", lambda *args: tmp_path)

    report = monitor.run_full_check(current)

    assert {"timestamp", "summary", "results", "report_path"} <= report.keys()
    assert {"features_checked", "drifted_features", "per_feature"} <= report["summary"].keys()
