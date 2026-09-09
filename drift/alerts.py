"""Human-readable drift alerts and alert log persistence."""

from datetime import datetime, timezone
from pathlib import Path


def format_alert(drift_report) -> str:
    """Format a drift report as a short human-readable alert."""
    summary = drift_report["summary"]
    drifted_features = summary["drifted_features"]
    indicator = "⚠️" if drifted_features else "✅"
    status = "Drift detected" if drifted_features else "No drift detected"
    return (
        f"{indicator} {status}\n"
        f"Features checked: {summary['features_checked']}\n"
        f"Drifted features: {', '.join(drifted_features) if drifted_features else 'None'}"
    )


def should_alert(drift_report, min_features_drifted=1) -> bool:
    """Return whether at least the requested number of features drifted."""
    return len(drift_report["summary"]["drifted_features"]) >= min_features_drifted


def log_alert(alert_text, output_dir) -> str:
    """Write an alert to a timestamped text file and return its path."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    filename = datetime.now(timezone.utc).strftime("alert_%Y%m%dT%H%M%S%fZ.txt")
    path = output / filename
    path.write_text(alert_text, encoding="utf-8")
    return str(path)
