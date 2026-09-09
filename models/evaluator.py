"""Metrics and model-performance comparison helpers."""

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def compute_metrics(model, X_test, y_test) -> dict:
    """Calculate common binary-classification metrics for a trained model."""
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]
    return {
        "accuracy": round(accuracy_score(y_test, predictions), 4),
        "precision": round(precision_score(y_test, predictions, zero_division=0), 4),
        "recall": round(recall_score(y_test, predictions, zero_division=0), 4),
        "f1_score": round(f1_score(y_test, predictions, zero_division=0), 4),
        "roc_auc": round(roc_auc_score(y_test, probabilities), 4),
        "confusion_matrix": confusion_matrix(y_test, predictions).tolist(),
        "classification_report": classification_report(
            y_test, predictions, zero_division=0
        ),
    }


def compare_metrics(baseline_metrics, current_metrics, thresholds=None) -> dict:
    """Compare current metrics with baseline and flag meaningful degradation."""
    thresholds = thresholds or {
        "accuracy_drop": 0.05,
        "f1_drop": 0.05,
        "roc_auc_drop": 0.03,
    }
    metric_thresholds = {
        "accuracy": thresholds.get("accuracy_drop", 0.05),
        "f1_score": thresholds.get("f1_drop", 0.05),
        "roc_auc": thresholds.get("roc_auc_drop", 0.03),
    }

    comparison = {}
    for metric, threshold in metric_thresholds.items():
        delta = round(current_metrics[metric] - baseline_metrics[metric], 4)
        comparison[metric] = {
            "delta": delta,
            "breached": delta < -threshold,
        }
    comparison["overall_degraded"] = any(
        result["breached"] for result in comparison.values()
    )
    return comparison
