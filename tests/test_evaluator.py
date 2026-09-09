import numpy as np

from models.evaluator import compare_metrics, compute_metrics
from models.trainer import train_model


def test_metrics_dict_has_expected_keys():
    random_state = np.random.RandomState(11)
    X = random_state.normal(size=(100, 2))
    y = (X[:, 0] > 0).astype(int)
    model = train_model(X[:80], y[:80], "logistic_regression")

    metrics = compute_metrics(model, X[80:], y[80:])

    assert {
        "accuracy",
        "precision",
        "recall",
        "f1_score",
        "roc_auc",
        "confusion_matrix",
        "classification_report",
    } <= metrics.keys()


def test_compare_metrics_flags_degradation_correctly():
    baseline = {"accuracy": 0.9, "f1_score": 0.9, "roc_auc": 0.9}
    current = {"accuracy": 0.8, "f1_score": 0.8, "roc_auc": 0.8}

    comparison = compare_metrics(baseline, current)

    assert comparison["accuracy"]["breached"] is True
    assert comparison["f1_score"]["breached"] is True
    assert comparison["roc_auc"]["breached"] is True
    assert comparison["overall_degraded"] is True


def test_identical_metrics_have_no_degradation():
    metrics = {"accuracy": 0.9, "f1_score": 0.9, "roc_auc": 0.9}

    comparison = compare_metrics(metrics, metrics)

    assert comparison["overall_degraded"] is False
