"""Statistical tests for detecting feature distribution drift."""

import numpy as np
from scipy.stats import ks_2samp, wasserstein_distance


def _result(statistic, p_value, drift_detected, method):
    return {
        "statistic": float(statistic),
        "p_value": float(p_value),
        "drift_detected": bool(drift_detected),
        "method": method,
    }


def ks_test(reference, current, threshold=0.05):
    """Use a two-sample KS test to compare the full distributions."""
    statistic, p_value = ks_2samp(reference, current)
    return _result(statistic, p_value, p_value < threshold, "ks")


def psi_test(reference, current, bins=10, threshold=0.2):
    """Measure population stability using proportions in shared bins."""
    reference = np.asarray(reference, dtype=float)
    current = np.asarray(current, dtype=float)
    minimum = min(reference.min(), current.min())
    maximum = max(reference.max(), current.max())
    if minimum == maximum:
        statistic = 0.0
    else:
        edges = np.linspace(minimum, maximum, bins + 1)
        reference_counts, _ = np.histogram(reference, bins=edges)
        current_counts, _ = np.histogram(current, bins=edges)
        epsilon = 1e-4
        reference_proportions = reference_counts / len(reference)
        current_proportions = current_counts / len(current)
        reference_proportions = np.maximum(reference_proportions, epsilon)
        current_proportions = np.maximum(current_proportions, epsilon)
        statistic = np.sum(
            (current_proportions - reference_proportions)
            * np.log(current_proportions / reference_proportions)
        )
    return _result(statistic, 0.0, statistic >= threshold, "psi")


def wasserstein_test(reference, current, threshold=0.1):
    """Measure the normalized distance between two one-dimensional distributions."""
    reference = np.asarray(reference, dtype=float)
    current = np.asarray(current, dtype=float)
    statistic = wasserstein_distance(reference, current)
    reference_std = np.std(reference)
    normalized_statistic = statistic / reference_std if reference_std > 0 else statistic
    return _result(
        normalized_statistic,
        0.0,
        normalized_statistic >= threshold,
        "wasserstein",
    )


def mean_shift_test(reference, current, threshold=2.0):
    """Measure the absolute difference between means in reference-standard-deviation units."""
    reference = np.asarray(reference, dtype=float)
    current = np.asarray(current, dtype=float)
    reference_std = np.std(reference)
    statistic = (
        abs(np.mean(current) - np.mean(reference)) / reference_std
        if reference_std > 0
        else abs(np.mean(current) - np.mean(reference))
    )
    return _result(statistic, 0.0, statistic >= threshold, "mean_shift")
