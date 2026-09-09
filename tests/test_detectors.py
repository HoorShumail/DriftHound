import numpy as np

from drift.detectors import ks_test, mean_shift_test, psi_test, wasserstein_test


def test_ks_detects_drift():
    random_state = np.random.RandomState(20)
    reference = random_state.normal(0, 1, 500)
    current = random_state.normal(3, 1, 500)

    result = ks_test(reference, current)

    assert result["drift_detected"] is True
    assert result["p_value"] < 0.05


def test_ks_detects_no_drift():
    random_state = np.random.RandomState(21)
    reference = random_state.normal(0, 1, 500)
    current = random_state.normal(0, 1, 500)

    assert ks_test(reference, current)["drift_detected"] is False


def test_psi_is_near_zero_for_identical_distribution():
    values = np.linspace(-2, 2, 500)

    result = psi_test(values, values)

    assert result["statistic"] == 0.0
    assert result["drift_detected"] is False


def test_all_detectors_return_expected_keys():
    reference = np.arange(100, dtype=float)
    current = reference + 1
    expected_keys = {"statistic", "p_value", "drift_detected", "method"}

    for detector in [ks_test, psi_test, wasserstein_test, mean_shift_test]:
        assert set(detector(reference, current)) == expected_keys
