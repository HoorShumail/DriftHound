import numpy as np

from data.generators.synthetic_data import generate_baseline_data, inject_drift


def test_correct_shape_and_columns():
    data = generate_baseline_data(100)

    assert data.shape == (100, 8)
    assert list(data.columns) == [
        "feature_1",
        "feature_2",
        "feature_3",
        "feature_4",
        "feature_5",
        "feature_6",
        "target",
        "timestamp",
    ]


def test_target_is_binary():
    data = generate_baseline_data()

    assert set(data["target"].unique()) <= {0, 1}


def test_sudden_drift_shifts_mean():
    baseline = generate_baseline_data(1000)
    drifted = inject_drift(baseline, ["feature_1"], magnitude=0.3, drift_type="sudden")

    assert drifted.iloc[-300:]["feature_1"].mean() > baseline.iloc[-300:]["feature_1"].mean()


def test_gradual_drift_shifts_mean():
    baseline = generate_baseline_data(1000)
    drifted = inject_drift(baseline, ["feature_1"], magnitude=0.3, drift_type="gradual")

    assert drifted.iloc[-500:]["feature_1"].mean() > baseline.iloc[-500:]["feature_1"].mean()


def test_drifted_column_exists():
    drifted = inject_drift(generate_baseline_data(100), ["feature_1"])

    assert "drifted" in drifted.columns
    assert drifted["drifted"].dtype == bool


def test_undrifted_features_unchanged():
    baseline = generate_baseline_data(100)
    drifted = inject_drift(baseline, ["feature_1"], drift_type="sudden")

    for feature in ["feature_2", "feature_3", "feature_4", "feature_5", "feature_6"]:
        assert np.array_equal(baseline[feature].to_numpy(), drifted[feature].to_numpy())
