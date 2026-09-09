"""Generate reproducible tabular data with controlled distribution drift."""

import numpy as np
import pandas as pd

from config.settings import FEATURE_COLUMNS, RANDOM_SEED


def generate_baseline_data(n_samples=10000) -> pd.DataFrame:
    """Create a baseline dataset with six numeric features and a binary target."""
    if n_samples < 1:
        raise ValueError("n_samples must be positive")

    random_state = np.random.RandomState(RANDOM_SEED)
    means = np.array([0.0, 1.0, -1.0, 2.0, 0.5, -0.5])
    standard_deviations = np.array([1.0, 1.5, 0.75, 2.0, 0.5, 1.25])
    features = random_state.normal(
        loc=means, scale=standard_deviations, size=(n_samples, len(FEATURE_COLUMNS))
    )

    data = pd.DataFrame(features, columns=FEATURE_COLUMNS)
    data["target"] = (
        data["feature_1"] + 0.5 * data["feature_2"] > 2.5
    ).astype(int)
    data["timestamp"] = pd.date_range(
        start="2024-01-01", periods=n_samples, freq="h"
    )
    return data


def inject_drift(
    df: pd.DataFrame,
    features_to_drift,
    magnitude=0.3,
    drift_type="gradual",
) -> pd.DataFrame:
    """Return a copy of ``df`` with controlled drift in selected features.

    Sudden drift shifts the final 30 percent of rows. Gradual drift applies a
    linear shift across the final 50 percent. The shift is scaled by each
    selected feature's baseline standard deviation.
    """
    if drift_type not in {"sudden", "gradual"}:
        raise ValueError("drift_type must be 'sudden' or 'gradual'")
    if magnitude < 0:
        raise ValueError("magnitude must be non-negative")

    missing_features = set(features_to_drift) - set(df.columns)
    if missing_features:
        raise ValueError(f"Unknown features: {sorted(missing_features)}")

    drifted_data = df.copy()
    drifted_data["drifted"] = False
    n_rows = len(drifted_data)
    if n_rows == 0 or not features_to_drift:
        return drifted_data

    if drift_type == "sudden":
        start = int(n_rows * 0.7)
        shift_factors = np.zeros(n_rows)
        shift_factors[start:] = magnitude
    else:
        start = int(n_rows * 0.5)
        shift_factors = np.zeros(n_rows)
        shift_factors[start:] = np.linspace(0.0, magnitude, n_rows - start)

    affected_rows = np.arange(n_rows) >= start
    for feature in features_to_drift:
        feature_std = df[feature].std()
        drifted_data.loc[:, feature] = (
            df[feature].to_numpy() + shift_factors * feature_std
        )
    drifted_data.loc[affected_rows, "drifted"] = True
    return drifted_data
