"""Model preparation, training, and persistence helpers."""

import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def prepare_features(df, feature_columns, target_column, test_size=0.2):
    """Split features and target, fitting a scaler only on training data."""
    X = df[feature_columns]
    y = df[target_column]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler


def train_model(X_train, y_train, model_type="random_forest"):
    """Train and return one of the supported scikit-learn classifiers."""
    if model_type == "random_forest":
        model = RandomForestClassifier(random_state=42)
    elif model_type == "logistic_regression":
        model = LogisticRegression(random_state=42)
    else:
        raise ValueError(
            "model_type must be 'random_forest' or 'logistic_regression'"
        )
    return model.fit(X_train, y_train)


def save_model(model, scaler, path) -> str:
    """Save a trained model and its feature scaler in one joblib file."""
    output_path = str(path)
    joblib.dump({"model": model, "scaler": scaler}, output_path)
    return output_path


def load_model(path):
    """Load and return the model and scaler saved by ``save_model``."""
    saved = joblib.load(path)
    return saved["model"], saved["scaler"]
