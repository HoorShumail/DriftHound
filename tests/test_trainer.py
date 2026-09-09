import numpy as np
import pandas as pd

from models.trainer import (
    load_model,
    prepare_features,
    save_model,
    train_model,
)


def make_data():
    random_state = np.random.RandomState(10)
    features = random_state.normal(size=(100, 2))
    data = pd.DataFrame(features, columns=["a", "b"])
    data["target"] = (data["a"] + data["b"] > 0).astype(int)
    return data


def test_model_trains_and_predicts_correct_shape():
    data = make_data()
    X_train, X_test, y_train, y_test, _ = prepare_features(
        data, ["a", "b"], "target"
    )
    model = train_model(X_train, y_train)

    assert model.predict(X_test).shape == y_test.shape


def test_save_load_roundtrip_works(tmp_path):
    data = make_data()
    X_train, X_test, y_train, _, scaler = prepare_features(data, ["a", "b"], "target")
    model = train_model(X_train, y_train)
    path = tmp_path / "model.joblib"

    assert save_model(model, scaler, path) == str(path)
    loaded_model, loaded_scaler = load_model(path)
    np.testing.assert_array_equal(model.predict(X_test), loaded_model.predict(X_test))
    np.testing.assert_allclose(scaler.transform(data[["a", "b"]]), loaded_scaler.transform(data[["a", "b"]]))


def test_both_model_types_work():
    data = make_data()
    X_train, X_test, y_train, _, _ = prepare_features(data, ["a", "b"], "target")

    for model_type in ["random_forest", "logistic_regression"]:
        model = train_model(X_train, y_train, model_type)
        assert len(model.predict(X_test)) == len(X_test)
