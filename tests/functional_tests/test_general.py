import pandas as pd
import pytest
import pytest_check as check

from src.ingest_data import data_collection_preprocessing
from src.score import predict_on_test_data
from src.train import random_forest_grid_search


@pytest.fixture
def data_preparation():
    train_data, test_data = data_collection_preprocessing()

    # features and results splits
    X_train = train_data.drop("median_house_value", axis=1)
    y_train = train_data["median_house_value"].copy()
    X_test = test_data.drop("median_house_value", axis=1)
    y_test = test_data["median_house_value"].copy()

    return X_train, y_train, X_test, y_test


@pytest.fixture
def random_forest_prediction(data_preparation):
    X_train, y_train, X_test, y_test = data_preparation

    rf_model = random_forest_grid_search(X_train, y_train)
    y_pred, score = predict_on_test_data(rf_model, X_test, y_test)
    return X_test, y_pred


@pytest.fixture
def return_models(data_preparation):
    X_train, y_train, X_test, y_test = data_preparation
    rf_model = random_forest_grid_search(X_train, y_train)
    return [rf_model]


def test_data_leak(data_preparation):
    """Test data leakage."""
    X_train, y_train, X_test, y_test = data_preparation
    concat_df = pd.concat([X_train, X_test])
    concat_df.drop_duplicates(inplace=True)
    assert concat_df.shape[0] == X_train.shape[0] + X_test.shape[0]


def test_predicted_output_shape(random_forest_prediction):
    """Predicted output shape validation."""
    print("Random Forest")
    X_test, y_pred = random_forest_prediction
    check.equal(y_pred.shape[0], X_test.shape[0])
