"""Script for evalution score of trained model.

Parameters
----------
validation-dataset:
    Processed validation dataset.
log-level: 
    Mention priority of logs according to severity.
log-path:
    Full path to a log file, if logs are to be written to a file.
no-console-log:
    Whether or not to write logs to the console.
"""
import argparse
import configparser
import os

import joblib
import mlflow
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# Import custom logger
from houseValuePrediction import log_configurar

# Configure default logger
logger = log_configurar.configure_logger()

# Read configuration
config = configparser.ConfigParser()
config.read("setup.cfg")

# Variable Initialization
VALIDATION_DATASET = "data/processed/test_data.csv"
MODEL_PATH = "artifacts/model.joblib"


def get_args():
    """Parse command line arugments.

    Parameters
    ----------
    None

    Returns
    -------
    argparse.Namespace
    """
    parser = argparse.ArgumentParser()

    # model arguments
    parser.add_argument(
        "-vd",
        "--validation-dataset",
        help="Processed validation dataset.",
        default=config["params"]["OUTPUT_DATA_PROCESSED_TEST"],
    )
    parser.add_argument("-mp", "--model_path", help="From Where to get model.", default=config["params"]["MODEL_PATH"])
    parser.add_argument(
        "--log-level",
        default="DEBUG",
        choices=["NOTSET", "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Mention priority of logs according to severity",
    )
    parser.add_argument(
        "-lp", "--log-path", help="Path where logs while get store", default=None,
    )
    parser.add_argument(
        "-ncl", "--no-console-log", action="store_false", help=" whether or not to write logs to the console"
    )

    # parse arugments
    return parser.parse_args()


def predict_on_test_data(model: RandomForestRegressor, X_test, y_test):
    """Predict test data on given model.

    Parameters
    ----------
    model: sklearn.ensemble.RandomForestRegressor
        Trained Random forest model with grid search.
    X_test: pandas.DataFrame
        features of testing dataset.
    y_test: pandas.DataFrame
        result of testing dataset.

    Returns
    -------
    pandas.DataFrame
        predictions.
    int
        Score of model.
    """
    logger.debug("Make prediction on validation dataset.")
    y_pred = model.predict(X_test)
    filename = str(model.__class__.__name__) + "predicted_output.csv"
    predictions = pd.DataFrame(y_pred)
    predictions.to_csv("artifacts/" + filename)

    final_mse = mean_squared_error(y_test, y_pred)
    final_rmse = np.sqrt(final_mse)

    return predictions, final_rmse


def score_model(model=None, X_test=None, y_test=None):
    """Find score of trained model.
    
    Parameters
    ----------
    model: sklearn.ensemble.RandomForestRegressor
        Trained Random forest model with grid search.
    X_test: pandas.DataFrame
        features of testing dataset.
    y_test: pandas.DataFrame
        result of testing dataset.
    
    Returns
    -------
    pandas.DataFrame
        predictions.
    int
        Score of model.
    """
    logger.debug("Loading validation dataset.")
    # load data
    test_data = pd.read_csv(VALIDATION_DATASET)
    final_model = model

    # validation dataset
    if type(X_test) == type(None):
        X_test = test_data.drop("median_house_value", axis=1)
        y_test = test_data["median_house_value"].copy()

    logger.debug("Load trained model.")
    if type(model) == type(None):
        # load, no need to initialize the loaded_rf
        final_model = joblib.load(MODEL_PATH)

    predictions, score = predict_on_test_data(final_model, X_test, y_test)
    logger.info(f"final RMSE Score: {score}")
    mlflow.log_metrics({"rmse": score})
    return predictions, score


if __name__ == "__main__":
    args = get_args()
    VALIDATION_DATASET = (
        args.validation_dataset if not VALIDATION_DATASET else config["params"]["OUTPUT_DATA_PROCESSED_TEST"]
    )
    MODEL_PATH = args.model_path if not MODEL_PATH else config["params"]["MODEL_PATH"]

    # Configure logger
    logger = log_configurar.configure_logger(
        log_file=args.log_path, console=args.no_console_log, log_level=args.log_level
    )

    logger.debug("Start Scoring Phase =======")
    score_model()
    logger.debug("End Scoring Phase =======")
