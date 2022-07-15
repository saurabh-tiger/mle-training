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
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error

# Import custom logger
from src import log_configurar

# Configure default logger
logger = log_configurar.configure_logger()

# Read configuration
config = configparser.ConfigParser()
config.read("setup.cfg")


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
        default=config["params"]["OUTPUT_DATA_PROCESSED_TRAIN"],
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


def predict_on_test_data(model, X_test, y_test):
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


def score():
    """Find score of trained model.
    
    Parameters
    ----------
    None
    
    Returns
    -------
    int
        Score of model
    """
    logger.debug("Loading validation dataset.")
    # load data
    test_data = pd.read_csv(args.validation_dataset)

    # validation dataset
    X_test = test_data.drop("median_house_value", axis=1)
    y_test = test_data["median_house_value"].copy()

    logger.debug("Load trained model.")
    # load, no need to initialize the loaded_rf
    final_model = joblib.load(args.model_path)

    predictions, score = predict_on_test_data(final_model, X_test, y_test)
    logger.info(f"final RMSE Score: {score}")
    return predictions, score


if __name__ == "__main__":
    args = get_args()

    # Configure logger
    logger = log_configurar.configure_logger(
        log_file=args.log_path, console=args.no_console_log, log_level=args.log_level
    )

    logger.debug("Start Scoring Phase =======")
    score()
    logger.debug("End Scoring Phase =======")
