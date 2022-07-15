"""Script for training preprocssed data and generating ML model.

Parameters
----------
train-dataset:
    Processed training dataset.
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
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV

# Import custom logger
from src.houseValuePrediction import log_configurar

# Configure default logger
logger = log_configurar.configure_logger()

# Read configuration
config = configparser.ConfigParser()
config.read("setup.cfg")

# Variable Initialization
TRAIN_DATASET = "data/processed/train_data.csv"


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
        "-td",
        "--train-dataset",
        help="Processed training dataset.",
        default=config["params"]["OUTPUT_DATA_PROCESSED_TRAIN"],
    )
    parser.add_argument("-mp", "--model_path", help="Where to store model.", default=config["params"]["MODEL_PATH"])
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


def random_forest_grid_search(X_train, y_train):
    """This function builds random forest model using grid search.
    
    Parameters
    ----------
    X_train: pandas.DataFrame
        Features of train dataset
    y_train: pandas.DataFrame
        Labels for train dataset
    
    Returns
    -------
    sklearn.ensemble.RandomForestRegressor
        Best model with good accuracy.
    """
    # Grid search
    param_grid = [
        # try 12 (3×4) combinations of hyperparameters
        {"n_estimators": [3, 10, 30], "max_features": [2, 4, 6, 8]},
        # then try 6 (2×3) combinations with bootstrap set as False
        {"bootstrap": [False], "n_estimators": [3, 10], "max_features": [2, 3, 4]},
    ]

    logger.debug("Model training with Grid search.")
    # model initialization

    forest_reg = RandomForestRegressor(random_state=42)
    # train across 5 folds, that's a total of (12+6)*5=90 rounds of training
    grid_search = GridSearchCV(forest_reg, param_grid, cv=5, scoring="neg_mean_squared_error", return_train_score=True)
    grid_search.fit(X_train, y_train)

    grid_search.best_params_
    cvres = grid_search.cv_results_
    for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
        print(np.sqrt(-mean_score), params)

    final_model = grid_search.best_estimator_
    return final_model


def train_model(X_train=None, y_train=None, model_dst_path: str = None):
    """This function builds random forest model using grid search and store model in deault path.
    
    Parameters
    ----------
    X_train: pandas.DataFrame
        Features of train dataset
    y_train: pandas.DataFrame
        Labels for train dataset
    model_dst_path: string
        Provide path where ML model will get store
    
    Returns
    -------
    sklearn.ensemble.RandomForestRegressor
        Best model with good accuracy.
    """
    # load data
    logger.debug("Loading training dataset.")
    train_data = pd.read_csv(TRAIN_DATASET)

    if type(X_train) == type(None):
        # training dataset
        X_train = train_data.drop("median_house_value", axis=1)
        y_train = train_data["median_house_value"].copy()

    final_model = random_forest_grid_search(X_train, y_train)

    model_path = model_dst_path if model_dst_path else args.model_path
    logger.info(f'Stored train model at: "{model_path}"')
    joblib.dump(final_model, model_path)

    return final_model


if __name__ == "__main__":
    args = get_args()
    TRAIN_DATASET = args.train_dataset if not TRAIN_DATASET else config["params"]["OUTPUT_DATA_PROCESSED_TRAIN"]

    # Configure logger
    logger = log_configurar.configure_logger(
        log_file=args.log_path, console=args.no_console_log, log_level=args.log_level
    )

    logger.debug("Start Training Phase =======")
    train_model()
    logger.debug("End Training Phase =======")
