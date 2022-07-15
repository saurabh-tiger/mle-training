""" Script for training preprocssed data and generating ML model."""
import argparse
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV

# Import custom logger
from src import log_configurar


def get_args():
    """Parse command line arugments.

    Parameters
    ----------
    None

    Returns
    -------
    <class 'argparse.Namespace'>
        parsed arguments
    """
    parser = argparse.ArgumentParser()

    # model arguments
    parser.add_argument(
        "-td",
        "--train_dataset",
        help="Processed training dataset.",
        default=os.path.join("data", "processed", "train_data.csv"),
    )
    parser.add_argument("-mp", "--model_path", help="Where to store model.", default=os.path.join("artifacts"))
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
    try:
        return parser.parse_args()
    except SystemExit as se:
        print(f"Unable to parse default argparse arguments. It may only occur for pytest. Here is error: {se}")
        logger.warning(f"Unable to parse default argparse arguments. It may only occur for pytest. Here is error: {se}")
        return parser.parse_args([])


args = get_args()

# Configure logger
logger = log_configurar.configure_logger(log_file=args.log_path, console=args.no_console_log, log_level=args.log_level)


def random_forest_grid_search(X_train, y_train):
    param_grid = [
        # try 12 (3×4) combinations of hyperparameters
        {"n_estimators": [3, 10, 30], "max_features": [2, 4, 6, 8]},
        # then try 6 (2×3) combinations with bootstrap set as False
        {"bootstrap": [False], "n_estimators": [3, 10], "max_features": [2, 3, 4]},
    ]

    logger.debug("Model training with Grid search.")
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


def train():
    # load data
    logger.debug("Loading training dataset.")
    train_data = pd.read_csv(args.train_dataset)

    # training dataset
    housing_targets = train_data.drop("median_house_value", axis=1)  # drop labels for training set
    housing_labels = train_data["median_house_value"].copy()

    final_model = random_forest_grid_search(housing_targets, housing_labels)

    model_path = os.path.join(args.model_path, "model.joblib")
    logger.info(f'Stored train model at: "{model_path}"')
    joblib.dump(final_model, model_path)


if __name__ == "__main__":
    logger.debug("Start Training Phase =======")
    train()
    logger.debug("End Training Phase =======")
