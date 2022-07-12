""" Script for evalution score of trained model."""
import argparse
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Import custom logger
import log_configurar

# Parse arugments
parser = argparse.ArgumentParser()
parser.add_argument(
    "-vd",
    "--validation_dataset",
    help="Processed validation dataset.",
    default=os.path.join("data", "processed", "test.csv"),
)
parser.add_argument(
    "-mp", "--model_path", help="From Where to get model.", default=os.path.join("artifacts", "model.joblib")
)
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
args = parser.parse_args()

# Configure logger
logger = log_configurar.configure_logger(log_file=args.log_path, console=args.no_console_log, log_level=args.log_level,)


def score():
    logger.debug("Loading validation dataset.")
    # load data
    test_data = pd.read_csv(args.validation_dataset)

    # validation dataset
    X_test_prepared = test_data.drop("median_house_value", axis=1)
    y_test = test_data["median_house_value"].copy()

    logger.debug("Load trained model.")
    # load, no need to initialize the loaded_rf
    final_model = joblib.load(args.model_path)

    logger.debug("Make prediction on validation dataset.")
    final_predictions = final_model.predict(X_test_prepared)
    final_mse = mean_squared_error(y_test, final_predictions)
    final_rmse = np.sqrt(final_mse)
    logger.info(f"final RMSE Score: {final_rmse}")


if __name__ == "__main__":
    logger.debug("Start Scoring Phase =======")
    score()
    logger.debug("End Scoring Phase =======")
