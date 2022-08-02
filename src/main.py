"""Workflow for running data collection, Preprocessing, training and scoring phases."""

import argparse
import configparser

import mlflow

# Import custom logger
from houseValuePrediction import log_configurar
from houseValuePrediction.ingest_data import data_collection_preprocessing
from houseValuePrediction.score import score_model
from houseValuePrediction.train import train_model

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
        "-op",
        "--output-path",
        help="Output folder to store downloaded data.",
        default=config["params"]["OUTPUT_DATA_PROCESSED"],
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
        "-ncl", "--no-console-log", action="store_false", help="Whether or not to write logs to the console"
    )

    # parse arugments
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()

    # Configure logger
    logger = log_configurar.configure_logger(
        log_file=args.log_path, console=args.no_console_log, log_level=args.log_level
    )
    mlflow.set_tracking_uri(config["params"]["MLFLOW_FOLDER"])
    # mlflow.set_tracking_uri(config["params"]["MLFLOW_REMOTE_SERVER_URI"])  # or set the MLFLOW_TRACKING_URI in the env
    with mlflow.start_run(run_name="PARENT_RUN") as parent_run:
        mlflow.log_param("parent", "yes")
        logger.debug("================ Start of ML Workflow ================")

        with mlflow.start_run(run_name="DATA_PREPROCESSING", nested=True) as data_preprocessing:
            mlflow.log_param("child", "yes")
            train_data, test_data = data_collection_preprocessing(args.output_path)
            # features and results splits
            X_train = train_data.drop("median_house_value", axis=1)
            y_train = train_data["median_house_value"].copy()
            X_test = test_data.drop("median_house_value", axis=1)
            y_test = test_data["median_house_value"].copy()
            mlflow.log_param("X_train.shape", X_train.shape)
            mlflow.log_param("y_train.shape", y_train.shape)
            mlflow.log_param("X_test.shape", X_test.shape)
            mlflow.log_param("y_test.shape", y_test.shape)

        with mlflow.start_run(run_name="MODEL_TRAINING", nested=True) as model_training:
            mlflow.log_param("child", "yes")
            rf_model = train_model(X_train=X_train, y_train=y_train, model_dst_path=args.model_path)
            mlflow.log_param("feature importances", str(rf_model.feature_importances_))

        with mlflow.start_run(run_name="MODEL_SCORING", nested=True) as model_scoring:
            mlflow.log_param("child", "yes")
            y_pred, score = score_model(rf_model, X_test, y_test)
            mlflow.log_param("Score", score)
            logger.info(f"Score of RF model is {score}")

            logger.debug("================ End of ML Workflow ================")
