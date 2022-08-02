"""Script for data collection, ingestion and preprocessing.

Parameters
----------
output-path: string
    Output folder to store processed, train and testdownloaded data.
log-level: string
    Mention priority of logs according to severity.
log-path: string
    Full path to a log file, if logs are to be written to a file.
no-console-log: bool
    Whether or not to write logs to the console.
"""
import argparse
import configparser
import os
import tarfile

import mlflow
import numpy as np
import pandas as pd
from six.moves import urllib  # pyright: ignore
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedShuffleSplit

# Import custom logger
from houseValuePrediction import log_configurar

# Configure default logger
logger = log_configurar.configure_logger()

# Read configuration
config = configparser.ConfigParser()
config.read("setup.cfg")

# Variable Initialization
HOUSING_URL = "https://raw.githubusercontent.com/ageron/handson-ml/master/datasets/housing/housing.tgz"
HOUSING_PATH = "data/raw/"
HOUSING_STORE_PATH = "data/processed/"


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


def fetch_housing_data(housing_url=HOUSING_URL, housing_path=HOUSING_PATH):
    """Fetch housing data.

    Download dataset from given URL. and then store it in given path.

    Parameters
    ----------
    housing_url:string
        Dataset URL
    housing_path: string
        Path to store the new data

    Returns
    ------
        None    
    """
    os.makedirs(housing_path, exist_ok=True)
    tgz_path = os.path.join(housing_path, "housing.tgz")
    urllib.request.urlretrieve(housing_url, tgz_path)  # noqa
    housing_tgz = tarfile.open(tgz_path)
    housing_tgz.extractall(path=housing_path)
    housing_tgz.close()


def load_housing_data(housing_path=HOUSING_PATH):
    """Load housing data in Pandas Dataframe.

    Parameters
    ----------
    housing_path: string
        path where data is stored

    Returns
    -------
    pandas.DataFrame
        dataframe loaded with data from csv file
    """
    csv_path = os.path.join(housing_path, "housing.csv")
    return pd.read_csv(csv_path)


def data_preprocessing():
    """Run data preprocessing step.

    Parameters
    ----------
    None

    Returns
    -------
    pandas.DataFrame
        Preprocessed dataframe
    """

    # Fetch and Load data
    logger.debug("Fetching and loading dataset")
    fetch_housing_data()
    housing_data = load_housing_data()

    # divide data into target and label category
    housing_targets = housing_data.drop("median_house_value", axis=1)
    housing_labels = housing_data["median_house_value"].copy()

    # handling missing data using imputation
    logger.debug("Handling missing values.")
    imputer = SimpleImputer(strategy="median")
    housing_subset_data = housing_targets.drop("ocean_proximity", axis=1)
    imputer.fit(housing_subset_data)
    X = imputer.transform(housing_subset_data)

    # data preprocessing step
    logger.debug("Preprocessing the dataset.")
    housing_not_null = pd.DataFrame(X, columns=housing_subset_data.columns, index=housing_targets.index)
    housing_not_null["rooms_per_household"] = housing_not_null["total_rooms"] / housing_not_null["households"]
    housing_not_null["bedrooms_per_room"] = housing_not_null["total_bedrooms"] / housing_not_null["total_rooms"]
    housing_not_null["population_per_household"] = housing_not_null["population"] / housing_not_null["households"]
    housing_processed = housing_not_null.join(pd.get_dummies(housing_targets[["ocean_proximity"]], drop_first=True))
    housing_processed = housing_processed.join(housing_labels)

    return housing_processed


def save_train_test_data(
    train_data, test_data, store_path, train_csv_name="train_data.csv", test_csv_name="test_data.csv"
):
    """Save given pandas.DataFrame in given storage path with file_name

    Parameters
    ----------
    train_data: pandas.DataFrame
        preprocessed train dataframe to store.
    test_data: pandas.DataFrame
        preprocessed test dataframe to store.
    store_path: string
        path where dataframe will get stored.
    train_csv_name: string
        name which will get assigned to stored train file.
    test_csv_name: string
        name which will get assigned to stored test file.

    Returns
    -------
    None
    """
    os.makedirs(store_path, exist_ok=True)
    train_data.to_csv(os.path.join(store_path, train_csv_name), index=False)
    test_data.to_csv(os.path.join(store_path, test_csv_name), index=False)
    return


def split_train_test_data(output_path=None):
    """Split preprocessed dataset into train-test and target-label section.

    Parameters
    ----------
    output_path: string
        Output folder to store processed, train and testdownloaded data.        

    Returns
    -------
    pandas.DataFrame
        training dataset
    pandas.DataFrame 
        testing dataset
    """
    processed_data = data_preprocessing()

    # Split dataset into train and test using stratifed shuffle technique
    logger.debug("Spliting dataset into train and test set using stratifed shuffle technique.")
    split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    processed_data["income_cat"] = pd.cut(
        processed_data["median_income"], bins=[0.0, 1.5, 3.0, 4.5, 6.0, np.inf], labels=[1, 2, 3, 4, 5],
    )
    for train_index, test_index in split.split(processed_data, processed_data["income_cat"]):
        train_set = processed_data.loc[train_index]
        test_set = processed_data.loc[test_index]

    for set_ in (train_set, test_set):
        set_.drop("income_cat", axis=1, inplace=True)

    # storing processed dataset
    store_path = output_path if output_path else config["params"]["OUTPUT_DATA_PROCESSED"]
    logger.info(f'storing train and test processed dataset into path "{store_path}"')
    save_train_test_data(train_data=train_set, test_data=test_set, store_path=store_path)
    mlflow.log_artifacts(store_path)
    logger.debug("End of data collection and preprocessing step.")
    return train_set, test_set


def data_collection_preprocessing(output_path=HOUSING_STORE_PATH):
    """Run ingest_data python script for all data preprocessing step.

    Parameters
    ----------
    output_path:
        Output folder to store processed, train and testdownloaded data.        
    """
    return split_train_test_data(output_path)


if __name__ == "__main__":
    args = get_args()
    HOUSING_URL = config["params"]["HOUSING_URL"]
    HOUSING_PATH = config["params"]["OUTPUT_DATA_RAW"]
    HOUSING_STORE_PATH = config["params"]["OUTPUT_DATA_PROCESSED"]

    # Configure logger
    logger = log_configurar.configure_logger(
        log_file=args.log_path, console=args.no_console_log, log_level=args.log_level,
    )

    logger.debug("Start Data Collection & Preprocessing Phase =======")
    data_collection_preprocessing()
    logger.debug("End Data Collection & Preprocessing Phase =========")
